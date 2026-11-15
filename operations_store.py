"""Private operational records; never used as model context or a search source."""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from cryptography.fernet import Fernet

from cloudops_hub.config import Settings

logger = logging.getLogger(__name__)

AUDIT_COLUMNS = {
    "category": "Category",
    "operation": "Operation",
    "outcome": "Outcome",
    "operation_id": "OperationId",
    "correlation_id": "CorrelationId",
    "ab_oid": "UserOid",
    "user_name": "UserName",
    "user_email": "UserEmail",
    "azp_oid": "AzpOid",
    "github_login": "GitHubLogin",
    "target": "Target",
}


def audit_row(event: dict) -> dict:
    row = {"TimeGenerated": event["timestamp"]}
    row.update({column: str(event.get(key) or "") for key, column in AUDIT_COLUMNS.items()})
    row["Category"] = row["Category"] or "azure"
    skip = {*AUDIT_COLUMNS, "timestamp"}
    row["Details"] = {key: value for key, value in event.items() if key not in skip}
    return row


def timestamp() -> str:
    return datetime.now(UTC).isoformat()


def owner_key(oid: str) -> str:
    return hashlib.sha256(oid.encode()).hexdigest()


class OperationsUnavailable(RuntimeError):
    pass


class FirewallAuditUnavailable(OperationsUnavailable):
    pass


class OperationsStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.credential = None
        self.container = None
        self.service = None
        self.ingestion = None
        self.shipper = None
        if settings.azure_storage_account_url:
            self.credential = DefaultAzureCredential()
            self.service = BlobServiceClient(
                settings.azure_storage_account_url, credential=self.credential
            )
            self.container = self.service.get_container_client("operations")
            if settings.audit_ingestion_endpoint and settings.audit_dcr_immutable_id:
                from azure.monitor.ingestion import LogsIngestionClient

                self.ingestion = LogsIngestionClient(
                    settings.audit_ingestion_endpoint, credential=self.credential
                )
                self.shipper = ThreadPoolExecutor(max_workers=1, thread_name_prefix="audit")

    def _container(self):
        if self.container is None:
            raise OperationsUnavailable("Private operations storage is not configured")
        return self.container

    def read(self, name: str) -> dict | None:
        try:
            return json.loads(self._container().get_blob_client(name).download_blob().readall())
        except ResourceNotFoundError:
            return None

    def write(self, name: str, value: dict, *, create: bool = False) -> bool:
        try:
            self._container().upload_blob(
                name,
                json.dumps(value).encode(),
                overwrite=not create,
                content_type="application/json",
            )
        except ResourceExistsError:
            if create:
                return False
            raise
        return True

    def delete(self, name: str) -> None:
        try:
            self._container().delete_blob(name)
        except ResourceNotFoundError:
            pass

    def list(self, prefix: str, *, limit: int = 100) -> list[dict]:
        records = []
        for blob in self._container().list_blobs(name_starts_with=prefix):
            if len(records) >= limit:
                break
            value = self.read(blob.name)
            if value is not None:
                records.append(value)
        return records

    def page(self, prefix: str, cursor: str | None = None) -> dict:
        pages = (
            self._container()
            .list_blobs(name_starts_with=prefix, results_per_page=50)
            .by_page(continuation_token=cursor)
        )
        page = next(pages, [])
        items = []
        for blob in page:
            value = self.read(blob.name)
            if value is not None:
                items.append(value)
        return {"items": items, "next_cursor": pages.continuation_token}

    def _cipher(self) -> Fernet:
        if len(self.settings.session_secret) < 32:
            raise OperationsUnavailable(
                "A production-strength session secret is required for GitHub login"
            )
        key = hashlib.sha256(("catalog-oauth:" + self.settings.session_secret).encode()).digest()
        return Fernet(base64.urlsafe_b64encode(key))

    def save_token(self, oid: str, connection: str, value: dict) -> None:
        encrypted = self._cipher().encrypt(json.dumps(value).encode()).decode()
        self.write(f"tokens/{owner_key(oid)}/{connection}.json", {"encrypted": encrypted})

    def token(self, oid: str, connection: str) -> dict | None:
        value = self.read(f"tokens/{owner_key(oid)}/{connection}.json")
        if value is None:
            return None
        return json.loads(self._cipher().decrypt(value["encrypted"].encode()))

    def audit(self, event: dict) -> None:
        day = datetime.now(UTC).strftime("%Y-%m-%d")
        reverse_time = 99_999_999_999_999_999 - int(datetime.now(UTC).timestamp() * 1_000_000)
        record = {**event, "timestamp": timestamp()}
        # The blob is the authoritative, fail-closed record; Log Analytics is a best-effort copy.
        self.write(f"audit/{day}/{reverse_time:017d}-{uuid.uuid4().hex}.json", record, create=True)
        if self.shipper is not None:
            self.shipper.submit(self._ship, record)

    def _ship(self, record: dict) -> None:
        try:
            self.ingestion.upload(
                rule_id=self.settings.audit_dcr_immutable_id,
                stream_name=self.settings.audit_stream_name,
                logs=[audit_row(record)],
            )
        except Exception as error:  # noqa: BLE001 - never fail the caller on telemetry
            logger.warning(
                "Audit copy to Log Analytics failed (%s); the blob record is kept",
                type(error).__name__,
            )

    def firewall_activity(self, records: list[dict]) -> None:
        """Content audit goes directly to restricted Log Analytics, never to Blob history."""
        if self.ingestion is None:
            raise FirewallAuditUnavailable("Firewall activity logging is not configured")
        try:
            self.ingestion.upload(
                rule_id=self.settings.audit_dcr_immutable_id,
                stream_name=self.settings.audit_stream_name,
                logs=[audit_row(record) for record in records],
            )
        except Exception as error:  # noqa: BLE001 - sanitize transport failures, fail closed
            logger.error("Firewall activity ingestion failed (%s)", type(error).__name__)
            raise FirewallAuditUnavailable("Firewall activity logging is unavailable") from None

    def close(self):
        if self.shipper:
            self.shipper.shutdown(wait=True)
        if self.ingestion:
            self.ingestion.close()
        if self.service:
            self.service.close()
        if self.credential:
            self.credential.close()
