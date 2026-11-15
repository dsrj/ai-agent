"""Manual Azure job entrypoint. Source credentials are never available to the chat API."""

import json
import logging
import os
import threading
from contextlib import AbstractContextManager

from azure.core.exceptions import (
    AzureError,
    HttpResponseError,
    ResourceExistsError,
    ResourceNotFoundError,
)
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from azure.storage.blob import BlobServiceClient

from cloudops_hub import announcements, confluence, github_app, knowledge_graph, repositories
from cloudops_hub.config import get_settings
from cloudops_hub.imports import IMAGE_REPORT_BLOB

logger = logging.getLogger(__name__)


def _secret(secrets: SecretClient, name: str) -> str:
    try:
        value = secrets.get_secret(name).value
    except ResourceNotFoundError:
        raise RuntimeError(f"Required import credential {name} is missing from Key Vault") from None
    if not value:
        raise RuntimeError(f"Import credential {name} is empty")
    return value


class RefreshLease(AbstractContextManager):
    def __init__(self, blob):
        self.blob = blob
        self.stopped = threading.Event()
        self.failed = threading.Event()
        self.lease = None
        self.thread = None

    def __enter__(self):
        try:
            self.blob.upload_blob(b"refresh lock", overwrite=False)
        except ResourceExistsError:
            pass
        try:
            self.lease = self.blob.acquire_lease(lease_duration=60)
        except HttpResponseError as exc:
            if exc.status_code == 409:
                raise RuntimeError("Another refresh holds the import lock; retry later") from None
            raise
        self.thread = threading.Thread(target=self._renew, daemon=True)
        self.thread.start()
        return self

    def _renew(self):
        while not self.stopped.wait(20):
            try:
                self.lease.renew()
            except AzureError:
                self.failed.set()
                logger.error("Import lock renewal failed; indexing will be refused")
                return

    def ensure_active(self):
        if self.failed.is_set():
            raise RuntimeError("Import lock was lost; no further indexing is allowed")
        try:
            self.lease.renew()
        except AzureError:
            self.failed.set()
            raise RuntimeError("Import lock was lost; no further indexing is allowed") from None

    def __exit__(self, exc_type, exc_value, traceback):
        self.stopped.set()
        self.thread.join(timeout=30)
        try:
            self.lease.release()
        except AzureError:
            logger.error("Import lock release failed; it expires within 60 seconds")
            if exc_type is None:
                raise RuntimeError("Import completed but lock release failed") from None
        return False


def main() -> None:
    for name in ("httpx", "httpcore", "openai", "azure"):
        logging.getLogger(name).setLevel(logging.CRITICAL)
    source = os.environ.get("REFRESH_SOURCE")
    if source not in {
        "confluence", "confluence-images", "catalog", "knowledge", "glossary", "graph", "all"
    }:
        raise RuntimeError(
            "REFRESH_SOURCE must be confluence, confluence-images, catalog, knowledge, "
            "glossary, graph or all"
        )
    settings = get_settings()
    if source == "graph" and not settings.knowledge_graph_enabled:
        raise RuntimeError("Knowledge Hub integration is disabled by configuration")
    vault = os.environ.get("IMPORT_VAULT_URL", "")
    if not vault or not settings.azure_storage_account_url:
        raise RuntimeError("Import vault and storage account are required")
    credential = DefaultAzureCredential()
    try:
        with (
            BlobServiceClient(settings.azure_storage_account_url, credential=credential) as storage,
            SecretClient(vault_url=vault, credential=credential) as secrets,
            RefreshLease(storage.get_blob_client("imports", "refresh.lock")) as lease,
        ):

            def save_announcements(feed):
                announcements.publish(storage, feed, lease.ensure_active)

            # "all" refreshes the GitHub knowledge sources; Confluence is imported only on
            # explicit request and is excluded from answers unless re-enabled.
            selected = (
                ("catalog",)
                if source == "all"
                else ("confluence",)
                if source == "confluence-images"
                else (source,)
            )
            for kind in selected:
                names = (
                    {
                        "CONFLUENCE_EMAIL": "confluence-email",
                        "CONFLUENCE_API_TOKEN": "confluence-api-token",
                        "CONFLUENCE_CLOUD_ID": "confluence-cloud-id",
                    }
                    if kind == "confluence"
                    else {"CATALOG_READ_TOKEN": None}
                )
                try:
                    for env, secret_name in names.items():
                        # GitHub sources use a short-lived Read Sources app token.
                        os.environ[env] = (
                            _secret(secrets, secret_name)
                            if secret_name
                            else github_app.read_token(
                                lambda name: _secret(secrets, name),
                                settings.github_read_app_id,
                                settings.github_read_app_installation_id,
                            )
                        )
                    if kind == "confluence":
                        os.environ["CONFLUENCE_IMPORT_IMAGES"] = (
                            "true" if source == "confluence-images" else "false"
                        )
                        if source == "confluence-images":

                            def save_image_report(report):
                                lease.ensure_active()
                                storage.get_blob_client("imports", IMAGE_REPORT_BLOB).upload_blob(
                                    json.dumps(report).encode("utf-8"), overwrite=True
                                )

                            confluence.main(
                                before_ingest=lease.ensure_active,
                                image_report_sink=save_image_report,
                            )
                        else:
                            confluence.main(before_ingest=lease.ensure_active)
                    elif kind == "glossary":
                        repositories.glossary_main(before_ingest=lease.ensure_active)
                    elif kind == "knowledge":
                        repositories.knowledge_main(
                            before_ingest=lease.ensure_active,
                            announcement_sink=save_announcements,
                        )
                    elif kind == "graph":
                        knowledge_graph.main(before_ingest=lease.ensure_active)
                    else:
                        repositories.main(
                            before_ingest=lease.ensure_active,
                            announcement_sink=save_announcements,
                        )
                    logger.info("Completed %s cache refresh", kind)
                finally:
                    for env in names:
                        os.environ.pop(env, None)
                    os.environ.pop("CONFLUENCE_IMPORT_IMAGES", None)
    except AzureError:
        raise RuntimeError(
            "Azure import storage or Key Vault access failed; check the import identity's roles"
        ) from None
    finally:
        credential.close()


if __name__ == "__main__":
    main()
