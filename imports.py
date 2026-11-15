"""Fixed-target Azure job control, available only to the Hub administrator role."""

import json
import logging
import re
from typing import Any
from urllib.parse import urlsplit

import httpx
from azure.core.exceptions import AzureError, ResourceNotFoundError
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient

from cloudops_hub.config import Settings

logger = logging.getLogger(__name__)
JOB_ID = re.compile(
    r"^/subscriptions/[0-9a-fA-F-]{36}/resourceGroups/[A-Za-z0-9_.()-]+"
    r"/providers/Microsoft\.App/jobs/[A-Za-z0-9-]+$"
)
API_VERSION = "2025-01-01"
FINISHED = {"Succeeded", "Failed", "Stopped"}
IMAGE_REPORT_BLOB = "reports/confluence-images.json"


def image_report(settings: Settings, credential) -> dict | None:
    if not settings.azure_storage_account_url:
        return None
    with BlobServiceClient(settings.azure_storage_account_url, credential=credential) as storage:
        blob = storage.get_blob_client("imports", IMAGE_REPORT_BLOB)
        try:
            raw = blob.download_blob().read(1_000_001)
        except ResourceNotFoundError:
            return None
        if len(raw) > 1_000_000:
            raise RuntimeError("Image import report exceeds the supported size")
        try:
            report = json.loads(raw)
        except (ValueError, UnicodeError):
            raise RuntimeError("Image import report is invalid") from None
        if (
            not isinstance(report, dict)
            or report.get("status") not in {"imported", "failed"}
            or not isinstance(report.get("finished_at"), str)
            or type(report.get("images_imported")) is not int
            or type(report.get("images_skipped")) is not int
            or not isinstance(report.get("failures"), list)
            or not all(
                isinstance(item, dict)
                and all(
                    isinstance(item.get(key), str)
                    for key in ("page_id", "page_url", "filename", "reason")
                )
                for item in report["failures"]
            )
        ):
            raise RuntimeError("Image import report is invalid")
        return report


class ImportBusy(RuntimeError):
    pass


def execution_records(client: httpx.Client, url: str) -> list[dict[str, str]]:
    endpoint = f"{url}/executions"
    next_url = endpoint
    visited = set()
    records = []
    for _ in range(10):
        if next_url in visited:
            raise RuntimeError("Import execution pagination repeated; no job was started")
        visited.add(next_url)
        response = client.get(next_url)
        if response.status_code != 200:
            raise RuntimeError(
                f"Import job status returned HTTP {response.status_code}; check job RBAC"
            )
        payload = response.json()
        executions = payload.get("value")
        if not isinstance(executions, list):
            raise RuntimeError("Import job returned an invalid execution list")
        for item in executions:
            properties = item.get("properties", {})
            record = {
                "name": item["name"],
                "status": properties.get("status", "Unknown"),
                "started_at": properties.get("startTime") or "",
                "finished_at": properties.get("endTime") or "",
            }
            if not all(isinstance(value, str) for value in record.values()):
                raise RuntimeError("Import job returned invalid execution metadata")
            records.append(record)
        next_url = payload.get("nextLink")
        if not next_url:
            return sorted(records, key=lambda item: item["started_at"], reverse=True)
        if not isinstance(next_url, str) or len(next_url) > 8192:
            raise RuntimeError("Import job returned invalid pagination")
        parsed = urlsplit(next_url)
        if (
            parsed.scheme != "https"
            or parsed.netloc != "management.azure.com"
            or parsed.path.casefold() != urlsplit(endpoint).path.casefold()
            or parsed.fragment
        ):
            raise RuntimeError("Import job returned pagination outside the configured job")
    raise RuntimeError("Import execution history exceeded the scan limit; no job was started")


def control_import(settings: Settings, source: str | None = None) -> dict[str, Any]:
    if source == "graph" and not settings.knowledge_graph_enabled:
        raise RuntimeError("Knowledge Hub integration is disabled by configuration")
    job_id = settings.import_job_id
    if not JOB_ID.fullmatch(job_id):
        raise RuntimeError("The Azure import job is not configured")
    if source is not None and source not in {
        "confluence",
        "confluence-images",
        "catalog",
        "knowledge",
        "glossary",
        "graph",
        "all",
    }:
        raise ValueError("Unsupported import source")
    url = f"https://management.azure.com{job_id}"
    credential = DefaultAzureCredential()
    try:
        token = credential.get_token("https://management.azure.com/.default").token
        with httpx.Client(
            headers={"Authorization": f"Bearer {token}"},
            params={"api-version": API_VERSION},
            timeout=30,
            follow_redirects=False,
        ) as client:
            records = execution_records(client, url)
            if source is None:
                report = image_report(settings, credential)
                return {
                    "enabled": True,
                    "knowledge_graph_enabled": settings.knowledge_graph_enabled,
                    "executions": records[:10],
                    **({"image_report": report} if report is not None else {}),
                }
            if any(record["status"] not in FINISHED for record in records):
                raise ImportBusy("An import is already active; wait for it to finish")
            response = client.get(url)
            if response.status_code != 200:
                raise RuntimeError(f"Import job configuration returned HTTP {response.status_code}")
            template = response.json()["properties"]["template"]
            containers = template.get("containers", [])
            if len(containers) != 1 or containers[0].get("name") != "importer":
                raise RuntimeError("Import job has an unexpected container configuration")
            if template.get("initContainers"):
                raise RuntimeError("Import job has unexpected initialization containers")
            container = {
                key: value
                for key, value in containers[0].items()
                if key in {"name", "image", "command", "args", "env", "resources"}
                and value is not None
            }
            env = [
                item for item in container.get("env", []) if item.get("name") != "REFRESH_SOURCE"
            ]
            container["env"] = [*env, {"name": "REFRESH_SOURCE", "value": source}]
            # Only a fixed source selector is overridden. No user-supplied commands, URLs or images.
            response = client.post(f"{url}/start", json={"containers": [container]})
            if response.status_code not in {200, 202}:
                raise RuntimeError(f"Import job start returned HTTP {response.status_code}")
            if response.status_code == 202 and not response.content:
                return {"status": "accepted", "execution": "pending", "source": source}
            result = response.json()
            name = result.get("name")
            if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9-]+", name):
                raise RuntimeError("Import job start did not return a valid execution name")
            logger.info("Administrator requested %s refresh; execution %s", source, name)
            return {"status": "started", "execution": name, "source": source}
    except (httpx.HTTPError, AzureError):
        logger.error("Azure import job control failed")
        raise RuntimeError(
            "Azure import job control is unavailable; check status before retrying a start"
        ) from None
    except (ValueError, KeyError, TypeError, AttributeError):
        logger.error("Azure import job returned an invalid response")
        raise RuntimeError(
            "Azure import job returned an invalid response; check status before retrying"
        ) from None
    finally:
        credential.close()


def control_keyword_scan(settings: Settings, start: bool = False) -> dict[str, Any]:
    """Read or start the separate scheduled glossary keyword scan job; no overrides."""
    job_id = settings.keyword_scan_job_id
    if not JOB_ID.fullmatch(job_id):
        raise RuntimeError("The keyword scan job is not configured")
    url = f"https://management.azure.com{job_id}"
    credential = DefaultAzureCredential()
    try:
        token = credential.get_token("https://management.azure.com/.default").token
        with httpx.Client(
            headers={"Authorization": f"Bearer {token}"},
            params={"api-version": API_VERSION},
            timeout=30,
            follow_redirects=False,
        ) as client:
            records = execution_records(client, url)
            if not start:
                return {"enabled": True, "executions": records[:10]}
            if any(record["status"] not in FINISHED for record in records):
                raise ImportBusy("A keyword scan is already running; wait for it to finish")
            response = client.get(url)
            if response.status_code != 200:
                raise RuntimeError(
                    f"Keyword scan job configuration returned HTTP {response.status_code}"
                )
            template = response.json()["properties"]["template"]
            containers = template.get("containers", [])
            if len(containers) != 1 or containers[0].get("name") != "keyword-scan":
                raise RuntimeError("Keyword scan job has an unexpected container configuration")
            if template.get("initContainers"):
                raise RuntimeError("Keyword scan job has unexpected initialization containers")
            container = {
                key: value
                for key, value in containers[0].items()
                if key in {"name", "image", "command", "args", "env", "resources"}
                and value is not None
            }
            response = client.post(f"{url}/start", json={"containers": [container]})
            if response.status_code not in {200, 202}:
                raise RuntimeError(f"Keyword scan job start returned HTTP {response.status_code}")
            if response.status_code == 202 and not response.content:
                return {"status": "accepted", "execution": "pending"}
            name = response.json().get("name")
            if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9-]+", name):
                raise RuntimeError("Keyword scan job start did not return a valid execution name")
            return {"status": "started", "execution": name}
    except (httpx.HTTPError, AzureError):
        logger.error("Keyword scan job control failed")
        raise RuntimeError(
            "Keyword scan job control is unavailable; check status before retrying a start"
        ) from None
    except (ValueError, KeyError, TypeError, AttributeError):
        logger.error("Keyword scan job returned an invalid response")
        raise RuntimeError(
            "Keyword scan job returned an invalid response; check status before retrying"
        ) from None
    finally:
        credential.close()
