"""On-demand Microsoft permission reference refresh; no tenant Graph access."""

import logging
import re
from datetime import UTC, datetime

import httpx
from fastapi import HTTPException

SOURCE = (
    "https://raw.githubusercontent.com/microsoftgraph/"
    "microsoft-graph-docs-contrib/main/concepts/permissions-reference.md"
)
RECORD = "catalog/graph-permissions.json"
logger = logging.getLogger(__name__)


def parse_permissions(markdown: str) -> list[dict]:
    permissions = []
    for section in re.split(r"^### ", markdown, flags=re.MULTILINE)[1:]:
        name, _, body = section.partition("\n")
        name = name.strip()
        fields = {}
        for line in body.splitlines():
            cells = [cell.strip() for cell in line.split("|")[1:-1]]
            if len(cells) == 3:
                fields[cells[0]] = cells[1:]
        if "Identifier" not in fields:
            continue
        for index, kind in enumerate(("Application", "Delegated")):
            identifier = fields["Identifier"][index]
            if identifier == "-":
                continue
            if not re.fullmatch(r"[a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12}", identifier):
                raise ValueError(f"Invalid permission ID for {name} ({kind})")
            consent = fields.get("AdminConsentRequired", ["", ""])[index]
            if consent not in {"Yes", "No"}:
                raise ValueError(f"Unknown consent requirement for {name} ({kind})")
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
                raise ValueError("Invalid permission name")
            permissions.append(
                {"name": name, "type": kind, "id": identifier, "adminConsent": consent == "Yes"}
            )
    if len(permissions) < 500 or len({(p["name"], p["type"]) for p in permissions}) != len(
        permissions
    ):
        raise ValueError("Incomplete or duplicate Microsoft Graph permissions")
    return sorted(permissions, key=lambda item: (item["name"].lower(), item["type"]))


def download_permissions() -> list[dict]:
    with httpx.stream("GET", SOURCE, timeout=60, follow_redirects=False) as response:
        response.raise_for_status()
        content = bytearray()
        for chunk in response.iter_bytes():
            content.extend(chunk)
            if len(content) > 5_000_000:
                raise ValueError("Microsoft permissions reference exceeds the size limit")
    return parse_permissions(content.decode("utf-8"))


def refresh_permissions(store) -> dict:
    try:
        permissions = download_permissions()
    except (httpx.HTTPError, ValueError, UnicodeError) as exc:
        logger.warning("Microsoft Graph permission refresh rejected (%s)", type(exc).__name__)
        raise HTTPException(
            status_code=502,
            detail="Could not refresh Microsoft's permission reference. "
            "The previous list is unchanged. Try again or contact CloudOps.",
        ) from exc
    snapshot = {
        "source": SOURCE,
        "retrievedAt": datetime.now(UTC).isoformat(),
        "permissions": permissions,
    }
    if not store.write(RECORD, snapshot):
        raise HTTPException(status_code=503, detail="Permission refresh could not be saved.")
    return snapshot
