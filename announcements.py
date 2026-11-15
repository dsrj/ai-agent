"""Validated, non-model announcement feed published only by the import identity."""

import base64
import json
import re
from datetime import UTC, datetime
from typing import Literal

from azure.core.exceptions import ResourceNotFoundError
from azure.storage.blob import BlobServiceClient
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from cloudops_hub.config import Settings

SOURCE_PATH = "announcements/cloudops.json"
FEED_BLOB = "published/announcements.json"
MAX_BYTES = 256_000


class Announcement(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{0,79}$")
    title: str = Field(min_length=1, max_length=200)
    status: Literal["upcoming", "update", "completed"]
    published_at: AwareDatetime
    starts_at: AwareDatetime | None = None
    ends_at: AwareDatetime | None = None
    pinned: bool = False
    audience: str = Field(default="All Hub users", min_length=1, max_length=200)
    services: list[str] = Field(default_factory=list, max_length=20)
    action_required: str = Field(default="", max_length=500)
    body: str = Field(min_length=1, max_length=12_000)

    @model_validator(mode="after")
    def valid_content(self):
        if not self.title.strip() or not self.body.strip():
            raise ValueError("Announcement title and body must not be blank")
        if any(not item.strip() or len(item) > 100 for item in self.services):
            raise ValueError("Announcement services must be nonempty and at most 100 characters")
        if self.ends_at and (not self.starts_at or self.ends_at < self.starts_at):
            raise ValueError("Announcement end must follow its start")
        return self


class AnnouncementSource(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    version: Literal[1]
    announcements: list[Announcement] = Field(max_length=100)

    @model_validator(mode="after")
    def unique_ids(self):
        ids = [item.id for item in self.announcements]
        if len(set(ids)) != len(ids):
            raise ValueError("Announcement IDs must be unique")
        return self


class PublishedAnnouncements(AnnouncementSource):
    source_commit: str = Field(pattern=r"^[a-f0-9]{40}$")
    refreshed_at: AwareDatetime


def fetch(client, sha: str, credential_pattern: re.Pattern) -> dict:
    from cloudops_hub.knowledge_folder import API

    if not re.fullmatch(r"[a-f0-9]{40}", sha):
        raise ValueError("Announcements require an immutable knowledge commit")
    response = client.get(f"{API}/contents/{SOURCE_PATH}", params={"ref": sha})
    if response.status_code != 200:
        raise RuntimeError(
            f"Announcement source read failed: HTTP {response.status_code}. "
            f"Check {SOURCE_PATH} and knowledge repository read access."
        )
    payload = response.json()
    if (
        payload.get("type") != "file"
        or payload.get("encoding") != "base64"
        or type(payload.get("size")) is not int
        or not 0 < payload["size"] <= MAX_BYTES
        or not isinstance(payload.get("content"), str)
        or len(payload["content"]) > MAX_BYTES * 2
    ):
        raise ValueError("Announcement source is not a supported bounded JSON file")
    raw = base64.b64decode(payload["content"].replace("\n", ""), validate=True)
    if len(raw) > MAX_BYTES or credential_pattern.search(raw.decode("utf-8")):
        raise ValueError("Announcement source exceeds limits or contains a potential credential")
    source = AnnouncementSource.model_validate_json(raw)
    return {
        **source.model_dump(mode="json"),
        "source_commit": sha,
        "refreshed_at": datetime.now(UTC).isoformat(),
    }


def read(settings: Settings, credential) -> dict:
    if not settings.azure_storage_account_url:
        raise RuntimeError("Announcement storage is not configured")
    with BlobServiceClient(settings.azure_storage_account_url, credential=credential) as storage:
        try:
            raw = storage.get_blob_client("imports", FEED_BLOB).download_blob().read(MAX_BYTES + 1)
        except ResourceNotFoundError:
            return {"status": "not_published", "announcements": []}
    if len(raw) > MAX_BYTES:
        raise RuntimeError("Published announcement feed exceeds the size limit")
    try:
        feed = PublishedAnnouncements.model_validate_json(raw)
    except ValueError:
        raise RuntimeError("Published announcement feed is invalid") from None
    return {"status": "published", **feed.model_dump(mode="json")}


def publish(storage, feed: dict, before_write) -> None:
    validated = PublishedAnnouncements.model_validate_json(json.dumps(feed))
    raw = validated.model_dump_json().encode("utf-8")
    if len(raw) > MAX_BYTES:
        raise ValueError("Published announcement feed exceeds the size limit")
    before_write()
    storage.get_blob_client("imports", FEED_BLOB).upload_blob(raw, overwrite=True)
