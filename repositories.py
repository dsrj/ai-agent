"""Import pinned, curated source files from the five approved repositories using GET only."""

import base64
import logging
import os
import re
import tempfile
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any

import httpx

from cloudops_hub import announcements, knowledge_folder
from cloudops_hub.catalog import SOURCE_REPOSITORIES, sources
from cloudops_hub.catalog_guide import ingest_guide
from cloudops_hub.config import get_settings
from cloudops_hub.firewall_guide import ingest_firewall_guide
from cloudops_hub.glossary_review import import_guide_aliases, import_keywords
from cloudops_hub.ingest import ingest

MAX_FILE_BYTES = 1_000_000
MAX_SNAPSHOT_BYTES = 3_000_000
logger = logging.getLogger(__name__)
LITERAL_CREDENTIAL = re.compile(
    r"""(?im)(?:authorization|password|client_secret|api_key|access_token|private_key)"""
    r"""["']?\s*[:=]\s*["']"""
    r"""(?!(?:Bearer|Basic|token)\s+\$\{(?:var|local|data)\.[a-z0-9_.]+\}["'])"""
    r"""(?!\$\{|var\.|local\.|data\.|https?://|null["'])"""
    r"""[^"'\r\n]{12,}["']|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"""
)


class ReadOnlyCatalogTransport(httpx.BaseTransport):
    """Reject writes and non-manifest URLs before any request reaches the network."""

    def __init__(self) -> None:
        entries = sources()
        if {item["repository"] for item in entries} != set(SOURCE_REPOSITORIES):
            raise ValueError("Catalog manifest must contain exactly the five approved repositories")
        self.allowed_urls = frozenset(
            httpx.URL(
                f"https://api.github.com/repos/{entry['repository']}/contents/{relative}",
                params={"ref": entry["commit"]},
            )
            for entry in entries
            for relative in entry["files"]
        )
        self.transport = httpx.HTTPTransport()

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        if (
            request.method != "GET"
            or request.read()
            or (
                request.url not in self.allowed_urls
                and not knowledge_folder.is_allowed(request.url)
            )
        ):
            logger.error("Blocked prohibited catalog request by the read-only transport")
            raise RuntimeError(
                "Catalog access is read-only: only GET of approved pinned files is permitted"
            )
        return self.transport.handle_request(request)

    def close(self) -> None:
        self.transport.close()


def fetch_snapshots(client: httpx.Client, destination: Path) -> list[dict[str, Any]]:
    entries = sources()
    if {item["repository"] for item in entries} != set(SOURCE_REPOSITORIES):
        raise ValueError("Catalog manifest must contain exactly the five approved repositories")
    total = 0
    snapshots = []
    for entry in entries:
        repository = entry["repository"]
        commit = entry["commit"]
        if not re.fullmatch(r"[a-f0-9]{40}", commit):
            raise ValueError("Catalog source must be pinned to a full commit SHA")
        source_dir = destination / repository.split("/")[1]
        metadata = {}
        if not entry["files"] or len(set(entry["files"])) != len(entry["files"]):
            raise ValueError("Catalog manifest must have a nonempty unique file list")
        operations = entry.get("operations_files", [])
        if (
            not isinstance(operations, list)
            or any(not isinstance(path, str) or path not in entry["files"] for path in operations)
            or len(set(operations)) != len(operations)
        ):
            raise ValueError("Catalog operations guides must be unique approved source files")
        for relative in entry["files"]:
            path = PurePosixPath(relative)
            if (
                path.is_absolute()
                or ".." in path.parts
                or not re.fullmatch(r"[A-Za-z0-9_./ -]+", relative)
                or path.suffix.casefold() not in {".md", ".tf", ".json"}
            ):
                raise ValueError("Catalog manifest contains an unsupported file path")
            response = client.get(
                f"https://api.github.com/repos/{repository}/contents/{relative}",
                params={"ref": commit},
            )
            if response.status_code != 200:
                raise RuntimeError(
                    f"Catalog read failed for {repository}/{relative}: "
                    f"HTTP {response.status_code}. Check read access and the pinned commit."
                )
            payload = response.json()
            if (
                payload.get("type") != "file"
                or payload.get("encoding") != "base64"
                or not 0 < payload.get("size", 0) <= MAX_FILE_BYTES
            ):
                raise ValueError(f"Catalog file is not supported or exceeds limits: {relative}")
            content = base64.b64decode(payload["content"].replace("\n", ""), validate=True)
            total += len(content)
            if len(content) > MAX_FILE_BYTES or total > MAX_SNAPSHOT_BYTES:
                raise ValueError("Catalog snapshot exceeds the import size limit")
            text = content.decode("utf-8")
            if LITERAL_CREDENTIAL.search(text):
                raise ValueError(
                    f"Potential literal credential in {repository}/{relative}; "
                    "remove that file from the reviewed manifest before importing."
                )
            file = source_dir.joinpath(*path.parts)
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text(text, encoding="utf-8")
            metadata[relative] = {
                "title": f"{repository}: {relative} ({entry['purpose']})",
                "url": f"https://github.com/{repository}/blob/{commit}/{relative}",
                "assistant": "landing-zone",
            }
        snapshots.append({**entry, "directory": source_dir, "metadata": metadata})
    return snapshots


def _client(token: str) -> httpx.Client:
    return httpx.Client(
        transport=ReadOnlyCatalogTransport(),
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        timeout=30,
        follow_redirects=False,
    )


def _token() -> str:
    token = os.environ.get("CATALOG_READ_TOKEN")
    if not token:
        raise RuntimeError(
            "CATALOG_READ_TOKEN is required; grant Contents: read-only on the five approved "
            "repositories. The default workflow token cannot read other private repositories."
        )
    return token


def glossary_main(*, before_ingest: Callable[[], None] | None = None) -> None:
    """Refresh only reviewed glossary terms at one main commit; never index reports."""
    import json

    from cloudops_hub.glossary_review import import_reviewed_glossary
    from cloudops_hub.knowledge import MAX_GLOSSARY_BYTES

    with _client(_token()) as client:
        sha = str(knowledge_folder._get(
            client, f"{knowledge_folder.API}/commits/{knowledge_folder.REF}"
        ).get("sha", ""))
        if not knowledge_folder.SHA.fullmatch(sha):
            raise ValueError("GitHub returned an invalid glossary commit SHA")
        payload = knowledge_folder._get(
            client, f"{knowledge_folder.API}/contents/{knowledge_folder.GLOSSARY_PATH}", ref=sha,
        )
        if (
            payload.get("type") != "file" or payload.get("encoding") != "base64"
            or not 0 < payload.get("size", 0) <= MAX_GLOSSARY_BYTES
        ):
            raise ValueError("Invalid glossary file or size exceeds 512 KiB")
        raw = base64.b64decode(payload["content"].replace("\n", ""), validate=True)
        if len(raw) > MAX_GLOSSARY_BYTES:
            raise ValueError("Glossary exceeds 512 KiB")
        text = raw.decode("utf-8")
        if LITERAL_CREDENTIAL.search(text):
            raise ValueError("Potential literal credential in glossary; import refused")
        dictionary = json.loads(text)
    import_reviewed_glossary(get_settings(), dictionary, sha, before_write=before_ingest)


def knowledge_main(
    *,
    before_ingest: Callable[[], None] | None = None,
    announcement_sink: Callable[[dict], None] | None = None,
) -> None:
    """Re-import only the CloudOps Hub knowledge folder at the release head."""
    token = _token()
    with tempfile.TemporaryDirectory(prefix="cloudops-knowledge-") as directory:
        general_directory = Path(directory) / "general"
        platform_directory = Path(directory) / "platform"
        with _client(token) as client:
            sha, metadata = knowledge_folder.fetch(
                client, general_directory, LITERAL_CREDENTIAL,
                keyword_sink=lambda manifest: import_keywords(
                    get_settings(), manifest, before_write=before_ingest
                ),
                alias_sink=lambda aliases: import_guide_aliases(
                    get_settings(), aliases, before_write=before_ingest
                ),
            )
            platform_sha, platform_metadata = knowledge_folder.fetch(
                client, platform_directory, LITERAL_CREDENTIAL, platform=True
            )
            feed = (
                announcements.fetch(client, sha, LITERAL_CREDENTIAL) if announcement_sink else None
            )
        knowledge_folder.ingest_knowledge_folder(
            general_directory, sha, metadata, before_write=before_ingest
        )
        knowledge_folder.ingest_knowledge_folder(
            platform_directory,
            platform_sha,
            platform_metadata,
            before_write=before_ingest,
            platform=True,
        )
        if announcement_sink:
            announcement_sink(feed)
    print(
        f"Imported {len(metadata)} general knowledge documents at {sha[:12]} and "
        f"{len(platform_metadata)} restricted platform documents at {platform_sha[:12]} "
        f"from {knowledge_folder.REF}, into separate indexes."
    )


def main(
    *,
    before_ingest: Callable[[], None] | None = None,
    announcement_sink: Callable[[dict], None] | None = None,
) -> None:
    token = _token()
    with tempfile.TemporaryDirectory(prefix="cloudops-catalog-") as directory:
        with _client(token) as client:
            snapshots = fetch_snapshots(client, Path(directory))
            knowledge_directory = Path(directory) / "knowledge-folder"
            knowledge_directory.mkdir()
            knowledge_sha, knowledge_metadata = knowledge_folder.fetch(
                client, knowledge_directory, LITERAL_CREDENTIAL,
                keyword_sink=lambda manifest: import_keywords(
                    get_settings(), manifest, before_write=before_ingest
                ),
                alias_sink=lambda aliases: import_guide_aliases(
                    get_settings(), aliases, before_write=before_ingest
                ),
            )
            platform_directory = Path(directory) / "platform-folder"
            platform_sha, platform_metadata = knowledge_folder.fetch(
                client, platform_directory, LITERAL_CREDENTIAL, platform=True
            )
            feed = (
                announcements.fetch(client, knowledge_sha, LITERAL_CREDENTIAL)
                if announcement_sink
                else None
            )
        ingest_guide(before_write=before_ingest)
        ingest_firewall_guide(before_write=before_ingest)
        knowledge_folder.ingest_knowledge_folder(
            knowledge_directory, knowledge_sha, knowledge_metadata, before_write=before_ingest
        )
        knowledge_folder.ingest_knowledge_folder(
            platform_directory,
            platform_sha,
            platform_metadata,
            before_write=before_ingest,
            platform=True,
        )
        for snapshot in snapshots:
            if snapshot.get("operations_files"):
                guide_directory = snapshot["directory"].with_name(
                    snapshot["directory"].name + "-operations"
                )
                guide_metadata = {}
                for relative in snapshot["operations_files"]:
                    destination = guide_directory / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    (snapshot["directory"] / relative).replace(destination)
                    guide_metadata[relative] = snapshot["metadata"][relative]
                ingest(
                    guide_directory,
                    source_id=f"guide:github:{snapshot['repository']}",
                    version=snapshot["commit"],
                    source_kind="operations",
                    assistant="landing-zone",
                    metadata=guide_metadata,
                    before_write=before_ingest,
                )
            if before_ingest:
                before_ingest()
            ingest(
                snapshot["directory"],
                source_id=f"github:{snapshot['repository']}",
                version=snapshot["commit"],
                source_kind="repository",
                assistant="landing-zone",
                metadata=snapshot["metadata"],
                before_write=before_ingest,
            )
        if announcement_sink:
            announcement_sink(feed)
    print(
        f"Imported the service-catalog and Firewall guides, {len(knowledge_metadata)} "
        f"general knowledge files, {len(platform_metadata)} restricted platform files "
        f"and {len(snapshots)} approved repository snapshots."
    )


if __name__ == "__main__":
    main()
