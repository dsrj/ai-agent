"""Import general and restricted admin knowledge into separate indexes at main.

The repository's authoring contract (``_SCHEMA.md``) puts JSON-valued YAML frontmatter on
every document. Only ``guides/``, ``reference/`` and ``known-errors/`` with the matching
``document_type`` are indexed; run records, historical collections, legacy instructions and
assets are never indexed. The same folders under admin/ go only to the platform index.
Pending-review documents are indexed but marked unreviewed, so
approved and verified documents rank higher.
"""

import base64
import json
import logging
import re
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote

import httpx

from cloudops_hub.announcements import SOURCE_PATH
from cloudops_hub.answer_scope import contracts, presentations
from cloudops_hub.config import get_settings
from cloudops_hub.glossary_review import (
    MANIFEST_PATH,
    validate_guide_aliases,
    validate_manifest,
)
from cloudops_hub.ingest import ingest

logger = logging.getLogger(__name__)
REPOSITORY = "almbrand/cloudops-openai-hub-knowledge-source"
REF = "main"
GLOSSARY_PATH = "glossary/retrieval-synonyms.json"
SOURCE_ID = f"knowledge:github:{REPOSITORY}"
FOLDERS = {"guides": "guide", "reference": "reference", "known-errors": "known-error"}
MAX_FILES = 300
MAX_FILE_BYTES = 1_000_000
MAX_TOTAL_BYTES = 8_000_000
API = f"https://api.github.com/repos/{REPOSITORY}"
SHA = re.compile(r"[a-f0-9]{40}")
SAFE_PATH = re.compile(r"[\w./ ()&,+-]+")
FIELD = re.compile(r"([a-z_]+):\s*(.+)")
HUB_ASSISTANTS = {"landing-zone": "landing-zone", "service-catalog": "landing-zone"}


def is_allowed(url: httpx.URL) -> bool:
    """Allow only the read-only GitHub routes this importer needs."""
    if url.scheme != "https" or url.host != "api.github.com":
        return False
    path = url.path
    params = dict(url.params)
    if path == f"/repos/{REPOSITORY}/commits/{REF}":
        return not params
    tree = re.fullmatch(rf"/repos/{REPOSITORY}/git/trees/([a-f0-9]{{40}})", path)
    if tree:
        return params == {"recursive": "1"}
    prefix = f"/repos/{REPOSITORY}/contents/"
    return (
        path.startswith(prefix)
        and (
            _safe(path.removeprefix(prefix))
            or path.removeprefix(prefix) in {SOURCE_PATH, MANIFEST_PATH, GLOSSARY_PATH}
        )
        and set(params) == {"ref"}
        and bool(SHA.fullmatch(params["ref"]))
    )


def _safe(relative: str) -> bool:
    path = PurePosixPath(relative)
    parts = path.parts[1:] if path.parts[:1] == ("admin",) else path.parts
    return (
        bool(SAFE_PATH.fullmatch(relative))
        and not path.is_absolute()
        and ".." not in path.parts
        and len(parts) > 1
        and parts[0] in FOLDERS
        and path.suffix.casefold() == ".md"
        and not path.name.startswith("_")
    )


def front_matter(text: str) -> tuple[dict, str]:
    """Parse the schema's JSON-valued frontmatter; return (fields, body without it)."""
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        raise ValueError("missing frontmatter")
    fields: dict = {}
    for number, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            return fields, "".join(lines[number + 1 :]).lstrip("\n")
        match = FIELD.fullmatch(line.strip())
        if not match or match[1] in fields:
            raise ValueError("invalid or duplicate frontmatter field")
        try:
            fields[match[1]] = json.loads(match[2])
        except ValueError:
            raise ValueError(f"frontmatter field {match[1]} is not JSON") from None
    raise ValueError("unterminated frontmatter")


def hub_assistant(assistants: object) -> str:
    """Map the repository's assistant list onto the Hub index's single assistant scope."""
    if not isinstance(assistants, list) or not assistants:
        raise ValueError("assistants must be a nonempty list")
    scopes = set()
    for item in assistants:
        if item == "firewall":
            scopes.add("firewall")
        elif item in HUB_ASSISTANTS:
            scopes.add(HUB_ASSISTANTS[item])
        else:
            raise ValueError(f"unsupported assistant {item!r}")
    return scopes.pop() if len(scopes) == 1 else "shared"


def document_metadata(relative: str, fields: dict) -> dict[str, str] | None:
    """Return index metadata, or None when the document must not be indexed."""
    parts = PurePosixPath(relative).parts
    platform = parts[0] == "admin"
    folder = parts[1] if platform else parts[0]
    if fields.get("document_type") != FOLDERS[folder]:
        return None
    if fields.get("review_status") == "rejected":
        return None
    for key in ("id", "title"):
        if not isinstance(fields.get(key), str) or not fields[key].strip():
            raise ValueError(f"frontmatter {key} is required")
    if platform:
        if fields.get("assistants") != ["platform-admin"]:
            raise ValueError("Admin documents must target only platform-admin")
        assistant = "platform-admin"
    else:
        assistant = hub_assistant(fields.get("assistants"))
    approved = (
        fields.get("review_status") == "approved"
        and fields.get("verification_status") == "verified"
        and bool(fields.get("reviewed_by"))
        and bool(fields.get("reviewed_at"))
    )
    return {
        "title": fields["title"].strip()[:250],
        "assistant": assistant,
        "doc_id": fields["id"].strip()[:200],
        "document_type": FOLDERS[folder],
        "review_status": "approved" if approved else "unreviewed",
        **(
            {"last_updated": fields["reviewed_at"]}
            if approved and isinstance(fields["reviewed_at"], str)
            else {}
        ),
    }


def _get(client: httpx.Client, url: str, **params: str) -> dict:
    response = client.get(url, params=params or None)
    if response.status_code != 200:
        raise RuntimeError(
            f"Knowledge repository read failed: HTTP {response.status_code}. "
            f"Check read access to {REPOSITORY}."
        )
    return response.json()


def fetch(
    client: httpx.Client,
    destination: Path,
    credential_pattern: re.Pattern,
    *,
    platform: bool = False,
    keyword_sink: Callable[[dict], None] | None = None,
    alias_sink: Callable[[list[dict]], None] | None = None,
) -> tuple[str, dict[str, dict[str, Any]]]:
    sha = str(_get(client, f"{API}/commits/{REF}").get("sha", ""))
    if not SHA.fullmatch(sha):
        raise ValueError("GitHub returned an invalid knowledge commit SHA")
    tree = _get(client, f"{API}/git/trees/{sha}", recursive="1")
    if tree.get("truncated") or not isinstance(tree.get("tree"), list):
        raise ValueError("GitHub returned an incomplete repository tree")
    keyword_documents = {}
    if any(
        isinstance(item, dict) and item.get("type") == "blob"
        and item.get("path") == MANIFEST_PATH for item in tree["tree"]
    ):
        payload = _get(client, f"{API}/contents/{MANIFEST_PATH}", ref=sha)
        if (
            payload.get("type") != "file" or payload.get("encoding") != "base64"
            or not 0 < payload.get("size", 0) <= MAX_FILE_BYTES
        ):
            raise ValueError("Invalid glossary keyword migration file")
        raw = base64.b64decode(payload["content"].replace("\n", ""), validate=True)
        if len(raw) > MAX_FILE_BYTES or credential_pattern.search(raw.decode("utf-8")):
            raise ValueError("Glossary keyword migration exceeds limits or contains credentials")
        manifest = validate_manifest(json.loads(raw))
        if keyword_sink:
            keyword_sink(manifest)
        keyword_documents = {doc["path"]: doc for doc in manifest["documents"]}
    prefixes = {("admin", name) if platform else (name,) for name in FOLDERS}
    depth = 2 if platform else 1
    files = sorted(
        str(item["path"])
        for item in tree["tree"]
        if isinstance(item, dict)
        and item.get("type") == "blob"
        and PurePosixPath(str(item.get("path", ""))).parts[:depth] in prefixes
        and str(item["path"]).casefold().endswith(".md")
        and not PurePosixPath(str(item["path"])).name.startswith("_")
    )
    if len(files) > MAX_FILES:
        raise ValueError(f"Knowledge repository has more than {MAX_FILES} Markdown files")
    total = 0
    metadata = {}
    aliases = []
    answer_ids = set()
    answer_routes = set()
    duplicate_presentations = []
    for relative in files:
        if not _safe(relative):
            raise ValueError(f"Unsupported knowledge file name: {relative}")
        payload = _get(client, f"{API}/contents/{quote(relative)}", ref=sha)
        if (
            payload.get("type") != "file"
            or payload.get("encoding") != "base64"
            or not 0 < payload.get("size", 0) <= MAX_FILE_BYTES
        ):
            raise ValueError(f"Knowledge file is not supported or exceeds limits: {relative}")
        content = base64.b64decode(payload["content"].replace("\n", ""), validate=True)
        total += len(content)
        if len(content) > MAX_FILE_BYTES or total > MAX_TOTAL_BYTES:
            raise ValueError("Knowledge repository exceeds the import size limit")
        text = content.decode("utf-8")
        if credential_pattern.search(text):
            raise ValueError(
                f"Potential literal credential in {REPOSITORY}/{relative}; "
                "remove it before importing."
            )
        try:
            fields, body = front_matter(text)
            fields_metadata = document_metadata(relative, fields)
        except ValueError as exc:
            raise ValueError(f"Invalid knowledge metadata in {relative}: {exc}") from None
        if fields_metadata is None:
            logger.info("Skipped non-indexable knowledge document %s", relative)
            continue
        answer_contracts = contracts(fields, body)
        answer_presentations = presentations(fields, body)
        if answer_contracts and fields_metadata["document_type"] not in {"guide", "reference"}:
            raise ValueError(f"Answer contracts require guidance: {relative}")
        if answer_presentations and fields_metadata["document_type"] not in {"guide", "reference"}:
            raise ValueError(f"Answer presentations require guidance: {relative}")
        duplicate_presentations.extend(
            (fields_metadata["assistant"], entry["canonical_answer_id"], relative)
            for entry in answer_presentations if entry["disposition"] == "duplicate"
        )
        for entry in answer_contracts:
            identity = (fields_metadata["assistant"], entry["id"])
            route = (fields_metadata["assistant"], entry["resource"], entry["task"])
            if identity in answer_ids or route in answer_routes:
                raise ValueError(f"Duplicate answer identity or route: {relative}")
            answer_ids.add(identity)
            answer_routes.add(route)
        if "retrieval_aliases" in fields:
            if (
                platform or fields.get("assistants") != ["landing-zone"]
                or fields_metadata["document_type"] != "reference"
            ):
                raise ValueError(
                    f"Guide aliases require a landing-zone-only reference: {relative}"
                )
            guide_aliases = validate_guide_aliases(fields["retrieval_aliases"])
            if fields_metadata["review_status"] == "approved":
                aliases.extend(guide_aliases)
        migrated = keyword_documents.get(relative)
        if migrated:
            same_scope = migrated["assistants"] == fields["assistants"] or (
                not platform
                and hub_assistant(migrated["assistants"]) == fields_metadata["assistant"]
            )
            if migrated["id"] != fields["id"] or not same_scope:
                raise ValueError(f"Guide keyword identity or scope changed: {relative}")
            # Keep document-topic search coverage without keeping keyword lists in Markdown.
            if not any(line.startswith("**Keywords:**") for line in body.splitlines()):
                body = f"{body.rstrip()}\n\n{migrated['raw']}\n"
        file = destination.joinpath(*PurePosixPath(relative).parts)
        file.parent.mkdir(parents=True, exist_ok=True)
        # Frontmatter is metadata, never answer text.
        file.write_text(body, encoding="utf-8")
        metadata[relative] = {
            **fields_metadata,
            **({"answer_contracts": answer_contracts} if answer_contracts else {}),
            **({"answer_presentations": answer_presentations} if answer_presentations else {}),
            "url": f"https://github.com/{REPOSITORY}/blob/{sha}/{quote(relative)}",
        }
    for assistant, canonical_id, relative in duplicate_presentations:
        if (assistant, canonical_id) not in answer_ids:
            raise ValueError(f"Unknown canonical answer for duplicate presentation: {relative}")
    validate_guide_aliases(aliases)
    if alias_sink:
        alias_sink(aliases)
    return sha, metadata


def ingest_knowledge_folder(
    directory: Path,
    sha: str,
    metadata: dict[str, dict[str, Any]],
    *,
    before_write: Callable[[], None] | None = None,
    platform: bool = False,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    if before_write:
        before_write()
    ingest(
        directory,
        source_id=f"{SOURCE_ID}:admin" if platform else SOURCE_ID,
        version=sha,
        source_kind="operations",
        assistant="platform-admin" if platform else "shared",
        allow_empty=True,
        metadata=metadata,
        before_write=before_write,
        **({"settings": get_settings().for_assistant("platform-admin")} if platform else {}),
    )
