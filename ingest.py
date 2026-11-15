import argparse
import hashlib
import json
import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

from azure.core.exceptions import HttpResponseError, ResourceNotFoundError
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)

from cloudops_hub.answer_scope import metadata as scope_metadata
from cloudops_hub.answer_scope import presentation_metadata
from cloudops_hub.config import Settings, get_settings
from cloudops_hub.knowledge import embed_texts, identifiers
from cloudops_hub.sections import section_chunks, section_chunks_with_bodies

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
SUPPORTED_SUFFIXES = {".md", ".txt", ".json", ".tf", ".yml", ".yaml"}
CHUNK_CHARS = 2400


def make_chunks(text: str, max_chars: int = CHUNK_CHARS) -> list[str]:
    return [content for _, content in section_chunks(text, max_chars)]


def configure_enriched_index(index_client, index_name, settings, before_write=None) -> None:
    index = index_client.get_index(index_name)
    additions = [
        SimpleField(name="content_hash", type="Edm.String"),
        SimpleField(name="document_hash", type="Edm.String"),
        SearchField(name="section", type="Edm.String", searchable=True),
        SimpleField(name="chunk_number", type="Edm.Int32"),
        SimpleField(name="answer_section_body", type="Edm.String"),
        SimpleField(name="technical_terms", type="Collection(Edm.String)", filterable=True),
        SimpleField(name="doc_id", type="Edm.String", filterable=True),
        SimpleField(name="document_type", type="Edm.String", filterable=True),
        SimpleField(name="review_status", type="Edm.String", filterable=True),
        SimpleField(name="scope_version", type="Edm.Int32", filterable=True),
        SimpleField(name="answer_presentation_version", type="Edm.Int32", filterable=True),
        *[
            SimpleField(name=name, type="Edm.String", filterable=True)
            for name in ("answer_layout", "answer_disposition", "answer_canonical_id")
        ],
        *[
            SimpleField(name=name, type="Edm.String", filterable=True)
            for name in ("answer_id", "answer_resource", "answer_task", "answer_support")
        ],
        *[
            SimpleField(name=name, type="Collection(Edm.String)")
            for name in ("answer_aliases", "answer_task_aliases", "answer_conditions")
        ],
    ]
    if settings.azure_search_hybrid_enabled:
        signature = (
            f"cloudops-hub:{settings.azure_openai_embedding_deployment}:"
            f"{settings.azure_openai_embedding_dimensions}"
        )
        additions.append(
            SearchField(
                name="content_vector",
                type="Collection(Edm.Single)",
                searchable=True,
                vector_search_dimensions=settings.azure_openai_embedding_dimensions,
                vector_search_profile_name="hub-vector",
            )
        )
        existing = next((field for field in index.fields if field.name == "content_vector"), None)
        if (
            existing
            and existing.vector_search_dimensions != settings.azure_openai_embedding_dimensions
        ):
            raise RuntimeError(
                "Embedding dimensions changed; create a new index and fully reimport"
            )
        if existing and index.description != signature:
            raise RuntimeError(
                "Embedding deployment changed; create a new index and fully reimport"
            )
        index.description = signature
        index.vector_search = VectorSearch(
            algorithms=[HnswAlgorithmConfiguration(name="hub-hnsw")],
            profiles=[
                VectorSearchProfile(name="hub-vector", algorithm_configuration_name="hub-hnsw")
            ],
        )
    existing_names = {field.name for field in index.fields}
    index.fields.extend(field for field in additions if field.name not in existing_names)
    if settings.azure_search_semantic_enabled:
        index.semantic_search = SemanticSearch(
            configurations=[
                SemanticConfiguration(
                    name="hub-semantic",
                    prioritized_fields=SemanticPrioritizedFields(
                        title_field=SemanticField(field_name="title"),
                        content_fields=[
                            SemanticField(field_name="section"),
                            SemanticField(field_name="content"),
                        ],
                    ),
                )
            ]
        )
    if before_write:
        before_write()
    index_client.create_or_update_index(index)


def _ensure_index_once(
    index_client: SearchIndexClient,
    index_name: str,
    before_write: Callable[[], None] | None = None,
) -> None:
    try:
        index_client.get_index(index_name)
        return
    except ResourceNotFoundError:
        pass
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchField(name="content", type=SearchFieldDataType.String, searchable=True),
        SearchField(name="title", type=SearchFieldDataType.String, searchable=True),
        SimpleField(name="source_url", type=SearchFieldDataType.String, filterable=False),
        SimpleField(name="source_path", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="source_id", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="source_kind", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="assistant", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="version", type=SearchFieldDataType.String),
        SimpleField(name="last_updated", type=SearchFieldDataType.String),
    ]
    if before_write:
        before_write()
    index_client.create_index(SearchIndex(name=index_name, fields=fields))


def ensure_index(
    index_client: SearchIndexClient,
    index_name: str,
    before_write: Callable[[], None] | None = None,
) -> None:
    for attempt in range(12):
        try:
            _ensure_index_once(index_client, index_name, before_write)
            return
        except HttpResponseError as exc:
            if exc.status_code != 403 or attempt == 11:
                raise
            logger.warning("Waiting for the ingestion identity's Search role assignments")
            time.sleep(15)


def _document_id(source_id: str, relative_path: str, chunk_number: int) -> str:
    identity = f"{source_id}:{relative_path}:{chunk_number}".encode()
    return hashlib.sha256(identity).hexdigest()


def stale_document_ids(indexed_ids: set[str], current_ids: set[str]) -> set[str]:
    return indexed_ids - current_ids


def load_source_metadata(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8") as metadata_file:
        values = json.load(metadata_file)
    if not isinstance(values, dict):
        raise ValueError("Metadata file must map relative paths to metadata objects")
    allowed_fields = {
        "title",
        "url",
        "last_updated",
        "assistant",
        "doc_id",
        "document_type",
        "review_status",
    }
    for relative_path, fields in values.items():
        path_parts = PurePosixPath(relative_path).parts
        if PurePosixPath(relative_path).is_absolute() or ".." in path_parts:
            raise ValueError(f"Metadata path must be relative and contained: {relative_path}")
        if not isinstance(fields, dict) or set(fields) - allowed_fields:
            raise ValueError(f"Invalid metadata fields for {relative_path}")
        if any(not isinstance(value, str) for value in fields.values()):
            raise ValueError(f"Metadata values for {relative_path} must be strings")
    return values


def build_documents(
    source_dir: Path,
    source_id: str,
    version: str,
    source_kind: str,
    assistant: str = "shared",
    metadata: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    documents = []
    for path in sorted(source_dir.rglob("*")):
        if not path.is_file() or path.suffix.casefold() not in SUPPORTED_SUFFIXES:
            continue
        relative_path = path.relative_to(source_dir).as_posix()
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            logger.error(
                "Unable to read approved import %s (%s)",
                relative_path,
                type(exc).__name__,
            )
            raise RuntimeError(f"Incomplete import; failed to read {relative_path}") from exc
        title = next(
            (line.lstrip("# ").strip() for line in text.splitlines() if line.strip()), path.name
        )
        modified = datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat()
        document_metadata = (metadata or {}).get(relative_path, {})
        source_url = document_metadata.get("url", "")
        parsed_url = urlparse(source_url)
        if source_url and (
            parsed_url.scheme != "https"
            or not parsed_url.netloc
            or parsed_url.username
            or parsed_url.password
        ):
            raise ValueError(f"Source URL for {relative_path} must use HTTPS")
        document_title = document_metadata.get("title", title)[:250]
        last_updated = document_metadata.get("last_updated", modified)
        document_assistant = document_metadata.get("assistant", assistant)
        if document_assistant not in {"shared", "firewall", "landing-zone", "platform-admin"}:
            raise ValueError(f"Unsupported assistant metadata for {relative_path}")
        prepared = (
            f"```{path.suffix.lstrip('.')}\n{text}\n```"
            if path.suffix in {".json", ".tf", ".yaml", ".yml"}
            else text
        )
        # Smaller heading-aware chunks keep each passage on one topic for vector and
        # semantic ranking; every chunk repeats its heading path.
        for number, (section, content, original_body) in enumerate(
            section_chunks_with_bodies(prepared, CHUNK_CHARS), start=0
        ):
            documents.append(
                {
                    **scope_metadata(section, document_metadata.get("answer_contracts", [])),
                    **presentation_metadata(
                        section, document_metadata.get("answer_presentations", []),
                    ),
                    "id": _document_id(source_id, relative_path, number),
                    "content": content,
                    "answer_section_body": original_body,
                    "title": document_title,
                    "source_url": source_url,
                    "source_path": relative_path,
                    "source_id": source_id,
                    "source_kind": source_kind,
                    "assistant": document_assistant,
                    "version": version,
                    "last_updated": last_updated,
                    "section": section,
                    "chunk_number": number,
                    "technical_terms": identifiers(content, limit=512),
                    "doc_id": document_metadata.get("doc_id", f"{source_id}:{relative_path}"),
                    "document_type": document_metadata.get("document_type", source_kind),
                    "review_status": document_metadata.get("review_status", ""),
                }
            )
    return documents


def ingest(
    source_dir: Path,
    source_id: str,
    version: str | None = None,
    source_kind: str = "repository",
    assistant: str = "shared",
    allow_empty: bool = False,
    metadata: dict[str, dict[str, Any]] | None = None,
    before_write: Callable[[], None] | None = None,
    settings: Settings | None = None,
) -> int:
    settings = settings or get_settings()
    platform = assistant == "platform-admin"
    expected_index = settings.azure_search_platform_index.strip()
    if platform != (settings.azure_search_index == expected_index):
        raise ValueError("Platform documents must use only the separate platform index")
    if not settings.azure_search_endpoint:
        raise RuntimeError("AZURE_SEARCH_ENDPOINT is required to ingest documents")
    version = version or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    documents = build_documents(
        source_dir.resolve(), source_id, version, source_kind, assistant, metadata
    )
    if any((doc["assistant"] == "platform-admin") != platform for doc in documents):
        raise ValueError("Restricted and general documents cannot share an import")
    credential = DefaultAzureCredential()
    try:
        index_client = SearchIndexClient(settings.azure_search_endpoint, credential)
        ensure_index(index_client, settings.azure_search_index, before_write)
        client = SearchClient(
            settings.azure_search_endpoint, settings.azure_search_index, credential=credential
        )
        configure_enriched_index(index_client, settings.azure_search_index, settings, before_write)
        indexed_documents = {
            item["id"]: item
            for item in client.search(
                search_text="*",
                filter=f"source_id eq '{source_id.replace(chr(39), chr(39) * 2)}'",
                select=["id", "content_hash", "document_hash"],
            )
        }
        changed = []
        metadata_only = []
        signature = (
            f"{settings.azure_openai_embedding_deployment}:"
            f"{settings.azure_openai_embedding_dimensions}"
            if settings.azure_search_hybrid_enabled
            else "keyword"
        )
        for doc in documents:
            embedding_text = f"{doc['title']}\n{doc['source_path']}\n{doc['content']}"
            doc["content_hash"] = hashlib.sha256(
                f"{signature}\n{embedding_text}".encode()
            ).hexdigest()
            doc["document_hash"] = hashlib.sha256(
                json.dumps(doc, sort_keys=True).encode()
            ).hexdigest()
            previous = indexed_documents.get(doc["id"], {})
            if previous.get("content_hash") != doc["content_hash"]:
                changed.append(doc)
            elif previous.get("document_hash") != doc["document_hash"]:
                metadata_only.append(doc)
        if settings.azure_search_hybrid_enabled and changed:
            from cloudops_hub.retrieval import create_openai_client

            with create_openai_client(settings, credential) as ai:
                for start in range(0, len(changed), 16):
                    batch = changed[start : start + 16]
                    vectors = embed_texts(
                        ai,
                        settings,
                        [
                            f"{doc['title']}\n{doc['source_path']}\n{doc['content']}"
                            for doc in batch
                        ],
                        rate_limit_retries=5,
                    )
                    for doc, vector in zip(batch, vectors, strict=True):
                        doc["content_vector"] = vector
        if changed:
            for start in range(0, len(changed), 100):
                if before_write:
                    before_write()
                result = client.upload_documents(changed[start : start + 100])
                failures = [item for item in result if not item.succeeded]
                if failures:
                    raise RuntimeError(f"Azure AI Search rejected {len(failures)} imported chunks")
        for start in range(0, len(metadata_only), 100):
            if before_write:
                before_write()
            result = client.merge_documents(metadata_only[start : start + 100])
            if any(not item.succeeded for item in result):
                raise RuntimeError("Azure AI Search rejected chunk metadata updates")
        indexed = set(indexed_documents)
        if not documents and indexed and not allow_empty:
            raise RuntimeError(
                "Refusing to delete an existing source with zero imported chunks; "
                "use --allow-empty to confirm"
            )
        current = {item["id"] for item in documents}
        stale = stale_document_ids(indexed, current)
        if stale:
            if before_write:
                before_write()
            deletion_results = client.delete_documents(documents=[{"id": item} for item in stale])
            failures = [item for item in deletion_results if not item.succeeded]
            if failures:
                raise RuntimeError(f"Azure AI Search failed to delete {len(failures)} stale chunks")
        logger.info(
            "Imported %d chunks for source %s; removed %d stale chunks; "
            "content_changed=%d metadata_changed=%d unchanged=%d",
            len(documents),
            source_id,
            len(stale),
            len(changed),
            len(metadata_only),
            len(documents) - len(changed) - len(metadata_only),
        )
        return len(documents)
    finally:
        credential.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import approved CloudOps Hub documentation into Azure AI Search"
    )
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--version", help="Repository commit or export version identifier")
    parser.add_argument(
        "--source-kind",
        choices=("repository", "operations"),
        default="repository",
    )
    parser.add_argument(
        "--assistant", choices=("shared", "firewall", "landing-zone"), default="shared"
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Confirm deleting all indexed documents for an empty source",
    )
    parser.add_argument(
        "--metadata-file",
        type=Path,
        help=(
            "Optional JSON map from relative paths to title, HTTPS URL, updated date, and assistant"
        ),
    )
    args = parser.parse_args()
    if not args.source_dir.is_dir():
        raise SystemExit(f"Source directory does not exist: {args.source_dir}")
    metadata = None
    if args.metadata_file:
        metadata = load_source_metadata(args.metadata_file)
    ingest(
        args.source_dir,
        args.source_id,
        args.version,
        args.source_kind,
        args.assistant,
        args.allow_empty,
        metadata,
    )


if __name__ == "__main__":
    main()
