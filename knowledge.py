"""Bounded query expansion and embeddings; configuration is data, never instructions."""

import hashlib
import json
import logging
import math
import re
import threading
import time
from collections import OrderedDict
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

from azure.core import MatchConditions
from azure.core.exceptions import AzureError, ResourceModifiedError
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from openai import RateLimitError

from cloudops_hub.config import Settings
from cloudops_hub.model_usage import complete
from cloudops_hub.resource_catalog import resource_matches

logger = logging.getLogger(__name__)
IDENTIFIER = re.compile(
    r"/subscriptions/[^\s`\"<>]+"
    r"|[A-Za-z][A-Za-z0-9]*(?:[_.:/-][A-Za-z0-9]+)+"
    r"|\b[A-Z][a-z]+[A-Z][A-Za-z0-9]*\b"
    r"|\b[A-Z][A-Z0-9_]{2,}\b"
)
EMBEDDING_CACHE_TTL_SECONDS = 1800
EMBEDDING_CACHE_MAX_ENTRIES = 128
_embedding_cache: OrderedDict[str, tuple[float, list[float]]] = OrderedDict()
_embedding_cache_lock = threading.Lock()
_glossary_warnings: ContextVar[list[str] | None] = ContextVar("glossary_warnings", default=None)
_glossary_lock = threading.Lock()
GLOSSARY_RETRY_SECONDS = 10
GLOSSARY_WARNING_HEADER = "X-Glossary-Warning"


@dataclass
class _GlossarySnapshot:
    value: dict | None = None
    etag: str | None = None
    retry_at: float = 0


_glossary_cache: OrderedDict[tuple[str, str, str], _GlossarySnapshot] = OrderedDict()


class GlossaryStatusMiddleware:
    """Share per-request warnings with worker threads without sharing them across users."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        warnings: list[str] = []
        token = _glossary_warnings.set(warnings)

        async def send_with_warning(message):
            if message["type"] == "http.response.start" and warnings:
                message.setdefault("headers", []).append(
                    (GLOSSARY_WARNING_HEADER.lower().encode(), warnings[-1].encode("ascii"))
                )
            await send(message)

        try:
            await self.app(scope, receive, send_with_warning)
        finally:
            _glossary_warnings.reset(token)


def _unavailable_glossary(snapshot: _GlossarySnapshot) -> dict:
    warning = (
        "Glossary updates are temporarily unavailable. Using the last verified glossary."
        if snapshot.value is not None
        else "Glossary is temporarily unavailable. Custom keyword matching is disabled for this "
        "answer; check terminology against your documentation."
    )
    warnings = _glossary_warnings.get()
    if warnings is not None and warning not in warnings:
        warnings.append(warning)
    return deepcopy(snapshot.value) if snapshot.value is not None else {"version": 1, "groups": []}


def _embedding_cache_key(settings: Settings, text: str) -> str:
    value = json.dumps(
        [
            settings.azure_openai_endpoint,
            settings.azure_openai_embedding_deployment,
            settings.azure_openai_embedding_dimensions,
            text,
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _get_cached_embedding(key: str) -> list[float] | None:
    now = time.monotonic()
    with _embedding_cache_lock:
        cached = _embedding_cache.get(key)
        if cached is None:
            return None
        if now - cached[0] >= EMBEDDING_CACHE_TTL_SECONDS:
            del _embedding_cache[key]
            return None
        _embedding_cache.move_to_end(key)
        return cached[1].copy()


def _cache_embedding(key: str, vector: list[float]) -> None:
    with _embedding_cache_lock:
        _embedding_cache[key] = (time.monotonic(), vector.copy())
        _embedding_cache.move_to_end(key)
        while len(_embedding_cache) > EMBEDDING_CACHE_MAX_ENTRIES:
            _embedding_cache.popitem(last=False)


def _clear_embedding_cache() -> None:
    with _embedding_cache_lock:
        _embedding_cache.clear()


def identifiers(text: str, limit: int = 32) -> list[str]:
    matches = IDENTIFIER.findall(text)
    matches.extend(re.findall(r'"([\w.-]+)"\s*:', text))
    matches.extend(re.findall(r"`([\w./:-]+)`", text))
    return list(dict.fromkeys(term.rstrip(".,;)]}") for term in matches))[:limit]


GLOSSARY_TERM_PATTERN = r"^[\w .:/()?-]+$"
GLOSSARY_SCOPES = {
    "shared", "other-agents", "firewall", "platform-admin", "landing-zone",
    "service-catalog",
}
EDITABLE_GLOSSARY_SCOPES = {"firewall", "other-agents"}


def _glossary_scope_applies(scope: str, assistant_id: str) -> bool:
    if scope == "shared":
        return True
    if assistant_id == "shared":
        return False
    if scope == "firewall":
        return assistant_id == "firewall"
    return assistant_id != "firewall"


def _normalized_glossary_scope(scope: str) -> str:
    if scope in {"shared", "firewall"}:
        return scope
    return "other-agents"


def scoped_dictionary(value: dict) -> dict:
    """Split legacy shared meanings into independent records without changing vocabulary."""
    result = deepcopy(value)
    aliases = []
    for group in result["groups"]:
        for scope in ("firewall", "other-agents"):
            aliases.append({"canonical": group[0], "terms": group[1:], "scope": scope})
    for alias in result.get("aliases", []):
        scope = _normalized_glossary_scope(alias.get("scope", "shared"))
        for target in ("firewall", "other-agents") if scope == "shared" else (scope,):
            aliases.append({**alias, "scope": target})
    result["groups"] = []
    result["aliases"] = aliases
    return validate_synonyms(result)


def validate_synonyms(value: object) -> dict:
    if (
        not isinstance(value, dict)
        or not {"version", "groups"} <= set(value) <= {
            "version", "groups", "aliases", "guide_keywords", "dismissed_keywords",
            "chat_keywords", "guide_aliases", "glossary_source"
        }
        or value["version"] != 1
        or not isinstance(value["groups"], list)
        or len(value["groups"]) > 200
    ):
        raise ValueError("Invalid synonym dictionary")
    for group in value["groups"]:
        if (
            not isinstance(group, list)
            or not 2 <= len(group) <= 16
            or any(
                not isinstance(term, str)
                or not 1 <= len(term) <= 100
                or not re.fullmatch(GLOSSARY_TERM_PATTERN, term)
                for term in group
            )
        ):
            raise ValueError("Invalid synonym group")
    aliases = value.get("aliases", [])
    if not isinstance(aliases, list) or len(aliases) > 1000:
        raise ValueError("Invalid alias dictionary")
    for alias in aliases:
        if (
            not isinstance(alias, dict)
            or not {"canonical", "terms"} <= set(alias) <= {
                "canonical", "terms", "scope", "ambiguous_terms",
            }
            or alias.get("scope", "shared") not in GLOSSARY_SCOPES
            or not isinstance(alias["canonical"], str)
            or not isinstance(alias["terms"], list)
            or not 1 <= len(alias["terms"]) <= 15
            or any(
                not isinstance(term, str)
                or not 1 <= len(term) <= 100
                or not re.fullmatch(GLOSSARY_TERM_PATTERN, term)
                for term in [alias["canonical"], *alias["terms"]]
            )
        ):
            raise ValueError("Invalid alias entry")
        if "ambiguous_terms" in alias and (
            alias.get("scope") != "other-agents"
            or not isinstance(alias["ambiguous_terms"], list)
            or not 1 <= len(alias["ambiguous_terms"]) <= 15
            or any(
                not isinstance(term, str) or not 1 <= len(term) <= 100
                or term != term.strip()
                or not re.fullmatch(GLOSSARY_TERM_PATTERN, term)
                for term in alias["ambiguous_terms"]
            )
            or len({term.casefold() for term in alias["ambiguous_terms"]})
            != len(alias["ambiguous_terms"])
        ):
            raise ValueError("Invalid ambiguous glossary terms")
    ambiguous: dict[tuple[str, str], set[str]] = {}
    unique: dict[tuple[str, str], set[str]] = {}
    for alias in aliases:
        scope = _normalized_glossary_scope(alias.get("scope", "shared"))
        for term in [alias["canonical"], *alias["terms"]]:
            for target in ("firewall", "other-agents") if scope == "shared" else (scope,):
                unique.setdefault((target, term.casefold()), set()).add(
                    alias["canonical"].casefold(),
                )
        for term in alias.get("ambiguous_terms", []):
            ambiguous.setdefault((scope, term.casefold()), set()).add(
                alias["canonical"].casefold(),
            )
    if "guide_keywords" in value:
        from cloudops_hub.glossary_review import validate_manifest

        validate_manifest(value["guide_keywords"])
    if "chat_keywords" in value:
        from cloudops_hub.glossary_review import validate_chat_keywords

        validate_chat_keywords(value["chat_keywords"])
    if "guide_aliases" in value:
        from cloudops_hub.glossary_review import validate_guide_aliases

        validate_guide_aliases(value["guide_aliases"])
        for alias in value["guide_aliases"]:
            for term in [alias["canonical"], *alias["terms"]]:
                unique.setdefault(("other-agents", term.casefold()), set()).add(
                    alias["canonical"].casefold(),
                )
    for group in value["groups"]:
        for term in group:
            unique.setdefault(("other-agents", term.casefold()), set()).add(group[0].casefold())
    if any(len(owners) < 2 or key in unique for key, owners in ambiguous.items()):
        raise ValueError("Ambiguous terms require multiple meanings and cannot be unique aliases")
    if "glossary_source" in value:
        source = value["glossary_source"]
        if (
            not isinstance(source, dict)
            or set(source) != {"repository", "path", "commit", "imported_at"}
            or source["repository"] != "almbrand/cloudops-openai-hub-knowledge-source"
            or source["path"] != "glossary/retrieval-synonyms.json"
            or not isinstance(source["commit"], str)
            or not re.fullmatch(r"[a-f0-9]{40}", source["commit"])
            or not isinstance(source["imported_at"], str)
        ):
            raise ValueError("Invalid glossary source")
        try:
            if datetime.fromisoformat(source["imported_at"]).tzinfo is None:
                raise ValueError("Missing timezone")
        except ValueError:
            raise ValueError("Invalid glossary import timestamp") from None
    dismissed = value.get("dismissed_keywords", [])
    if (
        not isinstance(dismissed, list) or len(dismissed) > 2000
        or any(not isinstance(item, str) or not re.fullmatch(r"[a-f0-9]{64}", item)
               for item in dismissed)
    ):
        raise ValueError("Invalid dismissed keyword list")
    return value


@contextmanager
def _knowledge_blob(settings: Settings, blob_name: str, credential=None):
    if not settings.azure_storage_account_url:
        raise ValueError("Glossary Blob storage account is not configured")
    with ExitStack() as stack:
        if credential is None:
            credential = stack.enter_context(DefaultAzureCredential())
        service = stack.enter_context(
            BlobServiceClient(
                settings.azure_storage_account_url,
                credential=credential,
                connection_timeout=2,
                read_timeout=3,
                retry_total=1,
                retry_backoff_factor=1,
            )
        )
        yield service.get_blob_client(
            settings.azure_storage_knowledge_container, blob_name
        )


@contextmanager
def glossary_blob(settings: Settings, credential=None):
    with _knowledge_blob(settings, settings.retrieval_synonyms_blob, credential) as blob:
        yield blob


GLOSSARY_SCAN_CACHE_BLOB = "glossary-keyword-scan.json"


@contextmanager
def glossary_scan_cache_blob(settings: Settings, credential=None):
    with _knowledge_blob(settings, GLOSSARY_SCAN_CACHE_BLOB, credential) as blob:
        yield blob


MAX_GLOSSARY_BYTES = 524_288


def _read_glossary(blob, properties) -> dict:
    if properties.size > MAX_GLOSSARY_BYTES:
        raise ValueError("Glossary exceeds 512 KiB")
    raw = blob.download_blob(
        offset=0,
        length=MAX_GLOSSARY_BYTES + 1,
        etag=properties.etag, match_condition=MatchConditions.IfNotModified
    ).readall()
    if len(raw) > MAX_GLOSSARY_BYTES:
        raise ValueError("Glossary exceeds 512 KiB")
    return validate_synonyms(json.loads(raw))


def read_glossary(settings: Settings) -> dict:
    """Administrator reads must be live, never a degraded or cached response."""
    with glossary_blob(settings) as blob:
        properties = blob.get_blob_properties()
        return _admin_snapshot(_read_glossary(blob, properties), properties.etag, blob.url)


def _admin_snapshot(value: dict, etag: str, blob_url: str) -> dict:
    from cloudops_hub.glossary_review import firewall_starters, pending_keywords

    return {
        "dictionary": {key: item for key, item in value.items() if key != "chat_keywords"},
        "etag": etag, "blob_url": blob_url,
        "pending": pending_keywords(value),
        "firewall_starters": firewall_starters(),
    }


def _write_glossary(settings: Settings, blob, value: dict, etag: str) -> dict:
    from azure.storage.blob import ContentSettings

    value = scoped_dictionary(validate_synonyms(value))
    raw = json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8")
    if len(raw) > MAX_GLOSSARY_BYTES:
        raise ValueError("Glossary exceeds 512 KiB")
    snapshot = _admin_snapshot(value, etag, blob.url)
    result = blob.upload_blob(
        raw,
        overwrite=True,
        etag=etag,
        match_condition=MatchConditions.IfNotModified,
        content_settings=ContentSettings(content_type="application/json"),
    )
    key = (
        settings.azure_storage_account_url,
        settings.azure_storage_knowledge_container,
        settings.retrieval_synonyms_blob,
    )
    # Publish only a confirmed successful write; other processes recheck Blob ETags.
    with _glossary_lock:
        _glossary_cache[key] = _GlossarySnapshot(deepcopy(value), result["etag"])
        _glossary_cache.move_to_end(key)
        while len(_glossary_cache) > 8:
            _glossary_cache.popitem(last=False)
    snapshot["etag"] = result["etag"]
    return snapshot


def edit_glossary_entry(
    settings: Settings, kind: str, index: int, canonical: str, terms: list[str], etag: str,
    scope: str | None = None,
) -> dict:
    if scope is not None and scope not in EDITABLE_GLOSSARY_SCOPES:
        raise ValueError("Invalid glossary scope")
    with glossary_blob(settings) as blob:
        properties = blob.get_blob_properties()
        if properties.etag != etag:
            raise ResourceModifiedError("Glossary changed; reload before saving.")
        value = _read_glossary(blob, properties)
        if kind not in {"groups", "aliases"} or not 0 <= index < len(value.get(kind, [])):
            raise ValueError("Glossary entry does not exist")
        if kind == "groups" and scope is None:
            value[kind][index] = [canonical, *terms]
        elif kind == "groups":
            original = value[kind].pop(index)
            opposite = "other-agents" if scope == "firewall" else "firewall"
            value.setdefault("aliases", []).append({
                "canonical": original[0], "terms": original[1:], "scope": opposite,
            })
            value.setdefault("aliases", []).append({
                "canonical": canonical, "terms": terms, "scope": scope,
            })
        else:
            original = value[kind][index]
            if scope is not None and original.get("scope", "shared") == "shared":
                opposite = "other-agents" if scope == "firewall" else "firewall"
                value[kind].append({**original, "scope": opposite})
            value[kind][index] = {
                **value[kind][index], "canonical": canonical, "terms": terms,
                **({"scope": scope} if scope is not None else {}),
            }
        return _write_glossary(settings, blob, value, etag)


def save_glossary_alias(
    settings: Settings, keyword: str, canonical: str, etag: str, scope: str = "other-agents"
) -> dict:
    if scope not in EDITABLE_GLOSSARY_SCOPES:
        raise ValueError("Invalid glossary scope")
    with glossary_blob(settings) as blob:
        properties = blob.get_blob_properties()
        if properties.etag != etag:
            raise ResourceModifiedError("Glossary changed; reload before saving.")
        value = scoped_dictionary(_read_glossary(blob, properties))
        normalized_scope = _normalized_glossary_scope(scope)
        alias = next(
            (alias for alias in value.get("aliases", [])
             if alias["canonical"].casefold() == canonical.casefold()
             and _normalized_glossary_scope(alias.get("scope", "shared")) == normalized_scope
             ),
            None,
        )
        if alias is not None:
            if keyword.casefold() not in {term.casefold() for term in
                                          [alias["canonical"], *alias["terms"]]}:
                alias["terms"].append(keyword)
        else:
            value.setdefault("aliases", []).append({
                "canonical": canonical, "terms": [keyword],
                "scope": scope,
            })
        return _write_glossary(settings, blob, value, etag)


def load_synonyms(settings: Settings, credential=None, *, assistant_id: str = "shared") -> dict:
    value = _load_synonyms(settings, credential)
    # Keep review context private; apply only shared and active-assistant terminology.
    return {
        "version": value["version"],
        "groups": value["groups"],
        **({"aliases": effective_aliases(value, assistant_id)}
           if "aliases" in value or "guide_aliases" in value else {}),
    }


def effective_aliases(value: dict, assistant_id: str) -> list[dict]:
    aliases = [
        {
            "canonical": alias["canonical"], "terms": alias["terms"],
            **({"ambiguous_terms": alias["ambiguous_terms"]}
               if "ambiguous_terms" in alias else {}),
        }
        for alias in value.get("aliases", [])
        if _glossary_scope_applies(alias.get("scope", "shared"), assistant_id)
    ]
    if assistant_id != "landing-zone":
        return aliases
    owners: dict[str, set[str]] = {}
    for group in [
        *value["groups"], *[[alias["canonical"], *alias["terms"]] for alias in aliases],
    ]:
        for term in group:
            owners.setdefault(term.casefold(), set()).add(group[0].casefold())
    for alias in value.get("guide_aliases", []):
        canonical = alias["canonical"].casefold()
        if owners.get(canonical, {canonical}) != {canonical}:
            continue
        terms = [
            term for term in alias["terms"]
            if owners.get(term.casefold(), {canonical}) == {canonical}
        ]
        if terms:
            aliases.append({"canonical": alias["canonical"], "terms": terms})
    return aliases


def merge_glossary_entries(settings: Settings, entries: list[dict], keep: dict, etag: str) -> dict:
    with glossary_blob(settings) as blob:
        properties = blob.get_blob_properties()
        if properties.etag != etag:
            raise ResourceModifiedError("Glossary changed; reload before saving.")
        value = _read_glossary(blob, properties)
        keys = {(item["kind"], item["index"]) for item in entries}
        target = (keep["kind"], keep["index"])
        if len(keys) != len(entries) or len(keys) < 2 or target not in keys:
            raise ValueError("Select distinct entries and a term to keep")
        selected = []
        ambiguous_terms = {}
        scopes = set()
        for kind, index in keys:
            if kind not in {"groups", "aliases"} or not 0 <= index < len(value.get(kind, [])):
                raise ValueError("Glossary entry does not exist")
            entry = value[kind][index]
            if kind == "aliases":
                for term in entry.get("ambiguous_terms", []):
                    ambiguous_terms.setdefault(term.casefold(), term)
            scopes.add(
                "shared" if kind == "groups"
                else _normalized_glossary_scope(entry.get("scope", "shared"))
            )
            names = entry if kind == "groups" else [entry["canonical"], *entry["terms"]]
            selected.append(names)
            if (kind, index) == target:
                canonical = names[0]
        if len(scopes) != 1:
            raise ValueError("Only merge meanings that apply to the same assistant")
        scope = scopes.pop()
        terms = {canonical.casefold(): canonical}
        for names in selected:
            for name in names:
                terms.setdefault(name.casefold(), name)
        others = [name for key, name in terms.items() if key != canonical.casefold()]
        if len(others) > 15:
            raise ValueError("Merged entry exceeds 15 aliases; edit entries first")
        for kind in ("groups", "aliases"):
            for index in sorted((index for k, index in keys if k == kind), reverse=True):
                del value[kind][index]
        value.setdefault("aliases", []).append({
            "canonical": canonical, "terms": others,
            **({"scope": scope} if scope != "shared" else {}),
            **({"ambiguous_terms": list(ambiguous_terms.values())} if ambiguous_terms else {}),
        })
        return _write_glossary(settings, blob, value, etag)


def _load_synonyms(settings: Settings, credential=None) -> dict:
    key = (
        settings.azure_storage_account_url,
        settings.azure_storage_knowledge_container,
        settings.retrieval_synonyms_blob,
    )
    with _glossary_lock:
        snapshot = _glossary_cache.setdefault(key, _GlossarySnapshot())
        _glossary_cache.move_to_end(key)
        while len(_glossary_cache) > 8:
            _glossary_cache.popitem(last=False)
        if time.monotonic() < snapshot.retry_at:
            return _unavailable_glossary(snapshot)
        try:
            with glossary_blob(settings, credential) as blob:
                properties = blob.get_blob_properties()
                if properties.size > MAX_GLOSSARY_BYTES:
                    raise ValueError("Glossary exceeds 512 KiB")
                if snapshot.value is not None and properties.etag == snapshot.etag:
                    return deepcopy(snapshot.value)
                value = _read_glossary(blob, properties)
        except (AzureError, ValueError, UnicodeError) as exc:
            snapshot.retry_at = time.monotonic() + GLOSSARY_RETRY_SECONDS
            logger.warning(
                "Glossary Blob refresh unavailable (%s); mode=%s",
                type(exc).__name__,
                "last_verified" if snapshot.value is not None else "no_custom_matching",
            )
            return _unavailable_glossary(snapshot)
        snapshot.value = value
        snapshot.etag = properties.etag
        snapshot.retry_at = 0
        return deepcopy(value)


def _term_matches(term: str, text: str) -> bool:
    return bool(
        re.search(rf"(?<![\w./:@-]){re.escape(term)}(?![\w/:@-]|\.[\w-])", text, re.IGNORECASE)
    )


def ambiguous_aliases(text: str, context: str, synonyms: dict) -> list[str]:
    """Only explicit canonical names disambiguate a shared approved alias."""
    meanings: dict[str, set[str]] = {}
    for alias in synonyms.get("aliases", []):
        for term in alias["terms"]:
            meanings.setdefault(term.casefold(), set()).add(alias["canonical"])
    for term, names in meanings.items():
        if (
            len(names) > 1
            and _term_matches(term, text)
            and not any(_term_matches(name, context) for name in names)
        ):
            return sorted(names)
    return []


def unresolved_glossary_meanings(text: str, synonyms: dict) -> list[str]:
    """Resolve explicit longer terms before considering a shared alias ambiguous."""
    matches = []
    for group in [
        *synonyms["groups"],
        *[[alias["canonical"], *alias["terms"], *alias.get("ambiguous_terms", [])]
          for alias in synonyms.get("aliases", [])],
    ]:
        for term in group:
            for match in re.finditer(
                rf"(?<![\w./:@-]){re.escape(term)}(?![\w/:@-]|\.[\w-])", text, re.I,
            ):
                matches.append((match.start(), match.end(), group[0]))
    all_spans: dict[tuple[int, int], dict[str, str]] = {}
    for start, end, canonical in matches:
        all_spans.setdefault((start, end), {})[canonical.casefold()] = canonical
    spans = []
    furthest_end = -1
    for (_start, end), meanings in sorted(all_spans.items(), key=lambda item: (
        item[0][0], -item[0][1],
    )):
        if end <= furthest_end:
            continue
        spans.append(meanings)
        furthest_end = end
    explicit = {
        canonical for meanings in spans if len(meanings) == 1
        for canonical in meanings
    }
    unresolved = {}
    for meanings in spans:
        if len(meanings) > 1 and len(explicit.intersection(meanings)) != 1:
            unresolved.update(meanings)
    return sorted(unresolved.values(), key=str.casefold)


def glossary_matches(text: str, synonyms: dict) -> list[list[str]]:
    ambiguous_terms = {
        term.casefold() for alias in synonyms.get("aliases", [])
        for term in alias.get("ambiguous_terms", [])
    }
    custom = [
        group
        for group in [
            *synonyms["groups"],
            *[[alias["canonical"], *alias["terms"]] for alias in synonyms.get("aliases", [])],
        ]
        if any(_term_matches(term, text) for term in group)
    ]
    return custom + [
        group for group in resource_matches(text)
        if group not in custom and any(
            term.casefold() not in ambiguous_terms and _term_matches(term, text)
            for term in group
        )
    ]


def resource_scope_groups(text: str, synonyms: dict) -> list[list[str]]:
    """Keep distinct reviewed meanings without choosing between ambiguous aliases."""
    groups = []
    for group in glossary_matches(text, synonyms):
        names = {term.casefold() for term in group}
        for existing in groups:
            existing_names = {term.casefold() for term in existing}
            if group[0].casefold() in existing_names or existing[0].casefold() in names:
                existing.extend(term for term in group if term.casefold() not in existing_names)
                break
        else:
            groups.append(list(group))
    return groups


def resource_scope_matches(text: str, groups: list[list[str]]) -> bool:
    return all(any(_term_matches(term, text) for term in group) for group in groups)


def expand_query(original: str, interpretation: str, synonyms: dict) -> str:
    # Keep the original wording and exact identifiers even when model interpretation is wrong.
    additions = [
        term
        for group in glossary_matches(f"{original} {interpretation}", synonyms)
        for term in group
    ]
    return " ".join(dict.fromkeys([original, interpretation, *additions]))[:16_000]


def embedding_retry_delay(error: RateLimitError, attempt: int) -> float:
    headers = error.response.headers
    for name, scale in (
        ("retry-after-ms", 0.001),
        ("x-ms-retry-after-ms", 0.001),
        ("retry-after", 1),
    ):
        value = headers.get(name)
        if value is None:
            continue
        try:
            delay = float(value) * scale
        except ValueError:
            if name != "retry-after":
                continue
            try:
                date = parsedate_to_datetime(value)
                delay = (date - datetime.now(UTC)).total_seconds()
            except (TypeError, ValueError, OverflowError):
                continue
        if math.isfinite(delay) and delay > 0:
            return delay
    return min(120, 60 * (2**attempt))


def embed_texts(
    client,
    settings: Settings,
    texts: list[str],
    *,
    rate_limit_retries: int = 0,
    cache_single_text: bool = False,
) -> list[list[float]]:
    from cloudops_hub.firewall_model_data import active_references, protect_model_data

    references = active_references()
    # Indexed vectors use real text; masked query vectors would not match them.
    if references is None or not references.restore_outputs:
        texts = protect_model_data(texts)
    cache_key = (
        _embedding_cache_key(settings, texts[0])
        if cache_single_text and len(texts) == 1
        else None
    )
    if cache_key:
        cached = _get_cached_embedding(cache_key)
        if cached is not None:
            return [cached]
    for attempt in range(rate_limit_retries + 1):
        try:
            response = complete(
                client, settings, "embedding", embeddings_api=True,
                model=settings.azure_openai_embedding_deployment,
                input=texts,
                dimensions=settings.azure_openai_embedding_dimensions,
                timeout=settings.model_request_timeout_seconds,
            )
            break
        except RateLimitError as error:
            if attempt == rate_limit_retries:
                raise
            delay = embedding_retry_delay(error, attempt)
            if delay > 120:
                logger.error("Embedding retry delay exceeds the 120-second import retry bound")
                raise
            logger.warning(
                "Embedding rate limited; retry %d/%d in %.3f seconds",
                attempt + 1,
                rate_limit_retries,
                delay,
            )
            time.sleep(delay)
    ordered = sorted(response.data, key=lambda item: item.index)
    if [item.index for item in ordered] != list(range(len(texts))):
        raise RuntimeError("Embedding response count/index mismatch")
    vectors = [item.embedding for item in ordered]
    if any(
        len(vector) != settings.azure_openai_embedding_dimensions
        or any(not math.isfinite(value) for value in vector)
        for vector in vectors
    ):
        raise RuntimeError("Embedding response has invalid dimensions or values")
    if cache_key:
        _cache_embedding(cache_key, vectors[0])
    return vectors
