"""Restricted Firewall activity records, separate from temporary conversation storage."""

import json
import re
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime

_context: ContextVar[dict | None] = ContextVar("firewall_activity", default=None)
_SECRET_KEYS = frozenset(
    {
        "authorization",
        "proxyauthorization",
        "cookie",
        "setcookie",
        "password",
        "passwd",
        "pwd",
        "secret",
        "clientsecret",
        "token",
        "accesstoken",
        "refreshtoken",
        "idtoken",
        "apikey",
        "accountkey",
        "sharedaccesskey",
        "sig",
        "signature",
        "privatekey",
        "sessionsecret",
        "connectionstring",
        "xapikey",
    }
)
_PRIVATE_KEY = re.compile(
    r"-----BEGIN (?:[\w ]*PRIVATE KEY)-----.*?-----END (?:[\w ]*PRIVATE KEY)-----",
    re.S,
)
PAYLOAD_CHARS = 8_000


def sanitize(value):
    from cloudops_hub.question_log import redact

    if isinstance(value, dict):
        return {
            key: "[redacted]"
            if re.sub(r"[^a-z0-9]", "", str(key).lower())
            in _SECRET_KEYS | {"reasoningcontent", "encryptedcontent"}
            else sanitize(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [sanitize(item) for item in value]
    if isinstance(value, str):
        # Tool arguments/results sometimes arrive as serialized JSON.
        try:
            parsed = json.loads(value)
        except ValueError:
            return redact(_PRIVATE_KEY.sub("[private key]", value), strip=False)
        if isinstance(parsed, (dict, list)):
            sanitized = sanitize(parsed)
            return value if sanitized == parsed else json.dumps(sanitized, ensure_ascii=False)
        return redact(_PRIVATE_KEY.sub("[private key]", value), strip=False)
    return value


@contextmanager
def activity_scope(
    store,
    user: dict,
    azp: dict,
    conversation_id: str | None,
    *,
    managed_identity_client_id: str = "",
):
    ab_access = azp.get("authorization_mode") == "ab_role"
    token = _context.set(
        {
            "store": store,
            "ab_oid": user["oid"],
            "user_name": user.get("name") or user.get("email") or user["oid"],
            "user_email": user.get("email", ""),
            "account": "AB" if ab_access else "AZP",
            "authorization_mode": "ab_role" if ab_access else "azp",
            "authorization_role": azp.get("authorization_role", "CloudOpsHub.AzureUser"),
            "azp_oid": None if ab_access else azp["oid"],
            "azp_name": None if ab_access else azp.get("name") or azp.get("email") or azp["oid"],
            "azp_email": None if ab_access else azp.get("email", ""),
            "tenant_id": azp["tid"],
            "managed_identity_client_id": managed_identity_client_id,
            "correlation_id": uuid.uuid4().hex,
            "conversation_id": conversation_id,
        }
    )
    try:
        yield
    finally:
        _context.reset(token)


def correlation_id() -> str | None:
    context = _context.get()
    return context["correlation_id"] if context else None


def emit(operation: str, outcome: str, *, operation_id: str | None = None, **payload) -> None:
    context = _context.get()
    if context is None:
        return
    text = json.dumps(
        sanitize(payload),
        ensure_ascii=True,
        separators=(",", ":"),
        default=lambda value: value.isoformat() if isinstance(value, datetime) else str(value),
    )
    chunks = [text[index : index + PAYLOAD_CHARS] for index in range(0, len(text), PAYLOAD_CHARS)]
    event = {
        **{key: value for key, value in context.items() if key != "store"},
        "category": "firewall_activity",
        "assistant": "firewall",
        "operation": operation,
        "outcome": outcome,
        "operation_id": operation_id or uuid.uuid4().hex,
        "event_id": uuid.uuid4().hex,
        "timestamp": datetime.now(UTC).isoformat(),
        "chunk_count": len(chunks),
    }
    # One ingestion batch per event; chunking preserves content without oversized fields.
    context["store"].firewall_activity(
        [{**event, "chunk_index": index, "payload": chunk} for index, chunk in enumerate(chunks)]
    )
