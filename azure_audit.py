"""Fail-closed audit context for Azure tools, separate from anonymous question insights."""

import uuid
from contextlib import contextmanager
from contextvars import ContextVar

from cloudops_hub import firewall_audit

_context: ContextVar[dict | None] = ContextVar("azure_audit", default=None)


@contextmanager
def audit_scope(store, settings, ab: dict, azp: dict, assistant: str = "azure-lookup"):
    ab_access = azp.get("authorization_mode") == "ab_role"
    token = _context.set(
        {
            "store": store,
            "account": "AB" if ab_access else "AZP",
            "authorization_mode": "ab_role" if ab_access else "azp",
            "authorization_role": azp.get("authorization_role", "CloudOpsHub.AzureUser"),
            "assistant": assistant,
            "ab_oid": ab["oid"],
            "user_name": ab.get("name", ""),
            "user_email": ab.get("email", ""),
            "azp_oid": None if ab_access else azp["oid"],
            "azp_email": None if ab_access else azp.get("email", ""),
            "tenant_id": azp["tid"],
            "managed_identity_client_id": settings.azure_client_id,
            "subscription_ids": settings.firewall_subscription_ids,
            "allowed_subscription_ids": settings.arm_subscription_ids,
            "workspace_id": settings.firewall_log_workspace_id,
            "runner_vnet_id": settings.firewall_runner_vnet_id,
            "correlation_id": firewall_audit.correlation_id() or uuid.uuid4().hex,
        }
    )
    try:
        yield
    finally:
        _context.reset(token)


def begin(operation: str) -> dict | None:
    context = _context.get()
    if context is None:
        return None
    event = {key: value for key, value in context.items() if key != "store"}
    event.update(operation=operation, operation_id=uuid.uuid4().hex, outcome="started")
    context["store"].audit(event)
    return event


def finish(event: dict | None, outcome: str) -> None:
    if event is not None:
        _context.get()["store"].audit({**event, "outcome": outcome})


def run_audited(store, settings, ab, azp, function, *args, assistant: str = "firewall", **kwargs):
    with audit_scope(store, settings, ab, azp, assistant):
        return function(*args, **kwargs)
