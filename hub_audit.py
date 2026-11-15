"""Audit every Hub write action and sign-in to the operations store and Log Analytics.

Only metadata is recorded: who, which route, HTTP status and duration. Request and response
bodies (chat prompts, catalog payloads, tokens) are never read here.
"""

import logging
import time
import uuid

import anyio

logger = logging.getLogger(__name__)

WRITE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
SIGN_IN_PATHS = {"/auth/callback": "sign_in", "/auth/azure/callback": "azp_connect"}


ASSISTANT_PREFIXES = (
    ("/api/admin/terraform-state", "terraform-state"),
    ("/api/admin/catalog-escalations", "service-catalog"),
    ("/api/catalog", "service-catalog"),
    ("/api/schemas", "service-catalog"),
    ("/api/lookups", "azure-lookup"),
    ("/api/admin", "admin"),
    ("/auth", "sign-in"),
)


def assistant_for(path: str, path_params: dict | None) -> str:
    """Which assistant or Hub area the request belongs to; chat routes carry it in the path."""
    if path_params and path_params.get("assistant_id"):
        return str(path_params["assistant_id"])
    return next((name for prefix, name in ASSISTANT_PREFIXES if path.startswith(prefix)), "hub")


def outcome_for(status: int) -> str:
    if status < 400:
        return "succeeded"
    if status in (401, 403):
        return "denied"
    if status == 409:
        return "refused"
    return "failed"


def identity(session) -> dict:
    user = session.get("user") if isinstance(session, dict) else None
    azp = session.get("azure_user") if isinstance(session, dict) else None
    user = user if isinstance(user, dict) else {}
    azp = azp if isinstance(azp, dict) else {}
    return {
        "ab_oid": user.get("oid", ""),
        "user_name": user.get("name", ""),
        "user_email": user.get("email", ""),
        "azp_oid": azp.get("oid", ""),
        "azp_email": azp.get("email", ""),
        "admin": "CloudOps.Admin" in (user.get("roles") or []),
    }


class HubAuditMiddleware:
    """Must run inside SessionMiddleware so scope["session"] is available."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        method = scope.get("method", "")
        path = scope.get("path", "")
        if scope["type"] != "http" or (method not in WRITE_METHODS and path not in SIGN_IN_PATHS):
            await self.app(scope, receive, send)
            return
        # Capture the caller first: logout clears the session before the response.
        before = identity(scope.get("session"))
        status = {"code": 500}
        started = time.monotonic()

        async def capture(message):
            if message["type"] == "http.response.start":
                status["code"] = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, capture)
        finally:
            who = before if before["ab_oid"] else identity(scope.get("session"))
            endpoint = scope.get("endpoint")
            route = getattr(scope.get("route"), "path", path)
            assistant = assistant_for(path, scope.get("path_params"))
            uses_azp = assistant in ("firewall", "azure-lookup") or path.startswith("/auth/azure")
            event = {
                "category": "hub",
                "account": "AZP" if uses_azp and who["azp_oid"] else "AB",
                "assistant": assistant,
                "operation": SIGN_IN_PATHS.get(path)
                or getattr(endpoint, "__name__", "")
                or f"{method} {route}",
                "operation_id": uuid.uuid4().hex,
                "outcome": outcome_for(status["code"]),
                **who,
                "target": path,
                "method": method,
                "route": route,
                "http_status": status["code"],
                "duration_ms": round((time.monotonic() - started) * 1000),
            }
            try:
                await anyio.to_thread.run_sync(scope["app"].state.operations.audit, event)
            except Exception as error:  # noqa: BLE001 - never break the request on audit I/O
                logger.warning("Hub audit write failed (%s)", type(error).__name__)
