import logging
import re
import secrets
import time
from typing import Any

from authlib.integrations.starlette_client import OAuth
from fastapi import APIRouter, HTTPException, Request
from starlette.responses import RedirectResponse

from cloudops_hub.config import Settings

router = APIRouter(prefix="/auth", tags=["authentication"])
logger = logging.getLogger(__name__)
oauth = OAuth()
ADMIN_ROLE = "CloudOpsHub.Admin"
AZURE_ROLE = "CloudOpsHub.AzureUser"
PLATFORM_ROLE = "CloudOpsHub.PlatformUser"
FIREWALL_AB_ROLE = "CloudOpsHub.FirewallABUser"


def firewall_ab_identity(request: Request) -> dict | None:
    user = current_user(request)
    settings = request.app.state.settings
    roles = user.get("roles", [])
    if FIREWALL_AB_ROLE not in roles or settings.entra_required_app_role not in roles:
        return None
    if user.get("tid") != settings.entra_tenant_id:
        raise HTTPException(status_code=403, detail="The account must belong to the Hub tenant")
    require_ab_account(user.get("email", ""))
    return {
        **user, "authorization_mode": "ab_role", "authorization_role": FIREWALL_AB_ROLE,
    }


def can_use_assistant(user: dict, assistant_id: str) -> bool:
    return assistant_id != "platform-admin" or PLATFORM_ROLE in user.get("roles", [])


def require_assistant(user: dict, assistant_id: str) -> None:
    if not can_use_assistant(user, assistant_id):
        raise HTTPException(status_code=403, detail=f"This assistant requires {PLATFORM_ROLE}")


def account_suffix(username: str, prefix: str) -> str:
    if not isinstance(username, str):
        raise HTTPException(status_code=403, detail="The account has no valid sign-in name")
    match = re.fullmatch(rf"{prefix}([a-z0-9]+)@[^@#\s]+", username.casefold())
    if not match:
        raise HTTPException(status_code=403, detail=f"Log in with your {prefix.upper()} account")
    return match[1]


def require_ab_account(username: str) -> None:
    if not isinstance(username, str) or not re.fullmatch(
        r"(?:ab[a-z0-9]+|ex[0-9]+)@almbrand\.dk", username.casefold()
    ):
        raise HTTPException(
            status_code=403, detail="Log in with your AB account (@almbrand.dk)"
        )


def ab_account_suffix(username: str) -> str:
    require_ab_account(username)
    return username.casefold().split("@", 1)[0][2:]


def require_azp_account(username: str, settings: Settings) -> None:
    account_suffix(username, "azp")
    domain = settings.azp_login_domain or "almbrand.onmicrosoft.com"
    if username.casefold().split("@", 1)[1] != domain.casefold():
        raise HTTPException(status_code=403, detail=f"Log in with your AZP account (@{domain})")


def identity_claims(claims: dict, settings: Settings, prefix: str) -> dict:
    if claims.get("tid") != settings.entra_tenant_id:
        raise HTTPException(status_code=403, detail="The account must belong to the Hub tenant")
    role = AZURE_ROLE if prefix == "azp" else settings.entra_required_app_role
    user = session_user_from_claims(claims, role)
    if prefix == "azp" and AZURE_ROLE not in user["roles"]:
        raise HTTPException(status_code=403, detail="The AZP account needs CloudOpsHub.AzureUser")
    if prefix == "ab":
        require_ab_account(user["email"])
    else:
        require_azp_account(user["email"], settings)
    return {**user, "tid": claims["tid"], "account_kind": prefix}


def azure_user(request: Request) -> dict:
    ab = current_user(request)
    azp = request.session.get("azure_user", {})
    reason = (
        "missing"
        if not isinstance(azp, dict) or not azp
        else "other AB user"
        if azp.get("ab_oid") != ab["oid"]
        else "tenant"
        if azp.get("tid") != request.app.state.settings.entra_tenant_id
        else "expired"
        if azp.get("expires_at", 0) <= time.time()
        else "role"
        if AZURE_ROLE not in azp.get("roles", [])
        else "same account"
        if azp.get("oid") == ab["oid"]
        else ""
    )
    if reason:
        if reason != "missing":
            logger.warning("AZP session rejected: %s", reason)
        raise HTTPException(
            status_code=403,
            detail="Log in with your matching AZP account to use Azure features. "
            "If you do not have an AZP account, contact the IAM team to create one "
            "and arrange the required Azure access.",
        )
    require_azp_account(azp.get("email", ""), request.app.state.settings)
    if ab_account_suffix(ab.get("email", "")) != account_suffix(azp.get("email", ""), "azp"):
        raise HTTPException(
            status_code=403, detail="AZP account does not match the signed-in AB user"
        )
    return azp


def session_user_from_claims(claims: dict[str, Any], required_role: str) -> dict[str, Any]:
    roles = claims.get("roles", [])
    if (
        not isinstance(roles, list)
        or not all(isinstance(role, str) for role in roles)
        or not ({required_role, ADMIN_ROLE} & set(roles))
    ):
        raise HTTPException(status_code=403, detail="The required CloudOps Hub app role is missing")
    oid = claims.get("oid")
    if not isinstance(oid, str) or not oid:
        raise HTTPException(status_code=401, detail="Entra identity is missing its object ID")
    return {
        "oid": oid,
        "name": claims.get("name", ""),
        "email": claims.get("preferred_username", ""),
        "roles": roles,
    }


def configure_oauth(settings: Settings) -> None:
    if settings.entra_tenant_id and settings.entra_client_id and settings.entra_client_secret:
        oauth.register(
            name="entra",
            client_id=settings.entra_client_id,
            client_secret=settings.entra_client_secret,
            server_metadata_url=(
                "https://login.microsoftonline.com/"
                f"{settings.entra_tenant_id}/v2.0/.well-known/openid-configuration"
            ),
            client_kwargs={"scope": "openid profile email"},
        )


def current_user(request: Request) -> dict[str, Any]:
    user = request.session.get("user")
    if not isinstance(user, dict) or not user.get("oid"):
        raise HTTPException(status_code=401, detail="Authentication required")
    if user.get("account_kind") != "ab":
        raise HTTPException(status_code=401, detail="Please sign in again with your AB account")
    return user


def require_csrf(request: Request) -> None:
    expected = request.session.get("csrf")
    supplied = request.headers.get("x-csrf-token")
    if not isinstance(expected, str) or not supplied or supplied != expected:
        raise HTTPException(status_code=403, detail="CSRF validation failed")


@router.get("/login")
async def login(request: Request) -> RedirectResponse:
    client = oauth.create_client("entra")
    if client is None:
        raise HTTPException(status_code=503, detail="Entra authentication is not configured")
    settings: Settings = request.app.state.settings
    redirect_uri = settings.entra_redirect_uri or str(request.url_for("auth_callback"))
    return await client.authorize_redirect(
        request, redirect_uri, prompt="select_account", domain_hint="almbrand.dk"
    )


@router.get("/callback", name="auth_callback")
async def auth_callback(request: Request) -> RedirectResponse:
    client = oauth.create_client("entra")
    if client is None:
        raise HTTPException(status_code=503, detail="Entra authentication is not configured")
    token = await client.authorize_access_token(request)
    claims = token.get("userinfo")
    if not isinstance(claims, dict):
        raise HTTPException(status_code=401, detail="Entra did not return valid identity claims")
    settings: Settings = request.app.state.settings
    try:
        user = identity_claims(claims, settings, "ab")
    except HTTPException as exc:
        logger.warning("AB sign-in rejected status=%s reason=%s", exc.status_code, exc.detail)
        request.session.clear()
        raise
    azp = request.session.get("azure_user")
    request.session.clear()
    if (
        isinstance(azp, dict)
        and azp.get("ab_oid") == user["oid"]
        and azp.get("expires_at", 0) > time.time()
    ):
        request.session["azure_user"] = azp
    request.session["user"] = user
    request.session["csrf"] = secrets.token_urlsafe(32)
    return RedirectResponse(url=settings.frontend_url or "/", status_code=303)


@router.get("/azure/login")
async def azure_login(request: Request) -> RedirectResponse:
    user = current_user(request)
    try:
        azure_user(request)
    except HTTPException as exc:
        if exc.status_code != 403:
            raise
    else:
        return RedirectResponse(
            url=request.app.state.settings.frontend_url or "/", status_code=303
        )
    client = oauth.create_client("entra")
    if client is None:
        raise HTTPException(status_code=503, detail="Entra authentication is not configured")
    request.session.pop("azure_user", None)
    request.session["azure_link_ab"] = user["oid"]
    settings: Settings = request.app.state.settings
    return await client.authorize_redirect(
        request,
        f"{settings.frontend_url}/auth/azure/callback",
        login_hint="azp"
        + ab_account_suffix(user["email"])
        + "@"
        + (settings.azp_login_domain or "almbrand.onmicrosoft.com"),
    )


@router.get("/azure/callback")
async def azure_callback(request: Request) -> RedirectResponse:
    ab = current_user(request)
    request.session.pop("azure_user", None)
    if request.session.pop("azure_link_ab", None) != ab["oid"]:
        raise HTTPException(status_code=403, detail="AB session changed; restart AZP login")
    client = oauth.create_client("entra")
    if client is None:
        raise HTTPException(status_code=503, detail="Entra authentication is not configured")
    token = await client.authorize_access_token(request)
    claims = token.get("userinfo")
    if not isinstance(claims, dict):
        raise HTTPException(status_code=401, detail="Entra did not return valid identity claims")
    azp = identity_claims(claims, request.app.state.settings, "azp")
    if azp["oid"] == ab["oid"] or ab_account_suffix(ab["email"]) != account_suffix(
        azp["email"], "azp"
    ):
        raise HTTPException(
            status_code=403, detail="AZP account does not match the signed-in AB user"
        )
    hours = request.app.state.settings.session_hours
    logger.warning(
        "AZP connected for %s h; id token exp in %.0f s",
        hours,
        float(claims.get("exp", 0)) - time.time(),
    )
    request.session["azure_user"] = {
        **azp,
        "ab_oid": ab["oid"],
        "expires_at": time.time() + hours * 3600,
    }
    return RedirectResponse(url=request.app.state.settings.frontend_url or "/", status_code=303)


@router.post("/azure/disconnect")
async def azure_disconnect(request: Request) -> dict:
    current_user(request)
    require_csrf(request)
    request.session.pop("azure_user", None)
    request.session.pop("azure_link_ab", None)
    return {"ok": True}


@router.post("/logout")
async def logout(request: Request) -> dict[str, bool]:
    user = current_user(request)
    require_csrf(request)
    connection = request.session.get("github_connection", "")
    if re.fullmatch(r"[a-f0-9]{32}", connection):
        from cloudops_hub.operations_store import owner_key

        request.app.state.operations.delete(f"tokens/{owner_key(user['oid'])}/{connection}.json")
    request.session.clear()
    return {"ok": True}
