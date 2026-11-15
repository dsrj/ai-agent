"""User-attributed operations for one fixed service-catalog repository."""

import base64
import hashlib
import json
import re
import secrets
import time
from contextlib import contextmanager
from typing import Any, Literal
from urllib.parse import urlencode, urlparse
from uuid import UUID

import httpx
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from cryptography.fernet import InvalidToken
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field
from starlette.responses import RedirectResponse

from cloudops_hub.auth import account_suffix, current_user, require_csrf
from cloudops_hub.operations_store import owner_key, timestamp

REPOSITORY = "almbrand/landingzone-service-catalog"
BASE = f"/repos/{REPOSITORY}"
WORKFLOW = "caller.yml"
WORKFLOW_SHA = "682163527247e800e0ae50744aea555a6faacfc6"
SHORT_TEAM = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
MODES = {
    "DEFAULT-[only changed workspaces]": "Changed workspaces",
    "RUN_ALL-[All workspaces]": "All workspaces",
    "GITHUBREPOS-[team.json]": "GitHub repositories",
    "APPS-[spns.json]": "Applications / SPNs",
    "GROUPS-[groups.json]": "Groups",
    "SFTP-[sftp.json]": "SFTP",
    "ONPREM_ACCESS-[onpremaccess.json]": "On-prem access",
    "PUBLICNETWORKACCESS-[publicnetworkaccess.json]": "Public network access",
    "CERTS-[certs.json]": "Certificates",
    "IDNS-[idns.json]": "Identities",
    "GSA-[gsa.json]": "Global Secure Access",
    "NSP-[nsp.json]": "Network Security Perimeter",
    "PIM-[pim.json]": "PIM",
    "CUSTOM": "Custom (single workspace)",
}
WORKSPACES = (
    "backend",
    "cloudops",
    "githubaccess",
    "branchprotection",
    "githubrepos",
    "githubenvs",
    "githubapp",
    "githubspns",
    "githubteam",
    "groups",
    "idns",
    "members",
    "rbac",
    "webapp",
    "devops",
    "githuboptional",
    "approles",
    "certs",
    "sftp",
    "gsa",
    "nsp",
    "onpremaccess",
    "keyvault",
    "publicnetworkaccess",
    "ghclientid",
    "ghsubscriptionid",
    "ghtenantid",
    "ghclientsecret",
    "ghcustomsecrets",
    "pim",
)
router = APIRouter(tags=["service-catalog"])


class CatalogError(RuntimeError):
    pass


class GitHubRefused(CatalogError):
    """GitHub declined a review/merge action; the message is GitHub's own reason."""


class GitHubNotFound(CatalogError):
    pass


def _bounded(response: httpx.Response, limit: int) -> bytes:
    content = bytearray()
    for block in response.iter_bytes():
        content.extend(block)
        if len(content) > limit:
            raise CatalogError("GitHub job log exceeds the safe read limit")
    return bytes(content)


def signed_log_url(location: str) -> bool:
    try:
        parsed = urlparse(location)
        return bool(
            parsed.scheme == "https"
            and parsed.hostname
            and parsed.hostname.endswith(
                (".blob.core.windows.net", ".actions.githubusercontent.com")
            )
            and parsed.port in (None, 443)
            and not parsed.username
            and not parsed.password
            and not parsed.fragment
        )
    except ValueError:
        return False


class GitHub:
    def __init__(self, token: str):
        self.client = httpx.Client(
            base_url="https://api.github.com",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            follow_redirects=False,
            timeout=30,
        )

    def call(self, method: str, path: str, *, body=None, params=None):
        routes = {
            "GET": [
                r"/user",
                r"/user/teams",
                r"/search/repositories",
                r"/repos/almbrand/[A-Za-z0-9_.-]+(?:/environments"
                r"|/environments/[A-Za-z0-9_.-]+/secrets|/teams)?",
                r"/orgs/almbrand/teams/[a-z0-9][a-z0-9_-]*/memberships/[a-zA-Z0-9_-]+",
                re.escape(BASE)
                + r"(?:/git/ref/heads/main|/contents/"
                + CONTENT_FILE
                + r"|/pulls/[0-9]+|/actions/runs/[0-9]+"
                r"|/actions/runs/[0-9]+/attempts/[0-9]+/jobs"
                r"|/actions/workflows/caller\.yml/runs"
                r"|/pulls|/pulls/[0-9]+/(?:files|reviews)|/actions/runs"
                r"|/actions/runs/[0-9]+/pending_deployments)?",
            ],
            "POST": [
                re.escape(BASE) + r"/(?:git/refs|pulls"
                r"|actions/workflows/(?:caller|state-list|state-rm|import)\.yml/dispatches"
                r"|actions/jobs/[0-9]+/rerun|actions/runs/[0-9]+/rerun-failed-jobs"
                r"|actions/runs/[0-9]+/cancel"
                r"|pulls/[0-9]+/reviews|issues/[0-9]+/comments"
                r"|actions/runs/[0-9]+/pending_deployments)",
            ],
            "PUT": [
                re.escape(BASE) + r"/contents/(?!\.github/|CODEOWNERS$)" + CONTENT_FILE,
                re.escape(BASE) + r"/pulls/[0-9]+/merge",
            ],
        }
        if ".." in path or not any(
            re.fullmatch(pattern, path) for pattern in routes.get(method, [])
        ):
            raise CatalogError("Service Catalog rejected an unapproved GitHub route")
        try:
            response = self.client.request(method, path, json=body, params=params)
        except httpx.HTTPError:
            raise CatalogError(
                "GitHub request failed; outcome may be unknown. Refresh before trying again."
            ) from None
        if response.status_code == 404:
            raise GitHubNotFound("GitHub resource not found or access denied")
        if response.status_code in {403, 405, 409, 422} and re.search(
            r"/(?:pulls/[0-9]+/(?:reviews|merge)|issues/[0-9]+/comments|pending_deployments"
            r"|cancel)$",
            path,
        ):
            try:
                reason = str(response.json().get("message", ""))
            except (ValueError, AttributeError):
                reason = ""
            reason = re.sub(r"[^\w .,:;'()/@#-]", "", reason)[:300]
            status = response.status_code
            raise GitHubRefused(reason or f"GitHub refused the action (HTTP {status})")
        if response.status_code not in {200, 201, 202, 204}:
            raise CatalogError(
                f"GitHub returned HTTP {response.status_code}. Check repository access, "
                "GitHub App permissions and organization SSO; no approval is bypassed."
            )
        try:
            return response.json() if response.content else None
        except ValueError:
            raise CatalogError("GitHub returned an invalid JSON response") from None

    def job_log(self, job_id: int, limit: int = 2_000_000) -> str:
        """Bounded job-log read; the signed download never receives the GitHub token."""
        if type(job_id) is not int or job_id < 1:
            raise CatalogError("Invalid workflow job")
        try:
            with self.client.stream("GET", f"{BASE}/actions/jobs/{job_id}/logs") as response:
                if response.status_code == 200:
                    raw = _bounded(response, limit)
                elif response.status_code == 302:
                    location = response.headers.get("location", "")
                    if not signed_log_url(location):
                        raise CatalogError("GitHub returned an unapproved log download host")
                    with httpx.Client(timeout=30, follow_redirects=False) as media:
                        with media.stream("GET", location) as download:
                            if download.status_code != 200:
                                raise CatalogError("GitHub job log download is unavailable")
                            raw = _bounded(download, limit)
                else:
                    raise CatalogError(
                        f"GitHub job logs returned HTTP {response.status_code}; logs may have "
                        "expired or Actions read permission may be missing"
                    )
        except httpx.HTTPError:
            raise CatalogError("GitHub job log download failed") from None
        return raw.decode("utf-8", errors="replace")

    def close(self):
        self.client.close()

    def teams(self) -> list[dict]:
        """Catalog folders whose CODEOWNERS owner team includes the user (fails closed)."""
        slugs: set[str] = set()
        for page in range(1, 21):
            values = self.call("GET", "/user/teams", params={"per_page": 100, "page": page})
            if not isinstance(values, list):
                raise CatalogError("GitHub returned an invalid team list")
            for item in values:
                if item.get("organization", {}).get("login", "").casefold() == "almbrand":
                    slugs.add(str(item.get("slug", "")).casefold())
            if len(values) < 100:
                break
        else:
            raise CatalogError("GitHub team pagination exceeded its limit; access was not inferred")
        result = []
        for folder, owners in codeowners(self).items():
            owner = next((slug for slug in owners if slug in slugs), None)
            if owner:
                result.append({"name": folder, "slug": owner})
        return sorted(result, key=lambda item: item["name"])

    def authorize(self, team: str) -> dict:
        if not SHORT_TEAM.fullmatch(team):
            raise HTTPException(status_code=403, detail="Choose a team from Allowed teams")
        owned = next((item for item in self.teams() if item["name"] == team), None)
        if owned is None:
            raise HTTPException(
                status_code=403,
                detail=f"You are not in the CODEOWNERS team for {team}. Request Omada access "
                "to that team's GitHub maintain group, then refresh after GitHub membership "
                "synchronizes.",
            )
        actor = self.call("GET", "/user")
        login = actor.get("login", "")
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", login):
            raise CatalogError("GitHub did not return a valid user identity")
        membership = self.call("GET", f"/orgs/almbrand/teams/{owned['slug']}/memberships/{login}")
        if membership.get("state") != "active":
            raise HTTPException(
                status_code=403, detail="Active maintain-team membership is required"
            )
        permissions = self.call("GET", BASE).get("permissions", {})
        if not permissions.get("push"):
            raise HTTPException(
                status_code=403, detail="GitHub repository write permission is required"
            )
        return {"id": actor["id"], "login": login}


def catalog_user(request: Request) -> dict:
    user = current_user(request)
    if request.method != "GET":
        require_csrf(request)
    return user


def client_secret(settings) -> str:
    if settings.github_catalog_client_secret:
        return settings.github_catalog_client_secret
    if not settings.catalog_live_vault_url:
        raise HTTPException(status_code=503, detail="GitHub App credentials are not configured")
    with DefaultAzureCredential() as credential:
        with SecretClient(settings.catalog_live_vault_url, credential) as vault:
            value = vault.get_secret("github-catalog-client-secret").value
    if not value:
        raise HTTPException(status_code=503, detail="GitHub App client secret is empty")
    return value


def expected_github_login(user: dict) -> str:
    return f"ab{account_suffix(user.get('email', ''), 'ab')}_gheab"


def matching_login(user: dict, login) -> bool:
    return isinstance(login, str) and login.casefold() == expected_github_login(user)


@contextmanager
def user_github(request: Request, user: dict):
    connection = request.session.get("github_connection")
    if not isinstance(connection, str) or not re.fullmatch(r"[a-f0-9]{32}", connection):
        raise HTTPException(status_code=403, detail="Connect GitHub with your AB account first")
    try:
        token = request.app.state.operations.token(user["oid"], connection)
    except InvalidToken:
        raise HTTPException(
            status_code=403, detail="GitHub connection key changed; reconnect GitHub"
        ) from None
    if not token or token["expires_at"] <= time.time():
        raise HTTPException(status_code=403, detail="GitHub connection expired; reconnect GitHub")
    if not matching_login(user, token.get("login")):
        raise HTTPException(
            status_code=403,
            detail=f"Reconnect GitHub as {expected_github_login(user)}",
        )
    client = GitHub(token["access_token"])
    try:
        yield client
    finally:
        client.close()


@router.get("/auth/github/login")
def github_login(request: Request, destination: str = ""):
    user = current_user(request)
    settings = request.app.state.settings
    if not settings.github_catalog_client_id:
        raise HTTPException(
            status_code=503, detail="Configure the Service Catalog GitHub App first"
        )
    expected = expected_github_login(user)
    state = secrets.token_urlsafe(32)
    request.session["github_oauth"] = {
        "state": state,
        "ab_oid": user["oid"],
        "expires_at": time.time() + 600,
        "destination": "landing-zone" if destination == "landing-zone" else "catalog",
    }
    return RedirectResponse(
        "https://github.com/login/oauth/authorize?"
        + urlencode(
            {
                "client_id": settings.github_catalog_client_id,
                "redirect_uri": f"{settings.frontend_url}/auth/github/callback",
                "state": state,
                "login": expected,
                "allow_signup": "false",
            }
        )
    )


@router.get("/auth/github/callback")
def github_callback(request: Request, state: str = "", code: str = ""):
    user = current_user(request)
    saved = request.session.pop("github_oauth", {})
    if (
        not code
        or len(code) > 256
        or not saved
        or not secrets.compare_digest(saved.get("state", ""), state)
        or saved.get("ab_oid") != user["oid"]
        or saved.get("expires_at", 0) <= time.time()
    ):
        raise HTTPException(status_code=403, detail="GitHub authorization expired; connect again")
    settings = request.app.state.settings
    try:
        with httpx.Client(timeout=30, follow_redirects=False) as client:
            response = client.post(
                "https://github.com/login/oauth/access_token",
                headers={"Accept": "application/json"},
                data={
                    "client_id": settings.github_catalog_client_id,
                    "client_secret": client_secret(settings),
                    "code": code,
                    "redirect_uri": f"{settings.frontend_url}/auth/github/callback",
                },
            )
    except httpx.HTTPError:
        raise CatalogError("GitHub authorization is unavailable; reconnect later") from None
    if response.status_code != 200:
        raise CatalogError("GitHub authorization failed; reconnect GitHub")
    try:
        token = response.json()
    except ValueError:
        raise CatalogError("GitHub returned an invalid authorization response") from None
    if not isinstance(token, dict):
        raise CatalogError("GitHub returned an invalid authorization response")
    if not token.get("access_token") or token.get("token_type", "").casefold() != "bearer":
        raise CatalogError("GitHub did not grant a user access token")
    # Require expiring GitHub App tokens; never accept a classic PAT/OAuth App scope token.
    expires = token.get("expires_in")
    if not isinstance(expires, int) or not 0 < expires <= 28800 or token.get("scope"):
        raise CatalogError("Configure expiring user access tokens on the repository GitHub App")
    github = GitHub(token["access_token"])
    try:
        login = github.call("GET", "/user").get("login")
    finally:
        github.close()
    if not matching_login(user, login):
        raise HTTPException(
            status_code=403,
            detail=f"GitHub account {login or 'unknown'} does not belong to you. Sign out of "
            f"GitHub and connect as {expected_github_login(user)}.",
        )
    connection = secrets.token_hex(16)
    request.app.state.operations.save_token(
        user["oid"],
        connection,
        {
            "access_token": token["access_token"],
            "expires_at": time.time() + expires,
            "login": login,
        },
    )
    old = request.session.get("github_connection")
    if isinstance(old, str) and re.fullmatch(r"[a-f0-9]{32}", old):
        request.app.state.operations.delete(f"tokens/{owner_key(user['oid'])}/{old}.json")
    request.session["github_connection"] = connection
    target = "?github_connected=1" if saved.get("destination") == "landing-zone" else "?catalog=1"
    return RedirectResponse(f"{settings.frontend_url}/{target}", status_code=303)


@router.post("/api/catalog/disconnect")
def disconnect_github(request: Request, user: dict = Depends(catalog_user)):
    connection = request.session.pop("github_connection", "")
    if re.fullmatch(r"[a-f0-9]{32}", connection):
        request.app.state.operations.delete(f"tokens/{owner_key(user['oid'])}/{connection}.json")
    return {"ok": True}


@router.get("/api/catalog")
def catalog_status(request: Request, user: dict = Depends(catalog_user)):
    settings = request.app.state.settings
    configured = bool(settings.github_catalog_client_id)
    result = {
        "repository": REPOSITORY,
        "configured": configured,
        "connected": False,
        "teams": [],
        "modes": [{"value": key, "label": value} for key, value in MODES.items()],
        "workspaces": list(WORKSPACES),
    }
    if not request.session.get("github_connection"):
        return result
    with user_github(request, user) as github:
        actor = github.call("GET", "/user")
        return {
            **result,
            "connected": True,
            "github_login": actor["login"],
            "teams": github.teams(),
        }


class GroupValues(BaseModel):
    model_config = ConfigDict(extra="forbid")
    group_name: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,79}$")
    description: str = Field(min_length=1, max_length=500)
    access_level: Literal["restricted", "unrestricted"] = "restricted"


class Submission(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request_id: UUID
    team: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$", max_length=40)
    operation: Literal["create-group", "create-resource", "dispatch"]
    action: Literal["create", "update", "delete"] = "create"
    group: GroupValues | None = None
    resource_type: str | None = Field(default=None, max_length=64)
    entry: dict[str, Any] | None = None
    entries: list[dict[str, Any]] | None = Field(default=None, min_length=1, max_length=50)
    mode: str = "GROUPS-[groups.json]"
    workspace: str | None = Field(default=None, max_length=40)
    github_refresh: bool = False
    disable_refresh: bool = False
    confirmed: Literal[True]


SHARED_TEAM = "cloudops"


def resource_types() -> dict[str, dict]:
    """Every reviewed catalog schema as a single-entry form and its target file."""
    from cloudops_hub.catalog import catalog_schemas

    types = {}
    for schema_id, schema in catalog_schemas().items():
        path = list(schema.get("x-entry-path", []))
        node = schema
        for part in path:
            node = node["properties"][part]
        append = node.get("type") == "array"
        target = schema["x-target-path"]
        types[schema_id] = {
            "id": schema_id,
            "title": schema["title"].removesuffix(" (reviewed subset)"),
            "description": schema.get("description", ""),
            "target_path": target,
            "entry_path": path,
            "kind": "append" if append else "replace",
            "identity_key": schema.get("x-identity-key") if append else None,
            "shared": "<team>" not in target,
            "entry_schema": node["items"] if append else node,
            "sources": schema.get("x-sources", []),
        }
    return types


CONTENT_FILE = (
    r"(?:\.github/workflows/(?:caller|state-list|state-rm|import)\.yml|CODEOWNERS|(?P<dir>[a-z0-9-]+)/(?P=dir)(?:_groups|_spns|_group_members"
    r"|_idns|_certs|_tags|_onprem_access|_sftp|_nsp|_publicnetworkaccess|_gsa|_pim"
    r"|_github_rulesets)?\.json|cloudops/cloudops_sftp_network_rules\.json"
    r"|cloudops/whitelist_adgroups\.json)"
)


def resource_target(kind: dict, team: str) -> str:
    if kind["shared"] and team != SHARED_TEAM:
        raise HTTPException(
            status_code=403, detail="This shared catalog file can only be requested by cloudops"
        )
    target = kind["target_path"].replace("<team>", team)
    if not re.fullmatch(CONTENT_FILE, target) or target.startswith(".github/"):
        raise HTTPException(status_code=422, detail="Unsupported catalog target file")
    return target


def identity_value(schema_id: str, value) -> str:
    text = str(value).casefold()
    if schema_id == "catalog-applications":
        return re.sub(r"\s+", "-", text)
    if schema_id == "catalog-sftp":
        return text.replace("-", "")
    return text


def validate_entry(kind: dict, entry: dict) -> None:
    from jsonschema import Draft202012Validator

    from cloudops_hub.catalog import duplicate_errors

    if len(json.dumps(entry)) > 100_000:
        raise HTTPException(status_code=422, detail="Catalog entry exceeds the size limit")
    errors = sorted(
        Draft202012Validator(kind["entry_schema"]).iter_errors(entry),
        key=lambda error: list(map(str, error.absolute_path)),
    )
    messages = [
        f"/{'/'.join(map(str, error.absolute_path))}: {error.message}" for error in errors[:10]
    ]
    if not messages and kind["kind"] == "append":
        document: dict | list = [entry]
        for part in reversed(kind["entry_path"]):
            document = {part: document}
        messages = [
            f"{item['path']}: {item['message']}" for item in duplicate_errors(kind["id"], document)
        ]
    if messages:
        raise HTTPException(status_code=422, detail="Invalid catalog entry. " + "; ".join(messages))


def catalog_document(raw: bytes) -> dict:
    if len(raw) > 1_000_000:
        raise CatalogError("Shared catalog file exceeds the reviewed size limit")
    try:
        document = json.loads(raw)
    except ValueError:
        raise CatalogError("Existing catalog file is not valid JSON") from None
    if not isinstance(document, dict):
        raise CatalogError("Existing catalog file has an unsupported shape")
    return document


def resource_update(raw: bytes, kind: dict, entry: dict, update: bool = False) -> dict | list:
    document = catalog_document(raw)
    path = kind["entry_path"]
    parent = document
    for part in path[:-1]:
        parent = parent.setdefault(part, {})
        if not isinstance(parent, dict):
            raise CatalogError("Existing catalog file has an unsupported shape")
    last = path[-1]
    if kind["kind"] == "replace":
        parent[last] = entry
        return document
    items = parent.setdefault(last, [])
    if not isinstance(items, list):
        raise CatalogError("Existing catalog file has an unsupported shape")
    key = kind["identity_key"]
    matches = [
        index
        for index, item in enumerate(items)
        if key
        and isinstance(item, dict)
        and identity_value(kind["id"], item.get(key, "")) == identity_value(kind["id"], entry[key])
    ]
    if update:
        if not key or not matches:
            raise HTTPException(
                status_code=404, detail=f"No existing entry with this {key}; create it instead"
            )
        if items[matches[0]] == entry:
            raise HTTPException(status_code=422, detail="No changes to submit")
        items[matches[0]] = entry
        return document
    if matches:
        raise HTTPException(status_code=409, detail=f"An entry with this {key} already exists")
    items.append(entry)
    return document


def resource_name(kind: dict, entry: dict) -> str:
    key = kind["identity_key"]
    return str(entry[key]) if key and key in entry else kind["title"]


def resource_remove(raw: bytes, kind: dict, entry: dict) -> dict:
    document = catalog_document(raw)
    parent = document
    for part in kind["entry_path"][:-1]:
        parent = parent.get(part)
        if not isinstance(parent, dict):
            raise CatalogError("Existing catalog file has an unsupported shape")
    items = parent.get(kind["entry_path"][-1])
    if not isinstance(items, list):
        raise CatalogError("Existing catalog file has an unsupported shape")
    key = kind["identity_key"]
    matches = [
        index
        for index, item in enumerate(items)
        if isinstance(item, dict)
        and identity_value(kind["id"], item.get(key, "")) == identity_value(kind["id"], entry[key])
    ]
    if not matches:
        raise HTTPException(status_code=404, detail="The selected entry no longer exists")
    if len(matches) != 1 or items[matches[0]] != entry:
        raise HTTPException(
            status_code=409, detail="The selected entry changed; reload it and review again"
        )
    del items[matches[0]]
    return document


def create_resource(github: GitHub, submission: Submission, record: dict, store, path: str):
    kind = resource_types()[submission.resource_type]
    team = submission.team
    filename = resource_target(kind, team)
    update = submission.action == "update"
    remove = submission.action == "delete"
    verb = "Remove" if remove else "Update" if update else "Request"
    entries = submission.entries or [submission.entry]
    base_sha = github.call("GET", f"{BASE}/git/ref/heads/main")["object"]["sha"]
    try:
        source = github.call("GET", f"{BASE}/contents/{filename}", params={"ref": base_sha})
    except GitHubNotFound:
        source = None
    if source is None:
        if update or remove:
            raise HTTPException(status_code=404, detail="The catalog file does not exist yet")
        raw = b"{}"
    else:
        if source.get("type") != "file" or source.get("encoding") != "base64":
            raise CatalogError("Existing catalog target is not a supported file")
        if source.get("size", 1_000_001) > 1_000_000:
            raise CatalogError("Shared catalog file exceeds the reviewed size limit")
        raw = base64.b64decode(source["content"], validate=False)
    for entry in entries:
        document = (
            resource_remove(raw, kind, entry)
            if remove
            else resource_update(raw, kind, entry, update)
        )
        raw = json.dumps(document).encode()
    if github.call("GET", f"{BASE}/git/ref/heads/main")["object"]["sha"] != base_sha:
        raise HTTPException(status_code=409, detail="Catalog changed; refresh and review again")
    name = f"{len(entries)} entries" if len(entries) > 1 else resource_name(kind, submission.entry)
    branch = f"hub/request-{submission.request_id.hex}"
    github.call("POST", f"{BASE}/git/refs", body={"ref": f"refs/heads/{branch}", "sha": base_sha})
    record.update(branch=branch, base_sha=base_sha, status="branch_created")
    store.write(path, record)
    commit = github.call(
        "PUT",
        f"{BASE}/contents/{filename}",
        body={
            "message": f"{verb} {kind['title']}: {name} for {team}",
            "content": base64.b64encode((json.dumps(document, indent=2) + "\n").encode()).decode(),
            **({"sha": source["sha"]} if source is not None else {}),
            "branch": branch,
        },
    )
    record.update(commit_sha=commit["commit"]["sha"], status="catalog_updated")
    store.write(path, record)
    change = (
        f"Remove the existing `{name}` entry from `{'.'.join(kind['entry_path'])}` of "
        f"`{filename}`. Other entries and file sections are preserved. "
        "Terraform may destroy the managed resource and related resources after merge and "
        "the normal plan/apply approvals. This is not a state-only removal."
        if remove
        else f"Update the existing `{name}` entry in `{'.'.join(kind['entry_path'])}` of "
        f"`{filename}`. Other entries are preserved."
        if update and kind["kind"] == "append"
        else f"Append {len(entries)} entry/entries to `{'.'.join(kind['entry_path'])}` "
        f"in `{filename}` "
        "(create the file if absent). Existing entries are preserved."
        if kind["kind"] == "append"
        else f"Replace the `{'.'.join(kind['entry_path'])}` section of `{filename}`. "
        "Reviewers must check the diff for removed values."
    )
    pr = github.call(
        "POST",
        f"{BASE}/pulls",
        body={
            "title": f"{'Remove ' if remove else 'Update ' if update else ''}"
            f"{kind['title']}: {name} for {team}",
            "head": branch,
            "base": "main",
            "body": f"Service Catalog request `{record['id']}` ({kind['id']}).\n\n{change} "
            "Another team member must review this PR. After merge, the existing "
            "service-catalog workflow controls plan and apply approvals. The Hub does not "
            "approve, merge or apply.",
        },
    )
    record.update(pr_number=pr["number"], pr_url=pr["html_url"], status="awaiting_review")
    store.write(path, record)


def group_update(raw: bytes, values: GroupValues) -> dict:
    if len(raw) > 1_000_000:
        raise CatalogError("Shared catalog file exceeds the reviewed size limit")
    try:
        document = json.loads(raw)
    except ValueError:
        raise CatalogError("Existing shared groups file is not valid JSON") from None
    if (
        not isinstance(document, dict)
        or not isinstance(document.get("groups"), dict)
        or not isinstance(document["groups"].get("instances"), list)
        or not all(
            isinstance(item, dict)
            and isinstance(item.get("group_name"), str)
            and item["group_name"]
            for item in document["groups"]["instances"]
        )
    ):
        raise CatalogError("Existing shared groups file has an unsupported shape")
    if any(
        item.get("group_name", "").casefold() == values.group_name.casefold()
        for item in document["groups"]["instances"]
    ):
        raise HTTPException(status_code=409, detail="A group with this name already exists")
    document["groups"]["instances"].append(values.model_dump())
    return document


def request_path(user: dict, identifier: UUID | str) -> str:
    return f"requests/{owner_key(user['oid'])}/{UUID(str(identifier)).hex}.json"


def create_group(github: GitHub, submission: Submission, record: dict, store, path: str):
    team = submission.team
    filename = f"{team}/{team}_groups.json"
    base_sha = github.call("GET", f"{BASE}/git/ref/heads/main")["object"]["sha"]
    try:
        source = github.call("GET", f"{BASE}/contents/{filename}", params={"ref": base_sha})
    except GitHubNotFound:
        source = None
    if source is None:
        document = group_update(b'{"groups":{"instances":[]}}', submission.group)
    else:
        if source.get("type") != "file" or source.get("encoding") != "base64":
            raise CatalogError("Existing shared groups catalog is not a supported file")
        if source.get("size", 1_000_001) > 1_000_000:
            raise CatalogError("Shared catalog file exceeds the reviewed size limit")
        document = group_update(
            base64.b64decode(source["content"], validate=False), submission.group
        )
    # A changed main requires a new review; never silently update against stale input.
    if github.call("GET", f"{BASE}/git/ref/heads/main")["object"]["sha"] != base_sha:
        raise HTTPException(status_code=409, detail="Catalog changed; refresh and review again")
    branch = f"hub/request-{submission.request_id.hex}"
    github.call("POST", f"{BASE}/git/refs", body={"ref": f"refs/heads/{branch}", "sha": base_sha})
    record.update(branch=branch, base_sha=base_sha, status="branch_created")
    store.write(path, record)
    commit = github.call(
        "PUT",
        f"{BASE}/contents/{filename}",
        body={
            "message": f"Request group {submission.group.group_name} for {team}",
            "content": base64.b64encode((json.dumps(document, indent=2) + "\n").encode()).decode(),
            **({"sha": source["sha"]} if source is not None else {}),
            "branch": branch,
        },
    )
    record.update(commit_sha=commit["commit"]["sha"], status="catalog_updated")
    store.write(path, record)
    pr = github.call(
        "POST",
        f"{BASE}/pulls",
        body={
            "title": f"Create group {submission.group.group_name} for {team}",
            "head": branch,
            "base": "main",
            "body": f"Service Catalog request `{record['id']}`.\n\n"
            f"Append one group entry to `{filename}` (create the file if absent). "
            "Existing entries are preserved. Another team member must review this PR. "
            "After merge, the existing service-catalog workflow controls plan and apply "
            "approvals. The Hub does not approve, merge or apply.",
        },
    )
    record.update(pr_number=pr["number"], pr_url=pr["html_url"], status="awaiting_review")
    store.write(path, record)


def dispatch_workflow(
    github: GitHub,
    team: str,
    mode: str,
    workspace: str | None = None,
    github_refresh: bool = False,
    disable_refresh: bool = False,
):
    """Dispatch the reviewed caller workflow; returns the run ID when GitHub provides it."""
    definition = github.call(
        "GET", f"{BASE}/contents/.github/workflows/{WORKFLOW}", params={"ref": "main"}
    )
    if definition.get("sha") != WORKFLOW_SHA:
        raise CatalogError("Workflow contract changed; CloudOps must review it before Hub dispatch")
    inputs = {
        "environment": team,
        "WORKFLOW_MODE": mode,
        "GITHUB_RESOURCE_REFRESH": "true" if github_refresh else "false",
        "DISABLE_ALL_REFRESH": "true" if disable_refresh else "false",
    }
    if workspace is not None:
        inputs["WORKSPACE_FOLDER"] = workspace
    result = github.call(
        "POST",
        f"{BASE}/actions/workflows/{WORKFLOW}/dispatches",
        body={"ref": "main", "inputs": inputs, "return_run_details": True},
    )
    run_id = result.get("workflow_run_id") if isinstance(result, dict) else None
    return run_id if type(run_id) is int and run_id > 0 else None


def dispatch(github: GitHub, submission: Submission, record: dict):
    if submission.mode not in MODES:
        raise HTTPException(status_code=422, detail="Unsupported workflow mode")
    if (
        submission.team == "oldsetup"
        and submission.mode != "PUBLICNETWORKACCESS-[publicnetworkaccess.json]"
    ):
        raise HTTPException(status_code=422, detail="oldsetup supports only Public network access")
    record["dispatched_at"] = timestamp()
    run_id = dispatch_workflow(
        github,
        submission.team,
        submission.mode,
        submission.workspace,
        submission.github_refresh,
        submission.disable_refresh,
    )
    if run_id:
        record["dispatched_run_id"] = run_id
    record.update(status="dispatch_accepted")


@router.get("/api/catalog/entries")
def catalog_entries(
    team: str, resource_type: str, request: Request, user: dict = Depends(catalog_user)
):
    """Current entries of the team's catalog file on main, used to pre-fill edit requests."""
    kind = resource_types().get(resource_type)
    if kind is None:
        raise HTTPException(status_code=404, detail="Unknown resource type")
    if not SHORT_TEAM.fullmatch(team):
        raise HTTPException(status_code=422, detail="Choose a team from Allowed teams")
    with user_github(request, user) as github:
        if not any(item["name"] == team for item in github.teams()):
            raise HTTPException(status_code=403, detail="Choose a team from Allowed teams")
        return read_entries(github, kind, team)


def read_entries(github: GitHub, kind: dict, team: str) -> dict:
    """Entries of one team's catalog file on main; callers must have authorized the team."""
    filename = resource_target(kind, team)
    result = {"target_path": filename, "exists": False, "items": [], "section": None}
    try:
        source = github.call("GET", f"{BASE}/contents/{filename}", params={"ref": "main"})
    except GitHubNotFound:
        return result
    if source.get("encoding") != "base64" or source.get("size", 1_000_001) > 1_000_000:
        raise HTTPException(status_code=422, detail="Catalog file is not a supported file")
    try:
        node = catalog_document(base64.b64decode(source["content"], validate=False))
    except CatalogError as error:
        raise HTTPException(status_code=422, detail=str(error)) from None
    for part in kind["entry_path"]:
        node = node.get(part) if isinstance(node, dict) else None
    result["exists"] = True
    if kind["kind"] == "append":
        key = kind["identity_key"]
        result["items"] = (
            [item for item in node or [] if isinstance(item, dict) and (not key or key in item)]
            if isinstance(node, list)
            else []
        )
    else:
        result["section"] = node
    return result


@router.get("/api/catalog/resource-types")
def catalog_resource_types(request: Request, user: dict = Depends(catalog_user)):
    try:
        with user_github(request, user) as github:
            teams = catalog_teams(github)
    except (CatalogError, HTTPException):
        teams = []
    return {"items": list(resource_types().values()), "shared_team": SHARED_TEAM, "teams": teams}


_teams_cache: dict[str, tuple[float, dict[str, list[str]]]] = {}
CODEOWNER_LINE = re.compile(r"^/([a-z0-9]+(?:-[a-z0-9]+)*)/(?=[\s@])(.*)$", re.MULTILINE)
CODEOWNER = re.compile(r"@almbrand/([A-Za-z0-9][A-Za-z0-9_.-]*)")
TEAM_FIELDS = {"teams", "allowed_teams"}


def codeowners(github: GitHub) -> dict[str, list[str]]:
    """Folder -> owner team slugs from CODEOWNERS on main; the last matching line wins."""
    cached = _teams_cache.get("codeowners")
    if cached and cached[0] > time.monotonic():
        return cached[1]
    source = github.call("GET", f"{BASE}/contents/CODEOWNERS", params={"ref": "main"})
    if (
        not isinstance(source, dict)
        or source.get("encoding") != "base64"
        or source.get("size", 200_001) > 200_000
    ):
        raise CatalogError("CODEOWNERS could not be read; team access was not inferred")
    text = base64.b64decode(source["content"], validate=False).decode("utf-8", "replace")
    owners = {}
    for folder, rest in CODEOWNER_LINE.findall(text):
        teams = [team.casefold() for team in CODEOWNER.findall(rest)]
        if teams:
            owners[folder] = teams
    if not owners:
        raise CatalogError("CODEOWNERS lists no team folders; team access was not inferred")
    _teams_cache["codeowners"] = (time.monotonic() + 600, owners)
    return owners


def catalog_teams(github: GitHub) -> list[str]:
    """Team folders listed in the catalog repository's CODEOWNERS (first path segment)."""
    return sorted(codeowners(github))


def entry_teams(value) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in TEAM_FIELDS:
                items = item if isinstance(item, list) else [item]
                found.update(str(team) for team in items)
            else:
                found |= entry_teams(item)
    elif isinstance(value, list):
        for item in value:
            found |= entry_teams(item)
    return found


@router.post("/api/catalog/requests", status_code=202)
def submit(payload: Submission, request: Request, user: dict = Depends(catalog_user)):
    if payload.entries is not None:
        if payload.operation != "create-resource" or payload.entry is not None:
            raise HTTPException(
                status_code=422, detail="Supply either entry or entries for catalog PRs"
            )
        if payload.action != "create":
            raise HTTPException(status_code=422, detail="Multiple entries support creation only")
        batch_kind = resource_types().get(payload.resource_type or "")
        if batch_kind is None or batch_kind["kind"] != "append":
            raise HTTPException(
                status_code=422, detail="This resource type does not support multiple entries"
            )
        raw = b"{}"
        for item in payload.entries:
            validate_entry(batch_kind, item)
            raw = json.dumps(resource_update(raw, batch_kind, item)).encode()
        if len(raw) > 100_000:
            raise HTTPException(status_code=422, detail="Catalog entries exceed the size limit")
        payload.entry = payload.entries[0]
    if payload.operation == "create-group" and payload.group is None:
        raise HTTPException(status_code=422, detail="Group details are required")
    if payload.operation != "create-resource" and (
        payload.resource_type is not None or payload.entry is not None
    ):
        raise HTTPException(status_code=422, detail="Resource details apply only to catalog PRs")
    if payload.action != "create" and payload.operation != "create-resource":
        raise HTTPException(status_code=422, detail="Only catalog PRs can update or remove entries")
    kind = None
    if payload.operation == "create-resource":
        kind = resource_types().get(payload.resource_type or "")
        if kind is None or payload.entry is None or payload.group is not None:
            raise HTTPException(status_code=422, detail="Choose a resource type and its values")
        if payload.action == "update" and kind["kind"] == "append" and not kind["identity_key"]:
            raise HTTPException(status_code=422, detail="This resource type cannot be edited")
        if payload.action == "delete":
            key = kind["identity_key"]
            if kind["kind"] != "append" or not key:
                raise HTTPException(
                    status_code=422, detail="Only entries with a unique identity can be removed"
                )
            if not isinstance(payload.entry.get(key), str) or not payload.entry[key].strip():
                raise HTTPException(status_code=422, detail="Select an existing entry to remove")
            if len(json.dumps(payload.entry)) > 100_000:
                raise HTTPException(status_code=422, detail="Catalog entry exceeds the size limit")
        else:
            validate_entry(kind, payload.entry)
        resource_target(kind, payload.team)
    if payload.operation == "dispatch" and (payload.group is not None or payload.mode not in MODES):
        raise HTTPException(status_code=422, detail="Choose an allowed workflow mode")
    if payload.operation != "dispatch" and (
        payload.workspace is not None or payload.github_refresh or payload.disable_refresh
    ):
        raise HTTPException(status_code=422, detail="Workflow options apply only to workflow runs")
    if payload.operation == "dispatch" and (payload.mode == "CUSTOM") != (
        payload.workspace in WORKSPACES
    ):
        raise HTTPException(
            status_code=422,
            detail="Custom mode needs one allowed workspace folder; other modes take none",
        )
    store = request.app.state.operations
    path = request_path(user, payload.request_id)
    fingerprint = hashlib.sha256(payload.model_dump_json().encode()).hexdigest()
    with user_github(request, user) as github:
        actor = github.authorize(payload.team)
        if kind and payload.action != "delete":
            unknown = sorted(
                set().union(*(entry_teams(item) for item in (payload.entries or [payload.entry])))
                - set(catalog_teams(github))
            )
            if unknown:
                raise HTTPException(
                    status_code=422,
                    detail="Teams must be team folders in CODEOWNERS: " + ", ".join(unknown),
                )
        existing = store.read(path)
        if existing:
            if existing["fingerprint"] != fingerprint:
                raise HTTPException(
                    status_code=409, detail="Request ID was already used for different inputs"
                )
            return existing
        record = {
            "id": payload.request_id.hex,
            "repository": REPOSITORY,
            "ab_oid": user["oid"],
            "github_actor": actor,
            "team": payload.team,
            "operation": payload.operation,
            "mode": payload.mode,
            "group": payload.group.model_dump() if payload.group else None,
            **(
                {
                    "workspace": payload.workspace,
                    "github_refresh": payload.github_refresh,
                    "disable_refresh": payload.disable_refresh,
                }
                if payload.operation == "dispatch"
                else {}
            ),
            **(
                {
                    "action": payload.action,
                    "resource_type": kind["id"],
                    "resource_title": kind["title"],
                    "resource_name": f"{len(payload.entries)} entries"
                    if payload.entries
                    else resource_name(kind, payload.entry),
                    "target_path": resource_target(kind, payload.team),
                    "entry_path": kind["entry_path"],
                    "entry": payload.entry,
                    **({"entries": payload.entries} if payload.entries else {}),
                }
                if kind
                else {}
            ),
            "created_at": timestamp(),
            "status": "submitting",
            "fingerprint": fingerprint,
        }
        if not store.write(path, record, create=True):
            raise HTTPException(
                status_code=409, detail="Request is already being submitted; refresh it"
            )
        try:
            if payload.operation == "create-group":
                name_hash = hashlib.sha256(payload.group.group_name.casefold().encode()).hexdigest()
                if not store.write(
                    f"claims/groups/{payload.team}/{name_hash}.json",
                    {"request_id": record["id"], "created_at": record["created_at"]},
                    create=True,
                ):
                    raise HTTPException(
                        status_code=409,
                        detail="Another group request exists. Check GitHub or contact CloudOps.",
                    )
                create_group(github, payload, record, store, path)
            elif kind:
                key = kind["identity_key"]
                if key and payload.action == "create":
                    for item in payload.entries or [payload.entry]:
                        name_hash = hashlib.sha256(
                            identity_value(kind["id"], item[key]).encode()
                        ).hexdigest()
                        claim = (
                            f"claims/groups/{payload.team}/{name_hash}.json"
                            if kind["id"] == "catalog-groups"
                            else f"claims/{kind['id']}/{payload.team}/{name_hash}.json"
                        )
                        if not store.write(
                            claim,
                            {"request_id": record["id"], "created_at": record["created_at"]},
                            create=True,
                        ):
                            raise HTTPException(
                                status_code=409,
                                detail="Another request for this entry exists. "
                                "Check GitHub or contact CloudOps.",
                            )
                create_resource(github, payload, record, store, path)
            else:
                dispatch(github, payload, record)
            store.write(path, record)
        except (CatalogError, HTTPException, ValueError, KeyError):
            record["status"] = "needs_attention"
            store.write(path, record)
            raise
    return record


def resolved_status(record: dict) -> dict:
    # Requests resolved by an admin before the status was set on resolve.
    if record.get("status") == "escalated" and (record.get("escalation") or {}).get("resolved"):
        record["status"] = "resolved"
    return record


@router.get("/api/catalog/requests")
def requests_list(request: Request, cursor: str | None = None, user: dict = Depends(catalog_user)):
    if cursor is not None and len(cursor) > 4096:
        raise HTTPException(status_code=422, detail="Invalid request-list cursor")
    result = request.app.state.operations.page(f"requests/{owner_key(user['oid'])}/", cursor)
    for item in result["items"]:
        resolved_status(item)
    result["items"].sort(key=lambda item: item["created_at"], reverse=True)
    return result


@router.get("/api/catalog/requests/{identifier}")
def request_detail(identifier: UUID, request: Request, user: dict = Depends(catalog_user)):
    # Read-only: GitHub status refresh and recovery happen only through the POST sync route.
    record = request.app.state.operations.read(request_path(user, identifier))
    if record is None:
        raise HTTPException(status_code=404, detail="Request not found")
    return resolved_status(record)
