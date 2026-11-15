"""Admin-only Terraform import for Service Catalog team workspaces.

The Hub dispatches the reviewed import.yml in the catalog repository with the admin's own
GitHub identity. Like caller.yml, every level plans in autorun and applies in the cloudops
environment, so a CloudOps reviewer approves each apply. Import IDs never enter the catalog:
the job writes temporary import blocks at run time and plans only the imported addresses.
"""

import base64
import json
import re
from uuid import uuid4

import httpx
from azure.identity import DefaultAzureCredential
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from cloudops_hub import service_catalog as catalog
from cloudops_hub.catalog_review import audited
from cloudops_hub.catalog_tracking import catalog_admin
from cloudops_hub.config import get_settings
from cloudops_hub.operations_store import timestamp
from cloudops_hub.terraform_state import FINISHED, dispatch, github_login, refresh

router = APIRouter(tags=["terraform-import"])

WORKFLOW = "import.yml"
RECORD = "terraform-import/{run_id}.json"
LEVELS = 4
MAX_IMPORTS = 200
MAX_WORKSPACES = 30
MAX_MATCH = 20
MAX_LOGS = 10
# Copy of workspace_config children in the code repo's validate.locals.tf.
CHILDREN = {
    "onpremaccess": ["devops"],
    "backend": ["rbac"],
    "githubaccess": ["githubenvs"],
    "branchprotection": ["githubenvs"],
    "githubrepos": [
        "githuboptional",
        "branchprotection",
        "githubaccess",
        "githubspns",
        "githubapp",
        "devops",
    ],
    "githubenvs": [
        "ghtenantid",
        "ghsubscriptionid",
        "ghcustomsecrets",
        "ghclientsecret",
        "ghclientid",
    ],
    "githubspns": [
        "members",
        "rbac",
        "ghtenantid",
        "ghsubscriptionid",
        "ghcustomsecrets",
        "ghclientsecret",
        "ghclientid",
        "keyvault",
        "cloudops",
    ],
    "groups": ["approles", "rbac", "members", "githubteam"],
    "idns": ["rbac", "keyvault"],
    "webapp": ["approles"],
    "approles": ["keyvault"],
    "certs": ["rbac", "keyvault"],
}
UNATTENDED = {
    "keyvault",
    "ghtenantid",
    "ghsubscriptionid",
    "ghcustomsecrets",
    "ghclientsecret",
    "ghclientid",
    "devops",
    "cloudops",
}
# sftp needs the runner IP allowed on its storage account; import it through the normal run.
IMPORTABLE = tuple(
    sorted(
        {*catalog.WORKSPACES, *CHILDREN, *(c for kids in CHILDREN.values() for c in kids)}
        - {"sftp"}
    )
)
GITHUB_WORKSPACES = frozenset(
    {
        "githubrepos",
        "githubaccess",
        "branchprotection",
        "githubenvs",
        "githuboptional",
        "githubapp",
        "githubteam",
        "ghtenantid",
        "ghsubscriptionid",
        "ghcustomsecrets",
        "ghclientsecret",
        "ghclientid",
    }
)
# Catalog files each workspace reads (code repo), so each page only offers matching types.
WORKSPACE_TYPES = {
    **{ws: ["catalog-repositories"] for ws in GITHUB_WORKSPACES},
    "githubenvs": ["catalog-repositories", "catalog-rulesets"],
    "githubteam": ["catalog-groups"],
    "groups": ["catalog-groups"],
    "members": ["catalog-group-members", "catalog-groups"],
    "webapp": ["catalog-applications"],
    "approles": ["catalog-applications"],
    "cloudops": ["catalog-applications"],
    "githubspns": ["catalog-applications", "catalog-tags"],
    "idns": ["catalog-identities"],
    "certs": ["catalog-certificates"],
    "onpremaccess": ["catalog-onprem"],
    "nsp": ["catalog-nsp"],
    "gsa": ["catalog-gsa"],
    "pim": ["catalog-pim"],
    "publicnetworkaccess": ["catalog-public-network-access"],
    "devops": ["catalog-repositories"],
    "keyvault": ["catalog-repositories"],
}


def provider_of(workspace: str) -> str:
    return "github" if workspace in GITHUB_WORKSPACES else "azure"


def address_provider(address: str) -> str:
    resource = re.sub(r"^(module\.[^.]+\.)*", "", address)
    return "github" if resource.startswith("github_") else "azure"


KEY = r'(\[(\d+|"[^"\\$%\n\r]*")\])?'
ADDRESS = re.compile(
    rf"^(module\.[A-Za-z0-9_-]+{KEY}\.)*[a-z][a-z0-9_]*\.[A-Za-z_][A-Za-z0-9_-]*{KEY}$"
)
MATCH = re.compile(r"^[A-Za-z0-9._@|:/ -]{3,200}$")
SEARCH = re.compile(r"^[A-Za-z0-9._ -]{2,80}$")
STAMP = re.compile(r"^\S+Z ")
ANSI = re.compile(r"\x1b\[[0-9;]*m")
EXISTS = re.compile(
    r'a resource with the ID "(?P<id>[^"\n]{1,2048})" already exists'
    r"[\s\S]{0,4000}?\bwith (?P<address>[^\s,]{1,1024}),"
)
# caller.yml names apply jobs "level1-tf-apply--GITHUBSPNS" and "automations-DEVOPS".
APPLY_JOB = re.compile(r"(?i)(?:tf-apply-+|automations-)([a-z]{2,40})")
IMPORT_JOB = re.compile(r"import-(plan|review|apply)-L([1-4])(?:-([a-z]{2,40}))?")
RUN_URL = re.compile(
    r"^https://github\.com/"
    + re.escape(catalog.REPOSITORY)
    + r"/actions/runs/(\d{1,20})(?:/job/(\d{1,20}))?(?:/attempts/\d+)?/?(?:[?#].*)?$"
)
RUN_TEAM = re.compile(r"\bto ([a-z0-9]+(?:-[a-z0-9]+)*) environment\b")
MARKER = re.compile(r"^HUB-IMPORT-(ITEM|PLAN|APPLIED) (\{.*\})$")
GRAPH_KINDS = {"group": "groups", "application": "applications"}


def clean(log: str) -> str:
    lines = []
    for raw in log.splitlines():
        line = ANSI.sub("", STAMP.sub("", raw.rstrip("\r")))
        lines.append(line.lstrip("│╷╵ ").rstrip())
    return "\n".join(lines)


def valid_id(value: str) -> bool:
    return (
        0 < len(value) <= 2048
        and not any(ord(char) < 32 for char in value)
        and "${" not in value
        and "%{" not in value
    )


def parse_errors(text: str, workspace: str = "") -> list[dict]:
    """Terraform 'already exists' errors as import items, de-duplicated by address."""
    items: dict[str, dict] = {}
    for match in EXISTS.finditer(clean(text)):
        address, value = match.group("address"), match.group("id")
        if ADDRESS.fullmatch(address) and valid_id(value):
            items[address] = {
                "workspace": workspace,
                "to": address,
                "id": value,
                "provider": address_provider(address),
            }
    return list(items.values())


def descendants(workspace: str) -> set[str]:
    found: set[str] = set()
    queue = [workspace]
    while queue:
        for child in CHILDREN.get(queue.pop(), []):
            if child not in found and child != workspace:
                found.add(child)
                queue.append(child)
    return found


def build_levels(workspaces: set[str]) -> list[list[str]]:
    """Order workspaces like called.yml: parents first, at most 3 levels, unattended last."""
    reach = {workspace: descendants(workspace) for workspace in workspaces}
    depth: dict[str, int] = {}

    def level(workspace: str, path: frozenset = frozenset()) -> int:
        if workspace not in depth:
            parents = [
                other
                for other in workspaces
                if other != workspace and workspace in reach[other] and other not in path
            ]
            depth[workspace] = 1 + max(
                (level(parent, path | {workspace}) for parent in parents), default=0
            )
        return depth[workspace]

    levels: list[list[str]] = [[] for _ in range(LEVELS)]
    for workspace in sorted(workspaces):
        index = LEVELS if workspace in UNATTENDED else min(level(workspace), LEVELS - 1)
        levels[index - 1].append(workspace)
    return levels


def check_workspace(workspace: str) -> None:
    if workspace not in IMPORTABLE:
        raise HTTPException(status_code=422, detail=f"Workspace {workspace!r} cannot be imported")


class ImportItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    workspace: str = Field(min_length=2, max_length=40)
    to: str = Field(min_length=3, max_length=1024)
    id: str = Field(default="", max_length=2048)
    provider: str = Field(default="", pattern=r"^(github|azure)?$")


class CatalogChange(BaseModel):
    model_config = ConfigDict(extra="forbid")
    resource_type: str = Field(min_length=1, max_length=64)
    action: str = Field(default="auto", pattern=r"^(auto|create|update)$")
    entry: dict


class ImportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    team: str = Field(min_length=1, max_length=63)
    source: str = Field(pattern=r"^(discover|error_log)$")
    workspace: str = Field(default="", max_length=40)
    include_children: bool = True
    items: list[ImportItem] = Field(default_factory=list, max_length=MAX_IMPORTS)
    match: list[str] = Field(default_factory=list, max_length=MAX_MATCH)
    catalog_change: CatalogChange | None = None
    source_run_id: int | None = Field(default=None, gt=0)
    reason: str = Field(min_length=10, max_length=500)
    provider: str = Field(default="", pattern=r"^(github|azure)?$")
    confirmed: bool = True


class PreviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    team: str = Field(min_length=1, max_length=63)
    catalog_change: CatalogChange


class ParseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=200_000)


def build_plan(payload: ImportRequest) -> dict:
    workspaces: set[str] = set()
    if payload.workspace:
        check_workspace(payload.workspace)
        workspaces.add(payload.workspace)
        if payload.include_children:
            workspaces |= descendants(payload.workspace) & set(IMPORTABLE)
    imports, seen = [], set()
    for item in payload.items:
        check_workspace(item.workspace)
        if not ADDRESS.fullmatch(item.to):
            raise HTTPException(status_code=422, detail=f"Invalid Terraform address: {item.to}")
        if item.id and not valid_id(item.id):
            raise HTTPException(status_code=422, detail=f"Invalid import ID for {item.to}")
        if (item.workspace, item.to) in seen:
            raise HTTPException(status_code=422, detail=f"Repeated address: {item.to}")
        seen.add((item.workspace, item.to))
        workspaces.add(item.workspace)
        imports.append({"workspace": item.workspace, "to": item.to, "id": item.id})
    match = list(dict.fromkeys(value.strip() for value in payload.match if value.strip()))
    if any(not MATCH.fullmatch(value) for value in match):
        raise HTTPException(
            status_code=422, detail="Names to match must be 3-200 letters, digits or ._@|:/ -"
        )
    if match and not payload.workspace:
        raise HTTPException(status_code=422, detail="Choose the workspace to discover in")
    if not imports and not match:
        raise HTTPException(status_code=422, detail="Add resources to import or names to match")
    if len(workspaces) > MAX_WORKSPACES:
        raise HTTPException(status_code=422, detail="Too many workspaces for one import")
    plan = {"levels": build_levels(workspaces), "imports": imports, "match": match}
    if len(json.dumps(plan, separators=(",", ":"))) > 60_000:
        raise HTTPException(status_code=422, detail="The import is too large for one run")
    return plan


def keep_unknown(old, new):
    """New values win; fields the form does not know (and named list items) are kept."""
    if isinstance(old, dict) and isinstance(new, dict):
        merged = dict(old)
        for key, value in new.items():
            merged[key] = keep_unknown(old.get(key), value)
        return merged
    if (
        isinstance(old, list)
        and isinstance(new, list)
        and all(isinstance(item, dict) and "name" in item for item in [*old, *new])
    ):
        before = {str(item["name"]).casefold(): item for item in old}
        return [keep_unknown(before.get(str(item["name"]).casefold()), item) for item in new]
    return new


def existing_entry(kind: dict, document: dict, entry: dict):
    node = document
    for part in kind["entry_path"]:
        node = node.get(part) if isinstance(node, dict) else None
    if kind["kind"] == "replace":
        return node if isinstance(node, dict) else None
    key = kind.get("identity_key")
    if not key or not isinstance(node, list) or key not in entry:
        return None
    wanted = catalog.identity_value(kind["id"], entry[key])
    return next(
        (
            item
            for item in node
            if isinstance(item, dict)
            and catalog.identity_value(kind["id"], item.get(key, "")) == wanted
        ),
        None,
    )


def catalog_document(github: catalog.GitHub, team: str, change: CatalogChange, ref: str) -> dict:
    """The team file after the change: a new entry is appended, an existing one is updated."""
    kind = catalog.resource_types().get(change.resource_type)
    if kind is None:
        raise HTTPException(status_code=422, detail="Choose a catalog resource type")
    filename = catalog.resource_target(kind, team)
    catalog.validate_entry(kind, change.entry)
    try:
        source = github.call("GET", f"{catalog.BASE}/contents/{filename}", params={"ref": ref})
    except catalog.GitHubNotFound:
        source = None
    raw = b"{}"
    if source is not None:
        if source.get("type") != "file" or source.get("encoding") != "base64":
            raise catalog.CatalogError("Existing catalog target is not a supported file")
        if source.get("size", 1_000_001) > 1_000_000:
            raise catalog.CatalogError("Catalog file exceeds the reviewed size limit")
        raw = base64.b64decode(source["content"], validate=False)
    current = catalog.catalog_document(raw)
    existing = existing_entry(kind, current, change.entry)
    update = change.action == "update" or (change.action == "auto" and existing is not None)
    if update and source is None:
        raise HTTPException(status_code=404, detail="The catalog file does not exist yet")
    entry = keep_unknown(existing, change.entry) if existing is not None else change.entry
    unchanged = update and existing == entry
    document = current if unchanged else catalog.resource_update(raw, kind, entry, update)
    return {
        "kind": kind,
        "file": filename,
        "source": source,
        "document": document,
        "update": update,
        "unchanged": unchanged,
    }


def catalog_branch(github: catalog.GitHub, team: str, change: CatalogChange, slug: str) -> dict:
    """Commit the catalog entry to an import branch and open a PR; merged after the import.

    None when the team file already has exactly this entry: the import then plans from main.
    """
    base_sha = github.call("GET", f"{catalog.BASE}/git/ref/heads/main")["object"]["sha"]
    result = catalog_document(github, team, change, base_sha)
    if result["unchanged"]:
        return None
    kind, filename, source = result["kind"], result["file"], result["source"]
    document, update = result["document"], result["update"]
    name = catalog.resource_name(kind, change.entry)
    branch = f"cloudops-hub/import-{slug}"
    github.call(
        "POST", f"{catalog.BASE}/git/refs", body={"ref": f"refs/heads/{branch}", "sha": base_sha}
    )
    github.call(
        "PUT",
        f"{catalog.BASE}/contents/{filename}",
        body={
            "message": f"{'Update' if update else 'Add'} {kind['title']}: {name} for {team} "
            "(import)",
            "content": base64.b64encode((json.dumps(document, indent=2) + "\n").encode()).decode(),
            **({"sha": source["sha"]} if source is not None else {}),
            "branch": branch,
        },
    )
    pr = github.call(
        "POST",
        f"{catalog.BASE}/pulls",
        body={
            "title": f"Import {kind['title']}: {name} for {team}",
            "head": branch,
            "base": "main",
            "body": f"CloudOps Hub import. `{filename}` gets the catalog entry for a resource "
            "that already exists. import.yml imports it into Terraform state from this branch "
            "first; merge this PR only after the import succeeds, so the normal run finds "
            "nothing to create. Import IDs are not stored in the catalog.",
        },
    )
    key = kind.get("identity_key")
    return {
        "branch": branch,
        "file": filename,
        "pr_number": pr["number"],
        "pr_url": pr["html_url"],
        "identity": str(change.entry.get(key, "")) if key else "",
    }


def parse_log(log: str, record: dict) -> None:
    for raw in clean(log).splitlines():
        match = MARKER.match(raw.strip())
        if not match:
            continue
        try:
            data = json.loads(match.group(2))
        except ValueError:
            continue
        workspace = data.get("workspace") if isinstance(data, dict) else None
        if not isinstance(workspace, str) or not re.fullmatch(r"[a-z]{2,40}", workspace):
            continue
        if match.group(1) == "ITEM":
            item = {
                key: str(data.get(key, ""))[:2048]
                for key in ("workspace", "address", "status", "id", "reason")
            }
            items = [
                existing
                for existing in record.setdefault("items", [])
                if (existing["workspace"], existing["address"])
                != (item["workspace"], item["address"])
            ]
            record["items"] = [*items, item][:1000]
        elif match.group(1) == "PLAN":
            record.setdefault("plans", {})[workspace] = {
                key: int(data.get(key) or 0) for key in ("import", "add", "change", "destroy")
            }
        else:
            applied = record.setdefault("applied", [])
            if workspace not in applied:
                applied.append(workspace)


def sync_jobs(github: catalog.GitHub, record: dict) -> None:
    attempt = int(record.get("run_attempt") or 1)
    jobs = github.call(
        "GET",
        f"{catalog.BASE}/actions/runs/{record['run_id']}/attempts/{attempt}/jobs",
        params={"per_page": 100},
    ).get("jobs", [])
    parsed = set(record.get("parsed_jobs", []))
    steps = []
    for job in jobs:
        match = IMPORT_JOB.search(str(job.get("name", "")))
        if not match:
            continue
        phase, level, workspace = match.group(1), int(match.group(2)), match.group(3) or ""
        steps.append(
            {
                "phase": phase,
                "level": level,
                "workspace": workspace,
                "status": job.get("status"),
                "conclusion": job.get("conclusion"),
                "url": job.get("html_url"),
            }
        )
        job_id = job.get("id")
        if (
            phase in {"plan", "apply"}
            and job.get("status") == "completed"
            and job.get("conclusion") in {"success", "failure"}
            and isinstance(job_id, int)
            and job_id not in parsed
        ):
            try:
                parse_log(github.job_log(job_id), record)
            except catalog.CatalogError:
                continue
            parsed.add(job_id)
    record["jobs"] = sorted(
        steps, key=lambda step: (step["level"], step["phase"], step["workspace"])
    )
    record["parsed_jobs"] = sorted(parsed)


def load(request: Request, run_id: int) -> dict:
    record = request.app.state.operations.read(RECORD.format(run_id=run_id))
    if record is None:
        raise HTTPException(status_code=404, detail="Terraform import run not found")
    return record


@router.get("/api/admin/terraform-import/options")
def options(request: Request, user: dict = Depends(catalog_admin)):
    with catalog.user_github(request, user) as github:
        teams = catalog.catalog_teams(github)
    types = [
        {
            "id": kind["id"],
            "title": kind["title"],
            "target_path": kind["target_path"],
            "identity_key": kind["identity_key"],
            "kind": kind["kind"],
        }
        for kind in catalog.resource_types().values()
    ]
    return {
        "teams": teams,
        "workspaces": list(IMPORTABLE),
        "providers": {
            provider: [ws for ws in IMPORTABLE if provider_of(ws) == provider]
            for provider in ("github", "azure")
        },
        "workspace_types": {ws: WORKSPACE_TYPES.get(ws, []) for ws in IMPORTABLE},
        "children": {workspace: sorted(descendants(workspace)) for workspace in IMPORTABLE},
        "resource_types": types,
        "graph_search": get_settings().graph_lookups_enabled,
    }


@router.post("/api/admin/terraform-import/parse")
def parse(payload: ParseRequest, request: Request, user: dict = Depends(catalog_admin)):
    """Find import items in pasted error text or in the failed jobs of a caller.yml run."""
    text = payload.text.strip()
    url = RUN_URL.match(text)
    if not url:
        return {"team": "", "run_id": None, "items": parse_errors(text)}
    run_id, only_job = int(url.group(1)), int(url.group(2) or 0)
    with catalog.user_github(request, user) as github:
        run = github.call("GET", f"{catalog.BASE}/actions/runs/{run_id}")
        if run.get("path") != f".github/workflows/{catalog.WORKFLOW}":
            raise HTTPException(status_code=422, detail="Paste a Service-Catalog caller.yml run")
        team_match = RUN_TEAM.search(str(run.get("display_title", "")))
        team = team_match.group(1) if team_match else ""
        if team and team not in catalog.catalog_teams(github):
            team = ""
        attempt = int(run.get("run_attempt") or 1)
        jobs = github.call(
            "GET",
            f"{catalog.BASE}/actions/runs/{run_id}/attempts/{attempt}/jobs",
            params={"per_page": 100},
        ).get("jobs", [])
        failed = [
            job
            for job in jobs
            if job.get("conclusion") == "failure"
            and isinstance(job.get("id"), int)
            and (not only_job or job["id"] == only_job)
            and APPLY_JOB.search(str(job.get("name", "")))
        ][:MAX_LOGS]
        items: dict[tuple[str, str], dict] = {}
        for job in failed:
            workspace = APPLY_JOB.search(job["name"]).group(1).lower()
            for item in parse_errors(github.job_log(job["id"]), workspace):
                items[(workspace, item["to"])] = {**item, "job": job["name"]}
    return {"team": team, "run_id": run_id, "items": list(items.values())}


def identity_values(node, key: str, found: set[str]) -> set[str]:
    if isinstance(node, dict):
        for name, value in node.items():
            if name == key and isinstance(value, str):
                found.add(value.casefold())
            else:
                identity_values(value, key, found)
    elif isinstance(node, list):
        for value in node:
            identity_values(value, key, found)
    return found


def catalog_names(github: catalog.GitHub, team: str, type_id: str) -> tuple[str, set[str]] | None:
    """Identity values already in the team's catalog file for this resource type."""
    kind = catalog.resource_types().get(type_id)
    if kind is None or not kind.get("identity_key"):
        return None
    filename = catalog.resource_target(kind, team)
    try:
        source = github.call("GET", f"{catalog.BASE}/contents/{filename}", params={"ref": "main"})
    except catalog.GitHubNotFound:
        return filename, set()
    if source.get("encoding") != "base64" or source.get("size", 1_000_001) > 1_000_000:
        return None
    try:
        document = json.loads(base64.b64decode(source["content"], validate=False))
    except ValueError:
        return None
    return filename, identity_values(document, kind["identity_key"], set())


def graph_search(kind: str, q: str) -> list[dict]:
    settings = get_settings()
    if not settings.graph_lookups_enabled:
        raise HTTPException(status_code=503, detail="Microsoft Graph lookups are disabled")
    credential = DefaultAzureCredential()
    try:
        token = credential.get_token("https://graph.microsoft.com/.default")
        with httpx.Client(timeout=10) as client:
            response = client.get(
                f"https://graph.microsoft.com/v1.0/{GRAPH_KINDS[kind]}",
                params={
                    "$filter": f"startswith(displayName,'{q}')",
                    "$select": "id,displayName" + (",appId" if kind == "application" else ""),
                    "$top": "20",
                },
                headers={"Authorization": f"Bearer {token.token}"},
            )
    except Exception:
        raise HTTPException(status_code=503, detail="Microsoft Graph is unavailable") from None
    finally:
        credential.close()
    if response.status_code >= 400:
        raise HTTPException(status_code=502, detail="Microsoft Graph search failed")
    return [
        {
            "id": str(value.get("id", "")),
            "name": str(value.get("displayName", "")),
            "detail": str(value.get("appId") or value.get("id") or ""),
        }
        for value in response.json().get("value", [])[:20]
        if isinstance(value, dict)
    ]


@router.get("/api/admin/terraform-import/discover")
def discover(
    kind: str,
    q: str,
    request: Request,
    team: str = "",
    resource_type: str = "",
    user: dict = Depends(catalog_admin),
):
    """Existing resources by name: Entra with the Hub identity, GitHub with the admin's AB login."""
    if kind not in {*GRAPH_KINDS, "repository"} or not SEARCH.fullmatch(q):
        raise HTTPException(status_code=422, detail="Search by name (2-80 letters or digits)")
    with catalog.user_github(request, user) as github:
        if kind == "repository":
            found = github.call(
                "GET",
                "/search/repositories",
                params={"q": f"{q} in:name org:almbrand", "per_page": 20},
            )
            results = [
                {
                    "id": str(repo.get("id", "")),
                    "name": str(repo.get("name", "")),
                    "detail": str(repo.get("visibility", "")),
                }
                for repo in (found or {}).get("items", [])[:20]
                if isinstance(repo, dict)
            ]
        else:
            results = graph_search(kind, q)
        known = None
        if team and resource_type and catalog.SHORT_TEAM.fullmatch(team):
            known = catalog_names(github, team, resource_type)
    for item in results:
        item["in_catalog"] = None if known is None else item["name"].casefold() in known[1]
        item["file"] = known[0] if known else ""
    return results


REPO_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,99}$")
CLOUD_FOUNDATION_ID = 5753831


def check_team(github: catalog.GitHub, team: str) -> None:
    if not catalog.SHORT_TEAM.fullmatch(team) or team not in catalog.catalog_teams(github):
        raise HTTPException(status_code=422, detail="Choose a team from CODEOWNERS")


@router.post("/api/admin/terraform-import/catalog-preview")
def catalog_preview(payload: PreviewRequest, request: Request, user: dict = Depends(catalog_admin)):
    """The complete team file the import PR will contain; nothing is written."""
    with catalog.user_github(request, user) as github:
        check_team(github, payload.team)
        result = catalog_document(github, payload.team, payload.catalog_change, "main")
    return {
        "file": result["file"],
        "file_exists": result["source"] is not None,
        "action": "update" if result["update"] else "create",
        "unchanged": result["unchanged"],
        "content": json.dumps(result["document"], indent=2) + "\n",
    }


def graph_get(client: httpx.Client, token: str, path: str, params: dict) -> list[dict]:
    response = client.get(
        f"https://graph.microsoft.com/v1.0/{path}",
        params=params,
        headers={"Authorization": f"Bearer {token}", "ConsistencyLevel": "eventual"},
    )
    if response.status_code >= 400:
        return []
    return [item for item in response.json().get("value", []) if isinstance(item, dict)]


def spn_subscriptions(team: str, repo: str, environments: list[str]) -> dict[str, str]:
    """Subscription per environment from its GitHub SPN's <subscription>-Contributor groups."""
    if not environments or not get_settings().graph_lookups_enabled:
        return {}
    found: dict[str, str] = {}
    credential = DefaultAzureCredential()
    try:
        token = credential.get_token("https://graph.microsoft.com/.default").token
        with httpx.Client(timeout=10) as client:
            for name in environments[:20]:
                spn = f"spn-level03-ab-{team}-{repo}-{name}"
                principals = graph_get(
                    client,
                    token,
                    "servicePrincipals",
                    {"$filter": f"displayName eq '{spn}'", "$select": "id", "$top": "1"},
                )
                if not principals:
                    continue
                groups = graph_get(
                    client,
                    token,
                    f"servicePrincipals/{principals[0]['id']}/transitiveMemberOf/microsoft.graph.group",
                    {"$select": "displayName", "$top": "100"},
                )
                for group in groups:
                    title = str(group.get("displayName", ""))
                    for suffix in ("-Contributor-spns", "-Contributor"):
                        subscription = title.removesuffix(suffix)
                        if subscription != title and re.fullmatch(
                            r"ab-[a-z0-9]+(?:-[a-z0-9]+)*", subscription
                        ):
                            found[name] = subscription
                            break
                    if name in found:
                        break
    except Exception:
        return found
    finally:
        credential.close()
    return found


def live_repository(github: catalog.GitHub, team: str, name: str, notes: list[str]) -> dict:
    """The catalog fields GitHub can show for an existing repository."""
    base = f"/repos/almbrand/{name}"
    try:
        repo = github.call("GET", base)
    except catalog.GitHubNotFound:
        raise HTTPException(status_code=404, detail=f"Repository {name} was not found") from None
    entry: dict = {"github_repo_name": str(repo.get("name") or name)}
    description = str(repo.get("description") or "").strip()
    if description and description != name:
        entry["description"] = description[:300]
    template = (repo.get("template_repository") or {}).get("name")
    if template:
        entry["template"] = str(template)
    own = {f"lz-ab-{team}-maintain", f"lz-ab-{team}-admin", "cloudfoundation"}
    environments: list[dict] = []
    gates: list[dict] = []
    listed = github.call("GET", f"{base}/environments", params={"per_page": 100}) or {}
    for env in (listed.get("environments") or [])[:50]:
        env_name = str(env.get("name", ""))
        if not REPO_NAME.fullmatch(env_name):
            continue
        users: list[str] = []
        groups: list[str] = []
        admin_team = False
        prevent = False
        for rule in env.get("protection_rules") or []:
            if rule.get("type") != "required_reviewers":
                continue
            prevent = bool(rule.get("prevent_self_review"))
            for reviewer in rule.get("reviewers") or []:
                who = reviewer.get("reviewer") or {}
                if reviewer.get("type") == "User":
                    login = str(who.get("login", ""))
                    if login.endswith("_gheab"):
                        users.append(login.removesuffix("_gheab"))
                    elif login:
                        notes.append(f"{env_name}: reviewer {login} is not _gheab; skipped.")
                elif reviewer.get("type") == "Team" and who.get("name"):
                    groups.append(str(who["name"]))
        try:
            secrets = github.call("GET", f"{base}/environments/{env_name}/secrets")
            names = {str(item.get("name", "")) for item in (secrets or {}).get("secrets", [])}
        except catalog.CatalogError:
            names = None
        if names is not None and "ARM_SUBSCRIPTION_ID" not in names:
            gate: dict = {"name": env_name}
            admin_team = any(group.casefold() == f"lz-ab-{team}-admin" for group in groups)
            extra = [group for group in groups if group.casefold() not in own]
            if admin_team:
                gate["include_admin_team"] = "true"
            if users:
                gate["reviewers_users"] = users
            if extra:
                gate["reviewers_groups"] = extra
            gates.append(gate)
            continue
        item: dict = {"name": env_name}
        if users:
            item["reviewers_users"] = users
        if groups:
            item["reviewers_groups"] = groups
        if prevent:
            item["prevent_self_review"] = True
        policy = env.get("deployment_branch_policy") or {}
        if policy.get("protected_branches"):
            item["deployment_branches_protected"] = True
        environments.append(item)
    subscriptions = spn_subscriptions(team, name, [item["name"] for item in environments])
    for item in environments:
        if item["name"] in subscriptions:
            item["subscription"] = subscriptions[item["name"]]
        else:
            notes.append(f"Subscription for {item['name']} was not found; fill it in.")
    entry["Environment"] = environments
    if gates:
        entry["manual_gate_githubenv"] = gates
    roles: dict[str, list[str]] = {}
    try:
        teams = github.call("GET", f"{base}/teams", params={"per_page": 100}) or []
    except catalog.CatalogError:
        teams = []
        notes.append("Team access could not be read.")
    for access in teams if isinstance(teams, list) else []:
        team_name = str(access.get("name", ""))
        if access.get("id") == CLOUD_FOUNDATION_ID or team_name.casefold() in own:
            continue
        short = re.fullmatch(r"lz-ab-(.+)-maintain", team_name.casefold())
        role = str(access.get("permission", ""))
        if role in {"pull", "triage", "push", "maintain", "admin"} and team_name:
            roles.setdefault(role, []).append(short.group(1) if short else team_name)
    if roles:
        entry["access_to_other_teams"] = [
            {"role": role, "teams": sorted(set(names))} for role, names in roles.items()
        ]
    return entry


def fill_missing(existing, live):
    """Existing catalog values win; live values only fill gaps (environments by name)."""
    if isinstance(existing, dict) and isinstance(live, dict):
        merged = dict(existing)
        for key, value in live.items():
            merged[key] = fill_missing(existing.get(key), value) if key in existing else value
        return merged
    if (
        isinstance(existing, list)
        and isinstance(live, list)
        and all(isinstance(item, dict) and "name" in item for item in [*existing, *live])
    ):
        known = {str(item["name"]).casefold(): item for item in existing}
        merged = [
            fill_missing(
                item, next((other for other in live if str(other["name"]).casefold() == key), {})
            )
            for key, item in known.items()
        ]
        return merged + [item for item in live if str(item["name"]).casefold() not in known]
    return existing if existing not in (None, "", [], {}) else live


@router.get("/api/admin/terraform-import/github-repo")
def github_repo(name: str, team: str, request: Request, user: dict = Depends(catalog_admin)):
    """A catalog entry for an existing repository: its catalog block plus what GitHub shows."""
    if not REPO_NAME.fullmatch(name):
        raise HTTPException(status_code=422, detail="Invalid repository name")
    kind = catalog.resource_types()["catalog-repositories"]
    notes: list[str] = []
    with catalog.user_github(request, user) as github:
        check_team(github, team)
        entries = catalog.read_entries(github, kind, team)
        wanted = catalog.identity_value(kind["id"], name)
        current = next(
            (
                item
                for item in entries["items"]
                if catalog.identity_value(kind["id"], item.get("github_repo_name", "")) == wanted
            ),
            None,
        )
        live = live_repository(github, team, name, notes)
    return {
        "action": "update" if current else "create",
        "file": entries["target_path"],
        "file_exists": entries["exists"],
        "entry": fill_missing(current, live) if current else live,
        "notes": notes[:20],
    }


def record_provider(record: dict) -> str:
    if record.get("provider") in {"github", "azure"}:
        return record["provider"]
    first = next((ws for level in record.get("levels", []) for ws in level), "")
    return provider_of(first)


@router.get("/api/admin/terraform-import/runs")
def runs(request: Request, provider: str = "", _: dict = Depends(catalog_admin)):
    items = request.app.state.operations.list("terraform-import/", limit=200)
    if provider:
        items = [item for item in items if record_provider(item) == provider]
    return sorted(items, key=lambda item: item["created_at"], reverse=True)[:50]


@router.post("/api/admin/terraform-import/imports", status_code=202)
def start_import(payload: ImportRequest, request: Request, user: dict = Depends(catalog_admin)):
    if payload.catalog_change and payload.source != "discover":
        raise HTTPException(status_code=422, detail="Catalog changes belong to Discover")
    reason = " ".join(payload.reason.split())
    slug = uuid4().hex[:12]
    if payload.catalog_change and not payload.workspace:
        raise HTTPException(status_code=422, detail="Choose the workspace to import")
    plan = build_plan(payload)
    with catalog.user_github(request, user) as github:
        if not catalog.SHORT_TEAM.fullmatch(payload.team) or payload.team not in (
            catalog.catalog_teams(github)
        ):
            raise HTTPException(status_code=422, detail="Choose a team from CODEOWNERS")
        login = github_login(github)
        workspaces = [ws for level in plan["levels"] for ws in level]
        change = None
        with audited(
            request,
            user,
            login,
            "terraform_import",
            f"{payload.team}/{','.join(workspaces)}",
            assistant="terraform-import",
            source=payload.source,
            imports=len(plan["imports"]),
            match=plan["match"],
            reason=reason,
            catalog_change=payload.catalog_change.resource_type if payload.catalog_change else "",
        ):
            if payload.catalog_change:
                change = catalog_branch(github, payload.team, payload.catalog_change, slug)
                kind = catalog.resource_types()[payload.catalog_change.resource_type]
                key = kind.get("identity_key")
                identity = str(payload.catalog_change.entry.get(key, "")) if key else ""
                if identity and MATCH.fullmatch(identity) and identity not in payload.match:
                    payload.match = [*payload.match, identity]
                    plan = build_plan(payload)
            plan["request_id"] = slug
            run_id = dispatch(
                github,
                WORKFLOW,
                {
                    "environment": payload.team,
                    "PLAN": json.dumps(plan, separators=(",", ":")),
                    "CATALOG_REF": change["branch"] if change else "main",
                    "REASON": reason,
                },
            )
    record = {
        "kind": "import",
        "run_id": run_id,
        "run_url": f"https://github.com/{catalog.REPOSITORY}/actions/runs/{run_id}",
        "team": payload.team,
        "source": payload.source,
        "source_run_id": payload.source_run_id,
        "provider": payload.provider or record_provider({"levels": plan["levels"]}),
        "levels": plan["levels"],
        "requested": plan["imports"],
        "match": plan["match"],
        "catalog": change,
        "reason": reason,
        "requested_by": user.get("email") or user.get("name", ""),
        "github_login": login,
        "created_at": timestamp(),
        "status": "running",
    }
    request.app.state.operations.write(RECORD.format(run_id=run_id), record)
    return record


@router.post("/api/admin/terraform-import/imports/{run_id}/sync")
def sync_import(run_id: int, request: Request, user: dict = Depends(catalog_admin)):
    record = load(request, run_id)
    if record["status"] not in FINISHED or "jobs" not in record:
        with catalog.user_github(request, user) as github:
            refresh(github, record, WORKFLOW)
            sync_jobs(github, record)
        request.app.state.operations.write(RECORD.format(run_id=run_id), record)
    return record


@router.post("/api/admin/terraform-import/imports/{run_id}/cancel")
def cancel_import(run_id: int, request: Request, user: dict = Depends(catalog_admin)):
    """Cancel a running or waiting import in GitHub; applies already finished are not undone."""
    record = load(request, run_id)
    with catalog.user_github(request, user) as github:
        refresh(github, record, WORKFLOW)
        if record["status"] not in FINISHED:
            login = github_login(github)
            with audited(
                request,
                user,
                login,
                "terraform_import_cancel",
                f"{record['team']}/{run_id}",
                assistant="terraform-import",
            ):
                github.call("POST", f"{catalog.BASE}/actions/runs/{run_id}/cancel")
            record["cancel_requested_by"] = user.get("email") or user.get("name", "")
            record["status"] = "cancelling"
    request.app.state.operations.write(RECORD.format(run_id=run_id), record)
    return record
