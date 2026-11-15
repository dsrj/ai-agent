"""Admin-only Terraform state tools for Service Catalog team workspaces.

The Hub dispatches two workflows from the catalog repository's main branch with the admin's own
GitHub identity: state-list.yml (read-only) and state-rm.yml. main is protected, so whatever is
merged there is the reviewed version. state-rm runs in the team's
GitHub environment, so the team's required reviewers still approve every removal.
"""

import json
import re
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from cloudops_hub import service_catalog as catalog
from cloudops_hub.catalog_review import audited
from cloudops_hub.catalog_tracking import catalog_admin
from cloudops_hub.operations_store import timestamp

router = APIRouter(tags=["terraform-state"])

LIST_WORKFLOW = "state-list.yml"
REMOVE_WORKFLOW = "state-rm.yml"
MAX_ADDRESSES = 5000
MAX_REMOVALS = 200
LIST_MAX_AGE = timedelta(hours=24)
LINE = re.compile(
    r"^(?:\S+Z )?(HUB-STATE-LIST-BEGIN count=(\d+)|HUB-STATE-LIST-END|HUB-STATE\|(.*))$"
)
DATA_SOURCE = re.compile(r"^(module\.[A-Za-z0-9_-]+(\[[^\]]*\])?\.)*data\.")
RECORD = "terraform-state/{run_id}.json"


def is_data_source(address: str) -> bool:
    return bool(DATA_SOURCE.match(address))


def valid_address(address: str) -> bool:
    return (
        0 < len(address) <= 1024
        and not address.startswith("-")
        and not any(char in address for char in "\r\n\x00")
    )


def parse_state_list(log: str) -> list[str] | None:
    """Addresses between the workflow's markers; None when the block is missing or cut."""
    addresses: list[str] = []
    expected = None
    inside = False
    for raw in log.splitlines():
        match = LINE.match(raw.rstrip("\r"))
        if not match:
            continue
        if match.group(2) is not None:
            if expected is not None:
                return None
            expected, inside = int(match.group(2)), True
        elif match.group(1) == "HUB-STATE-LIST-END":
            if not inside:
                return None
            inside = False
            break
        elif inside:
            address = match.group(3)
            if not valid_address(address) or len(addresses) >= MAX_ADDRESSES:
                return None
            addresses.append(address)
    if expected is None or inside or len(addresses) != expected:
        return None
    return addresses


FINISHED = {"succeeded", "failed", "cancelled"}


def dispatch(github: catalog.GitHub, workflow: str, inputs: dict) -> int:
    result = github.call(
        "POST",
        f"{catalog.BASE}/actions/workflows/{workflow}/dispatches",
        body={"ref": "main", "inputs": inputs, "return_run_details": True},
    )
    run_id = result.get("workflow_run_id") if isinstance(result, dict) else None
    if type(run_id) is not int or run_id < 1:
        raise catalog.CatalogError("GitHub did not return the workflow run; check Actions")
    return run_id


def check_target(github: catalog.GitHub, team: str, workspace: str) -> None:
    if workspace not in catalog.WORKSPACES:
        raise HTTPException(status_code=422, detail="Choose a workspace from the list")
    if not catalog.SHORT_TEAM.fullmatch(team) or team not in catalog.catalog_teams(github):
        raise HTTPException(status_code=422, detail="Choose a team from CODEOWNERS")


def github_login(github: catalog.GitHub) -> str:
    login = github.call("GET", "/user").get("login", "")
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", login):
        raise catalog.CatalogError("GitHub did not return a valid user identity")
    return login


def load(request: Request, run_id: int, kind: str) -> dict:
    record = request.app.state.operations.read(RECORD.format(run_id=run_id))
    if record is None or record.get("kind") != kind:
        raise HTTPException(status_code=404, detail="Terraform state run not found")
    return record


def refresh(github: catalog.GitHub, record: dict, workflow: str) -> None:
    run = github.call("GET", f"{catalog.BASE}/actions/runs/{record['run_id']}")
    if (
        run.get("path") != f".github/workflows/{workflow}"
        or run.get("head_branch") != "main"
        or run.get("event") != "workflow_dispatch"
    ):
        raise catalog.CatalogError("GitHub run is not this workflow on main")
    record.update(
        run_status=run.get("status"),
        conclusion=run.get("conclusion"),
        run_url=run.get("html_url", record.get("run_url")),
        run_attempt=run.get("run_attempt"),
        updated_at=timestamp(),
    )
    status = run.get("status")
    if status in {"waiting", "pending", "action_required"}:
        record["status"] = "awaiting_approval"
    elif status != "completed":
        record["status"] = "running"
    elif run.get("conclusion") == "success":
        record["status"] = "succeeded"
    elif run.get("conclusion") == "cancelled":
        record["status"] = "cancelled"
    else:
        record["status"] = "failed"
    if workflow != LIST_WORKFLOW or record["status"] != "succeeded" or "addresses" in record:
        return
    attempt = int(run.get("run_attempt") or 1)
    jobs = github.call(
        "GET", f"{catalog.BASE}/actions/runs/{record['run_id']}/attempts/{attempt}/jobs"
    ).get("jobs", [])
    for job in jobs:
        if job.get("conclusion") == "success" and isinstance(job.get("id"), int):
            addresses = parse_state_list(github.job_log(job["id"]))
            if addresses is not None:
                record["addresses"] = addresses
                record["listed_at"] = timestamp()
                return
    record["status"] = "failed"
    record["error"] = "The state list was not found in the job log"


class ListRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    team: str = Field(min_length=1, max_length=63)
    workspace: str = Field(min_length=1, max_length=40)


class RemoveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    list_run_id: int = Field(gt=0)
    addresses: list[str] = Field(min_length=1, max_length=MAX_REMOVALS)
    reason: str = Field(min_length=10, max_length=500)
    confirmed: bool


@router.get("/api/admin/terraform-state/options")
def options(request: Request, user: dict = Depends(catalog_admin)):
    with catalog.user_github(request, user) as github:
        teams = catalog.catalog_teams(github)
    return {"teams": teams, "workspaces": list(catalog.WORKSPACES)}


@router.get("/api/admin/terraform-state/runs")
def runs(request: Request, _: dict = Depends(catalog_admin)):
    items = request.app.state.operations.list("terraform-state/", limit=200)
    for item in items:
        item.pop("addresses", None)
    return sorted(items, key=lambda item: item["created_at"], reverse=True)[:50]


@router.post("/api/admin/terraform-state/lists", status_code=202)
def start_list(payload: ListRequest, request: Request, user: dict = Depends(catalog_admin)):
    with catalog.user_github(request, user) as github:
        check_target(github, payload.team, payload.workspace)
        login = github_login(github)
        target = f"{payload.team}/{payload.workspace}"
        with audited(
            request, user, login, "terraform_state_list", target, assistant="terraform-state"
        ):
            run_id = dispatch(
                github,
                LIST_WORKFLOW,
                {"environment": payload.team, "WORKSPACE_FOLDER": payload.workspace},
            )
    record = {
        "kind": "list",
        "run_id": run_id,
        "run_url": f"https://github.com/{catalog.REPOSITORY}/actions/runs/{run_id}",
        "team": payload.team,
        "workspace": payload.workspace,
        "requested_by": user.get("email") or user.get("name", ""),
        "github_login": login,
        "created_at": timestamp(),
        "status": "running",
    }
    request.app.state.operations.write(RECORD.format(run_id=run_id), record)
    return record


@router.post("/api/admin/terraform-state/lists/{run_id}/sync")
def sync_list(run_id: int, request: Request, user: dict = Depends(catalog_admin)):
    record = load(request, run_id, "list")
    if record["status"] not in FINISHED:
        with catalog.user_github(request, user) as github:
            refresh(github, record, LIST_WORKFLOW)
        request.app.state.operations.write(RECORD.format(run_id=run_id), record)
    return record


@router.post("/api/admin/terraform-state/removals", status_code=202)
def start_removal(payload: RemoveRequest, request: Request, user: dict = Depends(catalog_admin)):
    if not payload.confirmed:
        raise HTTPException(status_code=422, detail="Confirm the removal first")
    listed = load(request, payload.list_run_id, "list")
    available = set(listed.get("addresses") or [])
    if listed.get("status") != "succeeded" or not available:
        raise HTTPException(status_code=409, detail="Load a successful state list first")
    listed_at = datetime.fromisoformat(listed["listed_at"])
    if datetime.now(UTC) - listed_at > LIST_MAX_AGE:
        raise HTTPException(status_code=409, detail="The state list is older than 24 hours")
    addresses = list(dict.fromkeys(payload.addresses))
    if any(address not in available for address in addresses):
        raise HTTPException(status_code=422, detail="Select addresses from the loaded list")
    if any(is_data_source(address) for address in addresses):
        raise HTTPException(status_code=422, detail="Data sources cannot be removed")
    reason = " ".join(payload.reason.split())
    team, workspace = listed["team"], listed["workspace"]
    with catalog.user_github(request, user) as github:
        check_target(github, team, workspace)
        login = github_login(github)
        with audited(
            request,
            user,
            login,
            "terraform_state_rm",
            f"{team}/{workspace}",
            assistant="terraform-state",
            addresses=addresses,
            reason=reason,
            list_run_id=payload.list_run_id,
        ):
            run_id = dispatch(
                github,
                REMOVE_WORKFLOW,
                {
                    "environment": team,
                    "WORKSPACE_FOLDER": workspace,
                    "ADDRESSES": json_list(addresses),
                    "REASON": reason,
                },
            )
    record = {
        "kind": "remove",
        "run_id": run_id,
        "run_url": f"https://github.com/{catalog.REPOSITORY}/actions/runs/{run_id}",
        "team": team,
        "workspace": workspace,
        "list_run_id": payload.list_run_id,
        "removed": addresses,
        "reason": reason,
        "requested_by": user.get("email") or user.get("name", ""),
        "github_login": login,
        "created_at": timestamp(),
        "status": "awaiting_approval",
    }
    request.app.state.operations.write(RECORD.format(run_id=run_id), record)
    return record


@router.post("/api/admin/terraform-state/removals/{run_id}/sync")
def sync_removal(run_id: int, request: Request, user: dict = Depends(catalog_admin)):
    record = load(request, run_id, "remove")
    if record["status"] not in FINISHED:
        with catalog.user_github(request, user) as github:
            refresh(github, record, REMOVE_WORKFLOW)
        request.app.state.operations.write(RECORD.format(run_id=run_id), record)
    return record


def json_list(addresses: list[str]) -> str:
    value = json.dumps(addresses, separators=(",", ":"))
    if len(value) > 60_000:
        raise HTTPException(status_code=422, detail="Too many addresses for one run")
    return value
