"""Live status, Terraform plan summaries and bounded recovery for Hub catalog requests.

Tracking runs only while the requester's page polls the CSRF-protected sync route, with the
requester's own GitHub user token. Recovery is limited to owner-approved known-error rules
(pinned by solution hash), one attempt per request, and never approves, merges or cancels.
"""

import hashlib
import json
import logging
import re
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from cloudops_hub import catalog_review, known_errors, service_catalog
from cloudops_hub.auth import ADMIN_ROLE, current_user, require_csrf
from cloudops_hub.operations_store import timestamp
from cloudops_hub.repositories import LITERAL_CREDENTIAL
from cloudops_hub.service_catalog import (
    BASE,
    REPOSITORY,
    CatalogError,
    GitHub,
    catalog_user,
    dispatch_workflow,
    request_path,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["service-catalog-tracking"])

WORKSPACES = set(service_catalog.WORKSPACES)
RERUN = "rerun_workspace"
RATE_LIMIT = "rerun_after_rate_limit"
# Owner-approved automatic recovery rules. A changed solution text disables the rule.
APPROVED_RULES = {
    "KE-019": ("0ad561b04717dfc894d2ab041f91d6e233a16cc5417b06d96aacc0cc6dd05bb4", RERUN),
    "KE-020": ("388610207de17248ca1d9283bcc70fa6024f915dc590602aa200197013e57e26", RERUN),
    "KE-022": ("f0c5b7c09e46a16ff0252a7b85e42addcce32e3d4c55410373c0096ab0e62734", RERUN),
    "KE-025": ("f0c5b7c09e46a16ff0252a7b85e42addcce32e3d4c55410373c0096ab0e62734", RERUN),
    "KE-040": ("0490a903b5e71875e6c0c8d605c3000a35f7871908810370cdc044fd975a5f65", RERUN),
    "KE-052": ("c0f8f5e3291242713faaf927aaadd12160b3ec2ee9271e4ac924073bf0a7b769", RERUN),
    "KE-082": ("c0f8f5e3291242713faaf927aaadd12160b3ec2ee9271e4ac924073bf0a7b769", RERUN),
    "KE-068": ("40d43cdc965d7989acc64e625b6bec20e7805511c190226ed6d7bd0d5ff9c889", RERUN),
    "KE-088": ("0a98442305a500bf771efa095988fcaf8c87d7f44bd0b9ec1c2c4a9ee5879027", RERUN),
    "KE-061": ("d8fc83bc0e33ec617199d1dc4e158e131d5ff05035cd54be974b8ec443c4a06a", RATE_LIMIT),
}
RATE_LIMIT_WAIT = timedelta(hours=1, minutes=5)
FAILED = {"failure", "timed_out", "startup_failure"}
WORKSPACE_JOB = re.compile(r"(?:core|level[0-9])-tf-(?:init|apply)-+([A-Za-z_]+)")
PLAN_JOB = re.compile(r"(?i)terraform plan|automations-")
MAX_PLAN_ITEMS = 300
MAX_PLAN_DOWNLOADS = 3

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_STAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z ?")
_FILE = re.compile(r"Terraform Plan Summary:\s*(\S+)")
_ITEM = re.compile(r"🔸\s*([a-z-]+)\s*:\s*(\S+?)\s*(?:🔸\s*\[REASON\]\s*:\s*(.*)|📋\s*(.*))?$")
_ACTIONS = {
    "create": "create",
    "update": "update",
    "delete": "delete",
    "destroy": "delete",
    "forget": "forget",
    "import": "import",
    "replace": "replace",
}
_SECTION = re.compile(r"-{3,}\s*\[([A-Z _]+)\]\s*-{3,}")
_LEVEL_LINE = re.compile(r"(CORE|LEVEL[0-9]{1,2}|AUTOMATIONS)\s*:\s*(\[.*\])\s*$")
_WORKSPACE_NAME = re.compile(r"[A-Za-z0-9_-]{1,64}")
_ADDRESS = re.compile(r"[A-Za-z0-9_.\[\]\"'/:=@-]{1,400}")


def _utc(value: str | None) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def parse_plan_summary(log: str) -> list[dict]:
    """Structured output of generate_plan_summary.sh; warnings and contacts are not kept."""
    files: list[dict] = []
    current: dict | None = None
    items = 0
    for raw in log.splitlines():
        line = _STAMP.sub("", _ANSI.sub("", raw)).strip()
        if heading := _FILE.search(line):
            current = {"file": heading[1][:200], "no_changes": False, "changes": []}
            files.append(current)
            continue
        if current is None:
            continue
        if "No changes detected" in line:
            current["no_changes"] = True
            continue
        item = _ITEM.search(line)
        if not item or item[1] not in _ACTIONS or items >= MAX_PLAN_ITEMS:
            continue
        address, reason = item[2], (item[3] or item[4] or "").strip()
        if not _ADDRESS.fullmatch(address) or LITERAL_CREDENTIAL.search(line):
            continue
        current["changes"].append(
            {
                "action": _ACTIONS[item[1]],
                "address": address,
                "reason": "" if reason in {"", "none"} else reason[:200],
            }
        )
        items += 1
    return files


def parse_changed_workspaces(log: str) -> dict[str, list[str]]:
    """Workspaces per level from the [CHANGED WORKSPACES] block printed by get_workspaces.sh."""
    levels: dict[str, list[str]] = {}
    inside = False
    for raw in log.splitlines():
        line = _STAMP.sub("", _ANSI.sub("", raw)).strip()
        if section := _SECTION.search(line):
            inside = section[1].strip() == "CHANGED WORKSPACES"
            continue
        if not inside or not (found := _LEVEL_LINE.fullmatch(line)):
            continue
        try:
            names = json.loads(found[2])
        except ValueError:
            continue
        if isinstance(names, list):
            levels[found[1]] = [
                name
                for name in names[:100]
                if isinstance(name, str) and _WORKSPACE_NAME.fullmatch(name)
            ]
    return levels


def parse_workspace_settings(log: str) -> dict:
    settings: dict = {"refresh": {}, "workflow_mode": None}
    section = ""
    for raw in log.splitlines():
        line = _STAMP.sub("", _ANSI.sub("", raw)).strip()
        if found := _SECTION.search(line):
            section = found[1].strip()
            continue
        if section == "WORKFLOW MODE" and line.startswith("WORKFLOW_MODE:"):
            try:
                mode = json.loads(line.split(":", 1)[1].strip())
            except ValueError:
                continue
            if isinstance(mode, str) and re.fullmatch(r"[A-Za-z0-9_ .\[\]-]{1,160}", mode):
                settings["workflow_mode"] = mode
        if section == "TERRAFORM REFRESH":
            found = re.fullmatch(r"(CORE|LEVEL[0-9]{1,2}|AUTOMATIONS)\s*:\s*(\{.*\})", line)
            if not found:
                continue
            try:
                values = json.loads(found[2])
            except ValueError:
                continue
            if isinstance(values, dict):
                settings["refresh"][found[1]] = {
                    name: value is True or value == "true"
                    for name, value in list(values.items())[:100]
                    if _WORKSPACE_NAME.fullmatch(name)
                    and (type(value) is bool or value in ("true", "false"))
                }
    return settings


def refresh_workspaces(github: GitHub, record: dict, jobs: list[dict]):
    changed = [
        job
        for job in jobs
        if job.get("name", "").casefold().endswith("changed workspaces")
        and job.get("status") == "completed"
        and job.get("conclusion") == "success"
    ]
    cached = record.get("workspaces") or {}
    if len(changed) != 1 or (
        cached.get("job_id") == changed[0]["id"] and cached.get("settings_version") == 1
    ):
        return
    try:
        log = github.job_log(changed[0]["id"])
        levels = parse_changed_workspaces(log)
    except CatalogError as error:
        logger.warning("Could not read changed workspaces job %s: %s", changed[0]["id"], error)
        record["workspace_error"] = "Could not read CHANGED WORKSPACES. Review the job in GitHub."
        return
    if levels:
        record["workspaces"] = {
            "job_id": changed[0]["id"],
            "levels": levels,
            "settings_version": 1,
            **parse_workspace_settings(log),
        }
        record.pop("workspace_error", None)
    else:
        record["workspace_error"] = "No workspace summary was found. Review the job in GitHub."


def plan_totals(plans: dict) -> dict:
    totals: dict[str, int] = {}
    for plan in plans.values():
        for summary in plan.get("files", []):
            for change in summary["changes"]:
                totals[change["action"]] = totals.get(change["action"], 0) + 1
    return totals


def _run_view(run: dict) -> dict:
    return {
        key: run.get(key)
        for key in (
            "id",
            "run_attempt",
            "status",
            "conclusion",
            "html_url",
            "display_title",
            "event",
            "created_at",
            "updated_at",
        )
    }


def _claim_run(store, run_id: int, request_id: str) -> bool:
    if store.write(f"claims/runs/{run_id}.json", {"request_id": request_id}, create=True):
        return True
    return (store.read(f"claims/runs/{run_id}.json") or {}).get("request_id") == request_id


def _candidate_runs(github: GitHub, since: datetime) -> list[dict]:
    result = github.call(
        "GET",
        f"{BASE}/actions/workflows/caller.yml/runs",
        params={"branch": "main", "per_page": 50, "created": f">={since:%Y-%m-%dT%H:%M:%SZ}"},
    )
    runs = result.get("workflow_runs", []) if isinstance(result, dict) else []
    return sorted(
        (run for run in runs if isinstance(run, dict) and type(run.get("id")) is int),
        key=lambda run: run.get("created_at", ""),
    )


def correlate(github: GitHub, store, record: dict) -> int | None:
    """Link only an exact, unclaimed run; ambiguous evidence leaves the request unlinked."""
    team = record["team"]
    if record.get("dispatched_run_id"):
        run_id = record["dispatched_run_id"]
        return run_id if _claim_run(store, run_id, record["id"]) else None
    pr = record.get("github_pr") or {}
    if pr.get("merged") and pr.get("merge_commit_sha") and pr.get("merged_at"):
        since = _utc(pr["merged_at"])
        expected = ("on push event", pr["merge_commit_sha"])
        login = None
    elif record.get("dispatched_at"):
        since = _utc(record["dispatched_at"])
        expected = ("on workflow_dispatch event", None)
        login = record["github_actor"]["login"]
    else:
        return None
    if since is None:
        return None
    for run in _candidate_runs(github, since - timedelta(seconds=10)):
        created = _utc(run.get("created_at"))
        title = run.get("display_title", "")
        if (
            created is None
            or created > since + timedelta(minutes=15)
            or f" to {team} environment " not in title
            or expected[0] not in title
            or (expected[1] and run.get("head_sha") != expected[1])
            or (login and (run.get("actor") or {}).get("login") != login)
            or run.get("event") != "workflow_dispatch"
        ):
            continue
        if _claim_run(store, run["id"], record["id"]):
            record["correlation"] = (
                "merge commit and team" if expected[1] else "your GitHub user, team and time"
            )
            return run["id"]
    return None


def jobs_for(github: GitHub, run_id: int, attempt: int) -> list[dict]:
    jobs: list[dict] = []
    for page in range(1, 6):
        result = github.call(
            "GET",
            f"{BASE}/actions/runs/{run_id}/attempts/{attempt}/jobs",
            params={"per_page": 100, "page": page},
        )
        batch = result.get("jobs") if isinstance(result, dict) else None
        if not isinstance(batch, list):
            raise CatalogError("GitHub returned an invalid job listing")
        jobs.extend(job for job in batch if isinstance(job, dict) and type(job.get("id")) is int)
        if len(batch) < 100:
            return jobs
    return jobs


def _job_view(job: dict) -> dict:
    return {
        key: job.get(key)
        for key in ("id", "name", "status", "conclusion", "html_url", "started_at", "completed_at")
    }


def decide(record: dict, match: dict | None, jobs: list[dict], run: dict, now: datetime) -> dict:
    if match is None:
        return {"action": "escalate", "reason": "No known-error record matched the failed logs"}
    rule = APPROVED_RULES.get(match["id"])
    if rule is None:
        return {"action": "escalate", "reason": f"{match['id']} has no automatic recovery rule"}
    digest = hashlib.sha256(match["solution"].encode()).hexdigest()
    if digest != rule[0]:
        return {
            "action": "escalate",
            "reason": f"{match['id']} guidance changed; automatic rule is disabled until reviewed",
        }
    failed = [job for job in jobs if job.get("conclusion") in FAILED]
    if rule[1] == RATE_LIMIT:
        updated = _utc(run.get("updated_at")) or now
        if now < updated + RATE_LIMIT_WAIT:
            return {"action": "wait", "not_before": (updated + RATE_LIMIT_WAIT).isoformat()}
        return {"action": "rerun_failed"}
    if record.get("mode", "").startswith("DEFAULT") or record["operation"] in {
        "create-group",
        "create-resource",
    }:
        changed = [job for job in jobs if "changed workspaces" in job.get("name", "").casefold()]
        if len(changed) != 1:
            return {"action": "escalate", "reason": "CHANGED WORKSPACES job was not found"}
        return {"action": "rerun_job", "job_id": changed[0]["id"]}
    workspaces = {
        found[1].lower()
        for job in failed
        if (found := WORKSPACE_JOB.search(job.get("name", "").rsplit(" / ", 1)[-1]))
    }
    if len(workspaces) != 1 or not workspaces <= WORKSPACES:
        return {"action": "escalate", "reason": "Failing workspace is ambiguous or not allowed"}
    return {"action": "dispatch_custom", "workspace": workspaces.pop()}


def known_error_match(settings, github: GitHub, run_id: int, jobs: list[dict]):
    from cloudops_hub.live_catalog import live_reader

    logs = []
    for job in [job for job in jobs if job.get("conclusion") in FAILED][:3]:
        logs.append(github.job_log(job["id"])[-200_000:])
    with live_reader(settings) as reader:
        knowledge = reader.known_error_records()
    matches = known_errors.match(knowledge, logs, str(run_id))
    excerpt = known_errors.error_excerpt(logs)
    if LITERAL_CREDENTIAL.search(excerpt):
        excerpt = "Excerpt withheld: possible credential in log output."
    return (matches[0] if matches else None), excerpt


def escalate(store, record: dict, key: str, reason: str, match: dict | None, excerpt: str):
    run = record.get("run") or {}
    store.write(
        f"escalations/{record['id']}-{key}.json",
        {
            "id": f"{record['id']}-{key}",
            "request_id": record["id"],
            "owner": record["owner_path"],
            "team": record["team"],
            "github_actor": record["github_actor"]["login"],
            "operation": record["operation"],
            "mode": record.get("mode"),
            "run_url": run.get("html_url"),
            "known_error": match["id"] if match else None,
            "known_error_title": match["title"] if match else None,
            "reason": reason,
            "excerpt": excerpt[:2000],
            "status": "open",
            "created_at": timestamp(),
        },
        create=True,
    )
    record["status"] = "escalated"
    record["escalation"] = {"reason": reason, "known_error": match["id"] if match else None}


def handle_failure(github: GitHub, store, settings, record: dict, run: dict, jobs: list[dict]):
    key = f"{run['id']}-{run.get('run_attempt', 1)}"
    if record.get("handled_failure") == key:
        return
    pending = record.get("pending_recovery") or {}
    if pending.get("key") == key and datetime.now(UTC) < _utc(pending["not_before"]):
        return
    match, excerpt = None, ""
    if record.get("recovery") or record.get("escalation"):
        reason = "The workflow failed again after an automatic recovery attempt"
        decision = {"action": "escalate", "reason": reason}
    else:
        try:
            match, excerpt = known_error_match(settings, github, run["id"], jobs)
        except (CatalogError, RuntimeError) as exc:
            decision = {"action": "escalate", "reason": f"Diagnosis unavailable: {exc}"[:300]}
        else:
            decision = decide(record, match, jobs, run, datetime.now(UTC))
    if decision["action"] == "wait":
        record["status"] = "recovery_scheduled"
        record["pending_recovery"] = {"key": key, "known_error": match["id"], **decision}
        return
    record.pop("pending_recovery", None)
    if decision["action"] == "escalate":
        escalate(store, record, key, decision["reason"], match, excerpt)
        record["handled_failure"] = key
        return
    if not store.write(f"claims/recovery/{key}.json", {"request_id": record["id"]}, create=True):
        return
    recovery = {
        "known_error": match["id"],
        "known_error_title": match["title"],
        "action": decision["action"],
        "source_run_id": run["id"],
        "source_attempt": run.get("run_attempt", 1),
        "started_at": timestamp(),
    }
    record["recovery"] = recovery
    record["handled_failure"] = key
    try:
        if decision["action"] == "rerun_job":
            github.call("POST", f"{BASE}/actions/jobs/{decision['job_id']}/rerun", body={})
        elif decision["action"] == "rerun_failed":
            github.call("POST", f"{BASE}/actions/runs/{run['id']}/rerun-failed-jobs", body={})
        else:
            recovery["workspace"] = decision["workspace"]
            record["previous_runs"] = [*record.get("previous_runs", []), record["run"]]
            record["dispatched_at"] = timestamp()
            record["dispatched_run_id"] = dispatch_workflow(
                github, record["team"], "CUSTOM", decision["workspace"]
            )
            record["run_id"] = None
            record["run"] = None
            record["jobs"] = []
        recovery["result"] = "started"
        record["status"] = "recovering"
    except (CatalogError, HTTPException) as exc:
        recovery["result"] = "failed"
        if isinstance(exc, HTTPException):
            detail = exc.detail
        else:
            logger.warning("Automatic recovery could not start for %s", key)
            detail = "GitHub request failed; see the Hub logs"
        escalate(store, record, key, f"Automatic recovery could not start: {detail}", match, "")


def refresh_plans(github: GitHub, record: dict, jobs: list[dict]):
    plans = record.setdefault("plans", {})
    downloads = 0
    for job in jobs:
        if (
            downloads >= MAX_PLAN_DOWNLOADS
            or str(job["id"]) in plans
            or job.get("status") != "completed"
            or job.get("conclusion") != "success"
            or not PLAN_JOB.search(job.get("name", ""))
        ):
            continue
        downloads += 1
        try:
            files = parse_plan_summary(github.job_log(job["id"]))
        except CatalogError:
            logger.warning("Plan summary could not be read for job %s", job["id"])
            plans[str(job["id"])] = {
                "job": job.get("name"),
                "error": "The plan summary could not be read from the job log",
                "files": [],
            }
            continue
        plans[str(job["id"])] = {
            "job": job.get("name"),
            "html_url": job.get("html_url"),
            "files": files,
        }
    record["plan_totals"] = plan_totals(plans)


def sync_record(github: GitHub, store, settings, record: dict) -> dict:
    if record.get("pr_number"):
        pr = github.call("GET", f"{BASE}/pulls/{record['pr_number']}")
        record["github_pr"] = {
            key: pr.get(key)
            for key in ("state", "merged", "merge_commit_sha", "merged_at", "html_url")
        }
        if not pr.get("merged"):
            record["status"] = (
                "closed_without_merge" if pr.get("state") == "closed" else "awaiting_review"
            )
            record["last_synced_at"] = timestamp()
            return record
    if not record.get("run_id"):
        record["run_id"] = correlate(github, store, record)
        if not record["run_id"]:
            record["status"] = (
                "merged_awaiting_execution" if record.get("pr_number") else "waiting_for_run"
            )
            record["last_synced_at"] = timestamp()
            return record
    run = github.call("GET", f"{BASE}/actions/runs/{record['run_id']}")
    if run.get("id") != record["run_id"] or not str(run.get("path", "")).startswith(
        ".github/workflows/caller.yml"
    ):
        raise CatalogError("GitHub returned a mismatched workflow run")
    attempt = run.get("run_attempt") if type(run.get("run_attempt")) is int else 1
    pending_attempt = record.get("rerun_pending_attempt")
    if pending_attempt is not None:
        if attempt <= pending_attempt:
            record["status"] = "running"
            record["last_synced_at"] = timestamp()
            return record
        record.pop("rerun_pending_attempt")
    jobs = jobs_for(github, run["id"], attempt)
    record["run"] = _run_view(run)
    record["jobs"] = [_job_view(job) for job in jobs]
    refresh_plans(github, record, jobs)
    refresh_workspaces(github, record, jobs)
    waiting = run.get("status") == "waiting" or any(j.get("status") == "waiting" for j in jobs)
    if run.get("status") != "completed":
        record["status"] = "awaiting_approval" if waiting else "running"
    elif run.get("conclusion") == "success":
        record["status"] = "completed"
    elif run.get("conclusion") in FAILED:
        handle_failure(github, store, settings, record, run, jobs)
    else:
        record["status"] = run.get("conclusion") or "completed"
    record["last_synced_at"] = timestamp()
    return record


class WorkflowAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    confirmed: bool


@router.post("/api/catalog/requests/{identifier}/rerun-workspaces")
def rerun_workspaces(
    identifier: UUID,
    payload: WorkflowAction,
    request: Request,
    user: dict = Depends(catalog_user),
):
    return rerun_request_job(identifier, payload, request, user)


@router.post("/api/catalog/requests/{identifier}/jobs/{job_id}/rerun")
def rerun_stage(
    identifier: UUID,
    job_id: int,
    payload: WorkflowAction,
    request: Request,
    user: dict = Depends(catalog_user),
):
    return rerun_request_job(identifier, payload, request, user, job_id)


def matrix_stage(name: str) -> str | None:
    if re.fullmatch(r"automations-.+", name.split("/")[-1].strip(), re.IGNORECASE):
        return "automations"
    found = re.fullmatch(
        r"((?:core|level[0-9]+)-tf-(?:init|apply))-+.+",
        name.split("/")[-1].strip(),
        re.IGNORECASE,
    )
    return found[1].lower() if found else None


def rerun_request_job(
    identifier: UUID,
    payload: WorkflowAction,
    request: Request,
    user: dict,
    requested_job_id: int | None = None,
):
    if not payload.confirmed:
        raise HTTPException(status_code=400, detail="Confirm the rerun before submitting")
    store = request.app.state.operations
    path = request_path(user, identifier)
    record = store.read(path)
    if record is None:
        raise HTTPException(status_code=404, detail="Request not found")
    if record.get("repository") != REPOSITORY or not record.get("run_id"):
        raise HTTPException(status_code=409, detail="No Service Catalog run is linked")
    with service_catalog.user_github(request, user) as github:
        actor = github.authorize(record["team"])
        run = catalog_review.load_run(github, record["run_id"])
        if run.get("id") != record["run_id"] or run.get("path") != ".github/workflows/caller.yml":
            raise HTTPException(
                status_code=409,
                detail="The linked run is not the Service Catalog workflow",
            )
        attempt = run.get("run_attempt", 1)
        if record.get("rerun_pending_attempt", 0) >= attempt:
            raise HTTPException(status_code=409, detail="A rerun is already requested")
        if run.get("status") != "completed":
            raise HTTPException(
                status_code=409,
                detail="Wait for the workflow to finish before rerunning",
            )
        created = _utc(run.get("created_at"))
        if created and datetime.now(UTC) - created > timedelta(days=30):
            raise HTTPException(status_code=409, detail="GitHub's 30-day rerun window has expired")
        if attempt >= 51:
            raise HTTPException(
                status_code=409, detail="GitHub's limit of 50 reruns has been reached"
            )
        jobs = jobs_for(github, record["run_id"], attempt)
        if not jobs or any(job.get("status") != "completed" for job in jobs):
            raise HTTPException(
                status_code=409,
                detail="Wait for all jobs to finish before rerunning",
            )
        if requested_job_id is None:
            changed = [
                job
                for job in jobs
                if str(job.get("name", "")).split("/")[-1].strip() == "CHANGED WORKSPACES"
            ]
        else:
            selected = next((job for job in jobs if job["id"] == requested_job_id), None)
            stage = matrix_stage(selected.get("name", "")) if selected else None
            changed = [job for job in jobs if stage and matrix_stage(job.get("name", "")) == stage]
            if len(changed) != 1 or changed[0]["id"] != requested_job_id:
                raise HTTPException(
                    status_code=409,
                    detail="Only a stage with exactly one workspace job can be rerun",
                )
            level = (
                "AUTOMATIONS"
                if stage == "automations"
                else ("CORE" if stage.startswith("core-") else stage.split("-")[0].upper())
            )
            planned = (record.get("workspaces") or {}).get("levels", {}).get(level)
            if planned is not None and len(planned) != 1:
                raise HTTPException(
                    status_code=409,
                    detail="This level does not contain exactly one workspace",
                )
        if len(changed) != 1 or type(changed[0].get("id")) is not int:
            raise HTTPException(
                status_code=409,
                detail="The run has no unique CHANGED WORKSPACES job",
            )
        job_id = changed[0]["id"]
        try:
            with catalog_review.audited(
                request,
                user,
                actor["login"],
                "catalog_rerun_workspaces" if requested_job_id is None else "catalog_rerun_stage",
                f"run/{record['run_id']}/job/{job_id}",
            ):
                github.call("POST", f"{BASE}/actions/jobs/{job_id}/rerun", body={})
        except service_catalog.GitHubRefused as error:
            catalog_review.refused(error)
    record["status"] = "running"
    record["rerun_pending_attempt"] = attempt
    for key in ("plans", "plan_totals", "workspaces", "pending_recovery"):
        record.pop(key, None)
    store.write(path, record)
    return {"status": "accepted", "job_id": job_id}


@router.post("/api/catalog/requests/{identifier}/cancel")
def cancel_workflow(
    identifier: UUID,
    payload: WorkflowAction,
    request: Request,
    user: dict = Depends(catalog_user),
):
    if not payload.confirmed:
        raise HTTPException(status_code=400, detail="Confirm the cancellation before submitting")
    store = request.app.state.operations
    path = request_path(user, identifier)
    record = store.read(path)
    if record is None:
        raise HTTPException(status_code=404, detail="Request not found")
    if record.get("repository") != REPOSITORY or not record.get("run_id"):
        raise HTTPException(status_code=409, detail="No Service Catalog run is linked")
    with service_catalog.user_github(request, user) as github:
        actor = github.authorize(record["team"])
        run = catalog_review.load_run(github, record["run_id"])
        if run.get("id") != record["run_id"] or run.get("path") != ".github/workflows/caller.yml":
            raise HTTPException(status_code=409, detail="The linked run is not caller.yml")
        if run.get("status") not in {"queued", "in_progress", "waiting", "pending", "requested"}:
            raise HTTPException(status_code=409, detail="This workflow is no longer active")
        try:
            with catalog_review.audited(
                request,
                user,
                actor["login"],
                "catalog_cancel_workflow",
                f"run/{record['run_id']}",
            ):
                github.call("POST", f"{BASE}/actions/runs/{record['run_id']}/cancel", body={})
        except service_catalog.GitHubRefused as error:
            catalog_review.refused(error)
    return {"status": "accepted"}


@router.post("/api/catalog/requests/{identifier}/sync")
def sync(identifier: UUID, request: Request, user: dict = Depends(catalog_user)):
    store = request.app.state.operations
    path = request_path(user, identifier)
    record = store.read(path)
    if record is None:
        raise HTTPException(status_code=404, detail="Request not found")
    if record.get("repository") == REPOSITORY and record.get("status") in {"completed", "resolved"}:
        if (record.get("workspaces") or {}).get("settings_version") != 1:
            with service_catalog.user_github(request, user) as github:
                github.authorize(record["team"])
                refresh_workspaces(github, record, record.get("jobs") or [])
            store.write(path, record)
        return record
    if record.get("repository") != REPOSITORY or record.get("status") in {
        "submitting",
        "needs_attention",
        "completed",
        "closed_without_merge",
        "resolved",
    }:
        return record
    record["owner_path"] = path
    with service_catalog.user_github(request, user) as github:
        github.authorize(record["team"])
        sync_record(github, store, request.app.state.settings, record)
    store.write(path, record)
    return record


def catalog_admin(request: Request) -> dict:
    user = current_user(request)
    if request.method != "GET":
        require_csrf(request)
    if ADMIN_ROLE not in user.get("roles", []):
        raise HTTPException(status_code=403, detail="CloudOps Hub Admin role is required")
    return user


@router.get("/api/admin/catalog-escalations")
def escalations(request: Request, _: dict = Depends(catalog_admin)) -> list[dict[str, Any]]:
    items = request.app.state.operations.list("escalations/", limit=500)
    return sorted(items, key=lambda item: item["created_at"], reverse=True)


class Resolution(BaseModel):
    model_config = ConfigDict(extra="forbid")
    note: str = Field(min_length=1, max_length=1000)


@router.post("/api/admin/catalog-escalations/{identifier}/resolve")
def resolve(
    identifier: str, payload: Resolution, request: Request, user: dict = Depends(catalog_admin)
):
    if not re.fullmatch(r"[a-f0-9]{32}-[0-9]+-[0-9]+", identifier):
        raise HTTPException(status_code=404, detail="Escalation not found")
    store = request.app.state.operations
    name = f"escalations/{identifier}.json"
    item = store.read(name)
    if item is None:
        raise HTTPException(status_code=404, detail="Escalation not found")
    item.update(
        status="resolved",
        resolution=payload.note,
        resolved_by=user.get("email") or user.get("name"),
        resolved_at=timestamp(),
    )
    store.write(name, item)
    owner = item.get("owner", "")
    if re.fullmatch(r"requests/[a-f0-9]{64}/[a-f0-9]{32}\.json", owner):
        record = store.read(owner)
        if record is not None:
            record["escalation"] = {
                **record.get("escalation", {}),
                "resolved": True,
                "resolution": payload.note,
                "resolved_at": item["resolved_at"],
            }
            if record.get("status") == "escalated":
                record["status"] = "resolved"
            store.write(owner, record)
    return item
