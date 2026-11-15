"""Review, merge and deployment approval for Service Catalog work, inside the Hub.

Every action runs with the signed-in user's own GitHub token against
almbrand/landingzone-service-catalog only. The Hub never approves on its own; GitHub keeps
enforcing CODEOWNERS, branch protection/rulesets, no self-approval and environment reviewers,
and its refusal reason is shown to the user. Only PRs that change catalog JSON files can be
merged here, and the reviewed head commit is pinned so a later push cannot slip in.
"""

import logging
import re
import time
import uuid
from contextlib import contextmanager
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from cloudops_hub import service_catalog as catalog


def log_safe(value) -> str:
    """Strip line breaks so user-provided values cannot forge log entries."""
    return str(value).replace("\r", "").replace("\n", "")


router = APIRouter(tags=["service-catalog-review"])
logger = logging.getLogger(__name__)
CALLER = ".github/workflows/caller.yml"
# Runs whose environment approvals can be decided in the Hub.
APPROVABLE = {CALLER, ".github/workflows/state-rm.yml", ".github/workflows/import.yml"}
MERGEABLE = {"clean", "unstable", "has_hooks"}
MAX_LISTED_PRS = 30
SHA = r"[0-9a-f]{40}"
NAME_TTL_SECONDS = 6 * 3600
_names: dict[str, tuple[float, str]] = {}


class ReviewAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    event: Literal["APPROVE", "REQUEST_CHANGES", "COMMENT"]
    body: str = Field(default="", max_length=2000)
    head_sha: str = Field(pattern=SHA)
    confirmed: Literal[True]


class MergeAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    head_sha: str = Field(pattern=SHA)
    bypass: bool = False
    reason: str = Field(default="", max_length=500)
    confirmed: Literal[True]


class DeploymentAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    state: Literal["approved", "rejected"]
    environment_ids: list[int] = Field(min_length=1, max_length=10)
    comment: str = Field(default="", max_length=1000)
    confirmed: Literal[True]


def display_names(github: catalog.GitHub, logins) -> dict[str, str]:
    """Profile names for GitHub logins (enterprise accounts get them from Entra). Best effort."""
    now = time.monotonic()
    names = {}
    for login in {str(login) for login in logins if login}:
        cached = _names.get(login)
        if not cached or now - cached[0] > NAME_TTL_SECONDS:
            try:
                profile = github.call("GET", f"/users/{login}")
                name = str((profile or {}).get("name") or "").strip()[:100]
            except Exception:  # noqa: BLE001 - a missing name must not break the review
                logger.info("GitHub profile name lookup failed for a reviewer row")
                name = cached[1] if cached else ""
            cached = (now, name)
            _names[login] = cached
        if cached[1]:
            names[login] = cached[1]
    return names


def catalog_file(name: str) -> bool:
    return (
        bool(re.fullmatch(catalog.CONTENT_FILE, name))
        and not name.startswith(".github/")
        and name != "CODEOWNERS"
    )


def viewer(github: catalog.GitHub) -> dict:
    login = str(github.call("GET", "/user").get("login", ""))
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", login):
        raise catalog.CatalogError("GitHub did not return a valid user identity")
    permissions = github.call("GET", catalog.BASE).get("permissions", {})
    return {
        "login": login,
        "admin": bool(permissions.get("admin")),
        "teams": {item["name"] for item in github.teams()},
    }


def pr_files(github: catalog.GitHub, number: int) -> list[dict]:
    files = github.call("GET", f"{catalog.BASE}/pulls/{number}/files", params={"per_page": 100})
    return files if isinstance(files, list) else []


def scope(pr: dict, files: list[dict], who: dict) -> dict:
    """What this user may do with the PR, decided before any GitHub write."""
    names = [str(item.get("filename", "")) for item in files]
    is_catalog = (
        0 < len(files) < 100
        and all(
            catalog_file(name) and item.get("status") in {"added", "modified"}
            for name, item in zip(names, files, strict=True)
        )
        and pr.get("base", {}).get("ref") == "main"
        and pr.get("head", {}).get("repo", {}).get("full_name") == catalog.REPOSITORY
    )
    folders = sorted({name.split("/", 1)[0] for name in names})
    owner = is_catalog and set(folders) <= who["teams"]
    mine = str(pr.get("user", {}).get("login", "")).casefold() == who["login"].casefold()
    open_ = pr.get("state") == "open" and not pr.get("merged")
    return {
        "catalog": is_catalog,
        "teams": folders,
        "mine": mine,
        "owner": owner,
        "can_review": bool(owner and not mine and open_),
        "can_merge": bool(is_catalog and open_ and (owner or who["admin"])),
        "can_bypass": bool(is_catalog and open_ and who["admin"]),
    }


@contextmanager
def audited(request: Request, user: dict, login: str, operation: str, target: str, **details):
    """Record the attempt before the GitHub write (fail closed), then its outcome."""
    store = request.app.state.operations
    event = {
        "category": "catalog",
        "account": "AB",
        "assistant": "service-catalog",
        "operation": operation,
        "operation_id": uuid.uuid4().hex,
        "outcome": "started",
        "ab_oid": user["oid"],
        "user_name": user.get("name", ""),
        "user_email": user.get("email", ""),
        "github_login": login,
        "target": target,
        "repository": catalog.REPOSITORY,
        **details,
    }
    store.audit(event)
    try:
        yield
    except catalog.GitHubRefused:
        store.audit({**event, "outcome": "refused"})
        raise
    except Exception:
        store.audit({**event, "outcome": "failed"})
        raise
    store.audit({**event, "outcome": "succeeded"})


def refused(error: catalog.GitHubRefused):
    raise HTTPException(status_code=409, detail=f"GitHub refused: {error}") from None


@router.get("/api/catalog/reviews")
def review_queue(request: Request, user: dict = Depends(catalog.catalog_user)):
    with catalog.user_github(request, user) as github:
        who = viewer(github)
        items = []
        pulls = github.call(
            "GET",
            f"{catalog.BASE}/pulls",
            params={"state": "open", "base": "main", "per_page": MAX_LISTED_PRS},
        )
        for pr in pulls if isinstance(pulls, list) else []:
            allowed = scope(pr, pr_files(github, int(pr["number"])), who)
            if not allowed["catalog"] or not (allowed["owner"] or who["admin"]):
                continue
            items.append(
                {
                    "kind": "pr",
                    "number": pr["number"],
                    "title": str(pr.get("title", ""))[:200],
                    "author": pr.get("user", {}).get("login"),
                    "created_at": pr.get("created_at"),
                    "html_url": pr.get("html_url"),
                    "draft": bool(pr.get("draft")),
                    "head_sha": pr.get("head", {}).get("sha"),
                    **{key: allowed[key] for key in ("teams", "mine", "can_review", "can_bypass")},
                }
            )
        runs = github.call(
            "GET", f"{catalog.BASE}/actions/runs", params={"status": "waiting", "per_page": 20}
        )
        for run in (runs or {}).get("workflow_runs", [])[:20]:
            if run.get("path") != CALLER:
                continue
            pending = github.call(
                "GET", f"{catalog.BASE}/actions/runs/{int(run['id'])}/pending_deployments"
            )
            environments = [
                item["environment"]["name"]
                for item in pending or []
                if item.get("current_user_can_approve")
            ]
            if environments:
                items.append(
                    {
                        "kind": "run",
                        "id": run["id"],
                        "run_number": run.get("run_number"),
                        "title": str(run.get("display_title", ""))[:200],
                        "author": run.get("actor", {}).get("login"),
                        "created_at": run.get("created_at"),
                        "html_url": run.get("html_url"),
                        "environments": environments,
                    }
                )
        names = display_names(github, (item.get("author") for item in items))
    for item in items:
        item["author_name"] = names.get(str(item.get("author")), "")
    items.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
    return {"login": who["login"], "admin": who["admin"], "items": items}


def load_pr(github: catalog.GitHub, number: int) -> dict:
    pr = github.call("GET", f"{catalog.BASE}/pulls/{number}")
    if not isinstance(pr, dict):
        raise catalog.CatalogError("GitHub returned an invalid pull request")
    return pr


@router.get("/api/catalog/reviews/pr/{number}")
def pr_review(number: int, request: Request, user: dict = Depends(catalog.catalog_user)):
    if number < 1:
        raise HTTPException(status_code=404, detail="Pull request not found")
    with catalog.user_github(request, user) as github:
        who = viewer(github)
        pr = load_pr(github, number)
        files = pr_files(github, number)
        allowed = scope(pr, files, who)
        reviews = github.call(
            "GET", f"{catalog.BASE}/pulls/{number}/reviews", params={"per_page": 100}
        )
        reviewers = [r.get("user", {}).get("login") for r in reviews or [] if isinstance(r, dict)]
        names = display_names(github, [pr.get("user", {}).get("login"), *reviewers])
    latest: dict[str, dict] = {}
    for review in reviews if isinstance(reviews, list) else []:
        login = review.get("user", {}).get("login")
        if login and review.get("state") in {"APPROVED", "CHANGES_REQUESTED", "DISMISSED"}:
            latest[login] = {
                "login": login,
                "name": names.get(login, ""),
                "state": review["state"],
                "submitted_at": review.get("submitted_at"),
            }
    return {
        "number": number,
        "title": str(pr.get("title", ""))[:200],
        "author": pr.get("user", {}).get("login"),
        "author_name": names.get(str(pr.get("user", {}).get("login")), ""),
        "html_url": pr.get("html_url"),
        "state": "merged" if pr.get("merged") else pr.get("state"),
        "draft": bool(pr.get("draft")),
        "head_sha": pr.get("head", {}).get("sha"),
        "mergeable_state": pr.get("mergeable_state") or "unknown",
        "files": [
            {
                "filename": item.get("filename"),
                "status": item.get("status"),
                "additions": item.get("additions"),
                "deletions": item.get("deletions"),
                "patch": str(item.get("patch") or "")[:20_000],
            }
            for item in files[:20]
        ],
        "reviews": list(latest.values()),
        "login": who["login"],
        "admin": who["admin"],
        **allowed,
    }


def current_pr(github: catalog.GitHub, number: int, head_sha: str) -> tuple[dict, dict, dict]:
    who = viewer(github)
    pr = load_pr(github, number)
    allowed = scope(pr, pr_files(github, number), who)
    if not allowed["catalog"]:
        raise HTTPException(
            status_code=403,
            detail="Only pull requests that change catalog JSON files can be handled in the Hub",
        )
    if pr.get("head", {}).get("sha") != head_sha:
        raise HTTPException(
            status_code=409, detail="The pull request changed since you loaded it. Review again."
        )
    return who, pr, allowed


@router.post("/api/catalog/reviews/pr/{number}/review")
def submit_review(
    number: int, payload: ReviewAction, request: Request, user: dict = Depends(catalog.catalog_user)
):
    if payload.event != "APPROVE" and not payload.body.strip():
        raise HTTPException(status_code=422, detail="Add a comment explaining the requested change")
    with catalog.user_github(request, user) as github:
        who, _, allowed = current_pr(github, number, payload.head_sha)
        if not allowed["can_review"]:
            raise HTTPException(
                status_code=403,
                detail="You can't review this PR: it is yours or touches a team you don't maintain",
            )
        try:
            with audited(
                request,
                user,
                who["login"],
                "catalog_pr_review",
                f"pull/{number}",
                review=payload.event,
                head_sha=payload.head_sha,
            ):
                github.call(
                    "POST",
                    f"{catalog.BASE}/pulls/{number}/reviews",
                    body={
                        "commit_id": payload.head_sha,
                        "event": payload.event,
                        "body": payload.body.strip(),
                    },
                )
        except catalog.GitHubRefused as error:
            refused(error)
    logger.info(
        "Catalog PR %s review %s by %s",
        int(number),
        log_safe(payload.event),
        log_safe(who["login"]),
    )
    return {"status": "submitted", "event": payload.event}


@router.post("/api/catalog/reviews/pr/{number}/merge")
def merge_pr(
    number: int, payload: MergeAction, request: Request, user: dict = Depends(catalog.catalog_user)
):
    with catalog.user_github(request, user) as github:
        who, pr, allowed = current_pr(github, number, payload.head_sha)
        state = pr.get("mergeable_state") or "unknown"
        if state == "dirty":
            raise HTTPException(status_code=409, detail="The PR has merge conflicts")
        if state == "unknown":
            raise HTTPException(
                status_code=409, detail="GitHub is still checking the PR. Refresh in a moment."
            )
        reason = payload.reason.strip()
        if payload.bypass:
            if not allowed["can_bypass"]:
                raise HTTPException(
                    status_code=403, detail="Only repository admins can bypass and merge"
                )
        else:
            if not allowed["can_merge"]:
                raise HTTPException(
                    status_code=403, detail="You can only merge PRs for teams you maintain"
                )
            if state not in MERGEABLE:
                # Admin tokens would otherwise bypass protection silently; that needs an
                # explicit bypass instead.
                raise HTTPException(
                    status_code=409,
                    detail="Required reviews or checks are not satisfied yet",
                )
        try:
            with audited(
                request,
                user,
                who["login"],
                "catalog_pr_bypass_merge" if payload.bypass else "catalog_pr_merge",
                f"pull/{number}",
                head_sha=payload.head_sha,
                mergeable_state=state,
                **({"reason": reason} if payload.bypass else {}),
            ):
                if payload.bypass:
                    github.call(
                        "POST",
                        f"{catalog.BASE}/issues/{number}/comments",
                        body={
                            "body": f"Merged from CloudOps Hub by @{who['login']} bypassing "
                            f"branch protection." + (f"\n\nReason: {reason}" if reason else "")
                        },
                    )
                result = github.call(
                    "PUT",
                    f"{catalog.BASE}/pulls/{number}/merge",
                    body={"sha": payload.head_sha, "merge_method": "squash"},
                )
        except catalog.GitHubRefused as error:
            refused(error)
    logger.info(
        "Catalog PR %s merged by %s%s",
        int(number),
        log_safe(who["login"]),
        " (bypass)" if payload.bypass else "",
    )
    return {"status": "merged", "sha": (result or {}).get("sha")}


def load_run(github: catalog.GitHub, run_id: int) -> dict:
    run = github.call("GET", f"{catalog.BASE}/actions/runs/{run_id}")
    if not isinstance(run, dict) or run.get("path") not in APPROVABLE:
        raise HTTPException(
            status_code=403,
            detail="Only caller.yml, state-rm.yml and import.yml runs can be approved here",
        )
    return run


def pending(github: catalog.GitHub, run_id: int) -> list[dict]:
    items = github.call("GET", f"{catalog.BASE}/actions/runs/{run_id}/pending_deployments")
    result = []
    for item in items if isinstance(items, list) else []:
        reviewers = []
        for entry in item.get("reviewers", []):
            reviewer = entry.get("reviewer") or {}
            reviewers.append(str(reviewer.get("login") or reviewer.get("slug") or ""))
        result.append(
            {
                "id": item["environment"]["id"],
                "name": item["environment"].get("name"),
                "can_approve": bool(item.get("current_user_can_approve")),
                "reviewers": [name for name in reviewers if name][:20],
            }
        )
    return result


@router.get("/api/catalog/reviews/run/{run_id}")
def run_review(run_id: int, request: Request, user: dict = Depends(catalog.catalog_user)):
    if run_id < 1:
        raise HTTPException(status_code=404, detail="Run not found")
    with catalog.user_github(request, user) as github:
        run = load_run(github, run_id)
        environments = pending(github, run_id) if run.get("status") == "waiting" else []
        names = display_names(github, [run.get("actor", {}).get("login")])
    return {
        "id": run_id,
        "workflow": str(run.get("path", "")).rsplit("/", 1)[-1],
        "run_number": run.get("run_number"),
        "title": str(run.get("display_title", ""))[:200],
        "status": run.get("status"),
        "conclusion": run.get("conclusion"),
        "html_url": run.get("html_url"),
        "actor": run.get("actor", {}).get("login"),
        "actor_name": names.get(str(run.get("actor", {}).get("login")), ""),
        "environments": environments,
    }


@router.post("/api/catalog/reviews/run/{run_id}/deployments")
def review_deployments(
    run_id: int,
    payload: DeploymentAction,
    request: Request,
    user: dict = Depends(catalog.catalog_user),
):
    with catalog.user_github(request, user) as github:
        login = str(github.call("GET", "/user").get("login", ""))
        load_run(github, run_id)
        approvable = {item["id"] for item in pending(github, run_id) if item["can_approve"]}
        if not set(payload.environment_ids) <= approvable:
            raise HTTPException(
                status_code=403,
                detail="You are not a required reviewer for that environment, or it is no "
                "longer waiting",
            )
        verb = "Approved" if payload.state == "approved" else "Rejected"
        try:
            with audited(
                request,
                user,
                login,
                "catalog_deployment_review",
                f"run/{run_id}",
                decision=payload.state,
                environment_ids=sorted(set(payload.environment_ids)),
            ):
                github.call(
                    "POST",
                    f"{catalog.BASE}/actions/runs/{run_id}/pending_deployments",
                    body={
                        "environment_ids": sorted(set(payload.environment_ids)),
                        "state": payload.state,
                        "comment": payload.comment.strip() or f"{verb} in CloudOps Hub",
                    },
                )
        except catalog.GitHubRefused as error:
            refused(error)
    logger.info(
        "Catalog run %s deployment %s by %s", int(run_id), log_safe(payload.state), log_safe(login)
    )
    return {"status": payload.state}
