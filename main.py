import logging
import os
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any, Literal

from azure.core.exceptions import AzureError, ResourceModifiedError
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.sessions import SessionMiddleware
from starlette.responses import JSONResponse

from cloudops_hub import announcements, answer_tests, firewall_audit
from cloudops_hub.answer_index import AnswerIndex
from cloudops_hub.assistants import ASSISTANTS
from cloudops_hub.auth import (
    ADMIN_ROLE,
    azure_user,
    can_use_assistant,
    configure_oauth,
    current_user,
    firewall_ab_identity,
    require_assistant,
    require_csrf,
)
from cloudops_hub.auth import router as auth_router
from cloudops_hub.azure_audit import begin, finish, run_audited
from cloudops_hub.catalog import sources as catalog_sources
from cloudops_hub.catalog_assistant import router as catalog_assistant_router
from cloudops_hub.catalog_review import router as catalog_review_router
from cloudops_hub.catalog_tracking import router as catalog_tracking_router
from cloudops_hub.config import get_settings
from cloudops_hub.conversations import ConversationStore
from cloudops_hub.drafts import (
    ESCALATION,
    TEAM_PATTERN,
    catalog_chat,
    catalog_guidance_paths,
    landing_zone_json_redirect,
    landing_zone_json_request,
    parameter_explanation_context,
    service_catalog_redirect,
)
from cloudops_hub.feedback import record_feedback, record_retrieval
from cloudops_hub.firewall_json import firewall_json_chat
from cloudops_hub.glossary_review import (
    cache_logged_keyword_scan,
    clear_pending_keywords,
    dismiss_keyword,
    export_glossary_review,
    keyword_guide_coverage,
    read_cached_keyword_scan,
    save_firewall_starters,
    save_logged_keywords,
)
from cloudops_hub.hub_audit import HubAuditMiddleware
from cloudops_hub.imports import ImportBusy, control_import, control_keyword_scan
from cloudops_hub.knowledge import (
    GLOSSARY_TERM_PATTERN,
    GLOSSARY_WARNING_HEADER,
    GlossaryStatusMiddleware,
    edit_glossary_entry,
    read_glossary,
    save_glossary_alias,
)
from cloudops_hub.language import LANGUAGE_NOTE, conversation_language, localize_answer
from cloudops_hub.lookups import lookup_arm, lookup_graph
from cloudops_hub.model_usage import recent_history
from cloudops_hub.operations_store import (
    FirewallAuditUnavailable,
    OperationsStore,
    OperationsUnavailable,
)
from cloudops_hub.prompt_guard import BLOCKED_ANSWER, is_prompt_attack
from cloudops_hub.question_log import (
    GAP_STATUSES,
    RUN_PREFIX,
    QuestionLog,
    is_short_reply,
    markdown_answer,
    mentions_contact_cloudops,
)
from cloudops_hub.retrieval import ModelRateLimited, answer_question, configured
from cloudops_hub.schemas import (
    SchemaNotFound,
    list_schemas,
    load_schema,
    required_fields,
    validate_payload,
)
from cloudops_hub.service_catalog import CatalogError
from cloudops_hub.service_catalog import router as catalog_router
from cloudops_hub.terraform_import import router as terraform_import_router
from cloudops_hub.terraform_state import router as terraform_state_router
from cloudops_hub.troubleshooting import match_error

logging.basicConfig(level=logging.INFO)
for http_logger in ("httpx", "httpcore"):
    logging.getLogger(http_logger).setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
if os.environ.get("APPLICATIONINSIGHTS_CONNECTION_STRING"):
    from azure.monitor.opentelemetry import configure_azure_monitor

    configure_azure_monitor()
settings = get_settings()
configure_oauth(settings)
# Set only by the admin answer-test endpoint: temporary chat, no history or question logging.
_answer_test: ContextVar[bool] = ContextVar("answer_test", default=False)


@asynccontextmanager
async def lifespan(application: FastAPI):
    import asyncio

    def sync_answers() -> None:
        try:
            entries = application.state.questions.entries(limit=None)
        except Exception as exc:  # noqa: BLE001 - storage SDK errors vary by transport
            logger.warning("Answer index startup sync skipped (%s)", type(exc).__name__)
            return
        application.state.answer_index.sync_all(entries)

    if application.state.answer_index.enabled:
        application.state.answer_sync_task = asyncio.create_task(asyncio.to_thread(sync_answers))
    yield
    application.state.conversations.close()
    application.state.questions.close()
    application.state.operations.close()


app = FastAPI(
    title="CloudOps Hub API",
    version="0.1.0",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    lifespan=lifespan,
)
app.state.settings = settings
app.state.conversations = ConversationStore(settings)
app.state.questions = QuestionLog(settings)
app.state.answer_index = AnswerIndex(settings)
app.state.operations = OperationsStore(settings)
# Added before SessionMiddleware so it runs inside it and can read the session user.
app.add_middleware(HubAuditMiddleware)
app.add_middleware(GlossaryStatusMiddleware)
app.add_middleware(
    SessionMiddleware,
    secret_key=settings.session_secret or "development-only-change-me",
    session_cookie="cloudops_session",
    same_site="lax",
    https_only=settings.session_https_only,
    max_age=settings.session_hours * 3600,
)
if settings.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH", "DELETE"],
        allow_headers=["Content-Type", "X-CSRF-Token"],
        expose_headers=[GLOSSARY_WARNING_HEADER],
    )
app.include_router(auth_router)
app.include_router(catalog_router)
app.include_router(catalog_assistant_router)
app.include_router(catalog_review_router)
app.include_router(catalog_tracking_router)
app.include_router(terraform_state_router)
app.include_router(terraform_import_router)


@app.exception_handler(CatalogError)
@app.exception_handler(OperationsUnavailable)
async def catalog_error_handler(request: Request, exc: CatalogError):
    logger.warning("Service Catalog request failed")
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(AzureError)
async def azure_storage_error_handler(request: Request, exc: AzureError):
    logger.error("Operational Azure access failed (%s)", type(exc).__name__)
    return JSONResponse(
        status_code=503,
        content={
            "detail": "Azure operational storage/access failed. Contact CloudOps; "
            "refresh the request before retrying any submission.",
        },
    )


_request_times: dict[str, deque[float]] = defaultdict(deque)


class RequestProtectionMiddleware:
    def __init__(self, app, max_request_bytes: int):
        self.app = app
        self.max_request_bytes = max_request_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        content_length = next(
            (
                value
                for name, value in scope.get("headers", [])
                if name.lower() == b"content-length"
            ),
            b"",
        )
        if content_length.isdigit() and int(content_length) > self.max_request_bytes:
            await self._too_large(scope, receive, send)
            return

        messages = []
        total_size = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            if message["type"] != "http.request":
                continue
            total_size += len(message.get("body", b""))
            if total_size > self.max_request_bytes:
                await self._too_large(scope, receive, send)
                return
            messages.append(message)
            if not message.get("more_body", False):
                break

        async def replay():
            if messages:
                return messages.pop(0)
            return {"type": "http.request", "body": b"", "more_body": False}

        async def secure_send(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.extend(
                    [
                        (b"x-content-type-options", b"nosniff"),
                        (b"x-frame-options", b"DENY"),
                        (b"referrer-policy", b"no-referrer"),
                        (b"cache-control", b"no-store"),
                    ]
                )
                message["headers"] = headers
            await send(message)

        await self.app(scope, replay, secure_send)

    async def _too_large(self, scope, receive, send):
        from starlette.responses import JSONResponse

        response = JSONResponse(status_code=413, content={"detail": "Request body is too large"})
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        await response(scope, receive, send)


app.add_middleware(
    RequestProtectionMiddleware,
    max_request_bytes=settings.max_request_bytes,
)


class ValidateRequest(BaseModel):
    schema_id: str
    payload: Any


class GenerateRequest(BaseModel):
    schema_id: str
    values: dict[str, Any]


class MatchRequest(BaseModel):
    message: str = Field(min_length=1, max_length=20_000)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=10_000)
    conversation_id: str | None = None
    schema_id: str | None = None
    catalog_team: str | None = Field(default=None, max_length=40)


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    helpful: bool = Field(strict=True)


class QuestionReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["new", "answered", "dismissed"]
    answer: str = Field(default="", max_length=20_000)
    markdown: bool = True


class RefreshRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: Literal[
        "confluence", "confluence-images", "catalog", "knowledge", "glossary", "graph", "all"
    ]


class GlossaryAliasRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    keyword: str = Field(min_length=1, max_length=100, pattern=GLOSSARY_TERM_PATTERN)
    canonical: str = Field(min_length=1, max_length=100, pattern=GLOSSARY_TERM_PATTERN)
    etag: str = Field(min_length=1, max_length=200)
    scope: Literal["other-agents", "firewall"] = "other-agents"


class GlossaryEditRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    kind: Literal["groups", "aliases"]
    index: int = Field(ge=0, le=999)
    canonical: str = Field(min_length=1, max_length=100, pattern=GLOSSARY_TERM_PATTERN)
    terms: list[str] = Field(min_length=1, max_length=15)
    etag: str = Field(min_length=1, max_length=200)
    scope: Literal["other-agents", "firewall"] | None = None


class GlossaryDismissRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    keyword_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    etag: str = Field(min_length=1, max_length=200)


class GlossaryStarterRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    etag: str = Field(min_length=1, max_length=200)


class GlossaryEntryReference(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["groups", "aliases"]
    index: int = Field(ge=0, le=999)


class LoggedKeywordScanSuggestion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    keyword: str = Field(min_length=1, max_length=100, pattern=GLOSSARY_TERM_PATTERN)
    question: str = Field(min_length=1, max_length=10_000)
    assistant_id: Literal["landing-zone", "firewall", "service-catalog", "platform-admin"]


class LoggedKeywordScanCacheRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    suggestions: list[LoggedKeywordScanSuggestion] = Field(max_length=10_000)
    scanned: int = Field(ge=0, le=1_000_000)
    total: int = Field(ge=0, le=1_000_000)


class GlossaryCoverageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    keyword: str = Field(min_length=1, max_length=100, pattern=GLOSSARY_TERM_PATTERN)
    assistants: list[Literal["landing-zone", "firewall", "service-catalog", "platform-admin"]] = (
        Field(min_length=1, max_length=4)
    )


class LoggedKeywordSuggestion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    keyword: str = Field(min_length=1, max_length=100, pattern=GLOSSARY_TERM_PATTERN)


class LoggedKeywordSaveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    suggestions: list[LoggedKeywordSuggestion] = Field(min_length=1, max_length=200)
    etag: str = Field(min_length=1, max_length=200)


class GlossaryMergeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    entries: list[GlossaryEntryReference] = Field(min_length=2, max_length=20)
    keep: GlossaryEntryReference
    etag: str = Field(min_length=1, max_length=200)


class AzureLookupRequest(BaseModel):
    resource_id: str = Field(min_length=1, max_length=2048)


class GraphLookupRequest(BaseModel):
    object_type: str
    object_id: str


def api_user(request: Request) -> dict[str, Any]:
    user = current_user(request)
    if request.method not in {"GET", "HEAD", "OPTIONS"}:
        require_csrf(request)
    return user


def admin_user(user: dict = Depends(api_user)) -> dict:
    if ADMIN_ROLE not in user.get("roles", []):
        raise HTTPException(status_code=403, detail="CloudOps Hub Admin role is required")
    return user


@app.get("/api/catalog/graph-permissions")
def graph_permissions(request: Request, user: dict = Depends(api_user)) -> dict:
    from cloudops_hub.graph_permissions import RECORD

    return {"snapshot": request.app.state.operations.read(RECORD)}


@app.post("/api/admin/graph-permissions/refresh")
def refresh_graph_permissions(request: Request, user: dict = Depends(admin_user)) -> dict:
    from cloudops_hub.graph_permissions import refresh_permissions

    return {"snapshot": refresh_permissions(request.app.state.operations)}


def check_chat_rate(user_id: str) -> None:
    now = time.monotonic()
    timestamps = _request_times[user_id]
    while timestamps and now - timestamps[0] >= 60:
        timestamps.popleft()
    if len(timestamps) >= settings.chat_rate_limit_per_minute:
        raise HTTPException(status_code=429, detail="Chat rate limit exceeded")
    timestamps.append(now)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/session")
async def session_info(request: Request) -> dict[str, Any]:
    user = current_user(request)
    try:
        azp = azure_user(request)
    except HTTPException:
        azp = None
    return {
        "user": {
            "oid": user["oid"],
            "name": user.get("name"),
            "email": user.get("email"),
            "is_admin": ADMIN_ROLE in user.get("roles", []),
        },
        "azure": {
            "connected": azp is not None,
            "email": azp.get("email") if azp else None,
            "expires_at": azp.get("expires_at") if azp else None,
        },
        "firewall": {"ab_access": firewall_ab_identity(request) is not None},
        "csrf_token": request.session["csrf"],
    }


@app.get("/api/admin/azure-audit")
async def azure_audit_events(request: Request, _: dict = Depends(admin_user)):
    from datetime import timedelta

    # Bounded date prefixes; do not scan token or request records.
    records = []
    for offset in range(7):
        day = (datetime.now(UTC) - timedelta(days=offset)).strftime("%Y-%m-%d")
        records.extend(request.app.state.operations.list(f"audit/{day}/", limit=500))
    return sorted(records, key=lambda event: event["timestamp"], reverse=True)[:500]


@app.get("/api/announcements")
async def announcement_feed(_: dict = Depends(api_user)) -> dict[str, Any]:
    import asyncio

    def load():
        from azure.identity import DefaultAzureCredential

        with DefaultAzureCredential() as credential:
            return announcements.read(settings, credential)

    try:
        return await asyncio.to_thread(load)
    except (AzureError, RuntimeError):
        logger.error("Published announcements could not be read")
        raise HTTPException(
            status_code=503, detail="Announcements are unavailable; try again later."
        ) from None


@app.get("/api/admin/imports")
async def import_status(_: dict = Depends(admin_user)) -> dict[str, Any]:
    import asyncio

    try:
        return await asyncio.to_thread(control_import, settings)
    except RuntimeError as exc:
        logger.error("Import status unavailable: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from None


@app.post("/api/admin/imports", status_code=202)
async def refresh_sources(
    payload: RefreshRequest, user: dict = Depends(admin_user)
) -> dict[str, Any]:
    import asyncio

    try:
        result = await asyncio.to_thread(control_import, settings, payload.source)
    except ImportBusy as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None
    except RuntimeError as exc:
        logger.error("Import start unavailable: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from None
    logger.info(
        "Administrator %s started cache refresh %s",
        str(user["oid"]).replace("\r", "").replace("\n", ""),
        str(result["execution"]).replace("\r", "").replace("\n", ""),
    )
    return result


@app.get("/api/admin/questions")
async def question_dashboard(_: dict = Depends(admin_user)) -> dict[str, Any]:
    import asyncio

    try:
        return await asyncio.to_thread(app.state.questions.dashboard)
    except Exception as exc:  # noqa: BLE001 - storage SDK errors vary by transport
        logger.error("Question dashboard unavailable (%s)", type(exc).__name__)
        raise HTTPException(status_code=503, detail="Question insights are unavailable") from None


@app.get("/api/admin/glossary")
async def glossary_status(_: dict = Depends(admin_user)) -> dict[str, Any]:
    import asyncio

    try:
        return await asyncio.to_thread(read_glossary, settings)
    except (AzureError, ValueError, UnicodeError) as exc:
        logger.warning("Administrator glossary read unavailable (%s)", type(exc).__name__)
        raise HTTPException(
            status_code=503, detail="Live glossary could not be read. No changes were made."
        ) from None


@app.get("/api/admin/glossary/review-export")
async def glossary_review_export(_: dict = Depends(admin_user)) -> JSONResponse:
    import asyncio

    try:
        package = await asyncio.to_thread(
            export_glossary_review, settings, app.state.questions,
        )
    except (AzureError, ValueError, RuntimeError) as exc:
        logger.warning("Glossary review export unavailable (%s)", type(exc).__name__)
        raise HTTPException(
            status_code=503, detail="Glossary review JSON could not be generated. Retry download."
        ) from None
    return JSONResponse(package, headers={
        "Content-Disposition": 'attachment; filename="glossary-review.json"',
    })


@app.post("/api/admin/glossary")
async def glossary_save(
    payload: GlossaryAliasRequest, _: dict = Depends(admin_user)
) -> dict[str, Any]:
    return await _glossary_write(
        save_glossary_alias, settings, payload.keyword, payload.canonical, payload.etag,
        payload.scope,
    )


@app.post("/api/admin/glossary/dismiss")
async def glossary_dismiss(
    payload: GlossaryDismissRequest, _: dict = Depends(admin_user)
) -> dict[str, Any]:
    return await _glossary_write(dismiss_keyword, settings, payload.keyword_id, payload.etag)


@app.post("/api/admin/glossary/firewall-starters")
async def glossary_firewall_starters(
    payload: GlossaryStarterRequest, _: dict = Depends(admin_user)
) -> dict[str, Any]:
    return await _glossary_write(save_firewall_starters, settings, payload.etag)


@app.post("/api/admin/glossary/clear-review")
async def glossary_clear_review(
    payload: GlossaryStarterRequest, _: dict = Depends(admin_user)
) -> dict[str, Any]:
    return await _glossary_write(clear_pending_keywords, settings, payload.etag)


@app.get("/api/admin/glossary/keyword-scan")
async def glossary_keyword_scan_status(_: dict = Depends(admin_user)) -> dict[str, Any]:
    import asyncio

    try:
        return await asyncio.to_thread(control_keyword_scan, settings)
    except RuntimeError as exc:
        logger.error("Keyword scan status unavailable: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from None


@app.post("/api/admin/glossary/keyword-scan", status_code=202)
async def glossary_keyword_scan_start(user: dict = Depends(admin_user)) -> dict[str, Any]:
    import asyncio

    try:
        result = await asyncio.to_thread(control_keyword_scan, settings, True)
    except ImportBusy as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None
    except RuntimeError as exc:
        logger.error("Keyword scan start unavailable: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from None
    logger.info(
        "Administrator %s started keyword scan %s",
        str(user["oid"]).replace("\r", "").replace("\n", ""),
        str(result["execution"]).replace("\r", "").replace("\n", ""),
    )
    return result


@app.get("/api/admin/glossary/scan-cache")
async def glossary_scan_cache(_: dict = Depends(admin_user)) -> dict[str, Any]:
    import asyncio

    try:
        return await asyncio.to_thread(
            read_cached_keyword_scan, settings, app.state.questions,
        )
    except (AzureError, RuntimeError, ValueError, UnicodeError) as exc:
        logger.warning("Glossary scan cache could not be loaded (%s)", type(exc).__name__)
        raise HTTPException(
            status_code=503, detail="Saved keyword suggestions could not be loaded. Retry refresh.",
        ) from None


@app.post("/api/admin/glossary/scan-cache")
async def glossary_save_scan_cache(
    payload: LoggedKeywordScanCacheRequest, _: dict = Depends(admin_user)
) -> dict[str, Any]:
    import asyncio

    try:
        return await asyncio.to_thread(
            cache_logged_keyword_scan, settings, app.state.questions,
            [suggestion.model_dump() for suggestion in payload.suggestions],
            payload.scanned, payload.total,
        )
    except (AzureError, RuntimeError, ValueError, UnicodeError) as exc:
        logger.warning("Glossary scan cache could not be saved (%s)", type(exc).__name__)
        raise HTTPException(
            status_code=503,
            detail="Keyword scan completed, but its results could not be saved. "
            "The current results are still available until you leave this page.",
        ) from None


@app.post("/api/admin/glossary/guide-coverage")
async def glossary_guide_coverage(
    payload: GlossaryCoverageRequest, _: dict = Depends(admin_user)
) -> dict[str, Any]:
    import asyncio

    try:
        return await asyncio.to_thread(
            keyword_guide_coverage, settings, payload.keyword, payload.assistants,
        )
    except (AzureError, RuntimeError, ValueError) as exc:
        logger.warning("Glossary guide coverage unavailable (%s)", type(exc).__name__)
        raise HTTPException(
            status_code=503, detail="Guide coverage could not be checked. Try again.",
        ) from None


@app.post("/api/admin/glossary/logged-keywords")
async def glossary_save_logged_keywords(
    payload: LoggedKeywordSaveRequest, _: dict = Depends(admin_user)
) -> dict[str, Any]:
    return await _glossary_write(
        save_logged_keywords, settings, app.state.questions,
        [suggestion.model_dump() for suggestion in payload.suggestions], payload.etag,
    )


@app.patch("/api/admin/glossary")
async def glossary_edit(
    payload: GlossaryEditRequest, _: dict = Depends(admin_user)
) -> dict[str, Any]:
    return await _glossary_write(
        edit_glossary_entry, settings, payload.kind, payload.index,
        payload.canonical, payload.terms, payload.etag,
        payload.scope,
    )


@app.post("/api/admin/glossary/merge")
async def glossary_merge(
    payload: GlossaryMergeRequest, _: dict = Depends(admin_user)
) -> dict[str, Any]:
    from cloudops_hub.knowledge import merge_glossary_entries

    return await _glossary_write(
        merge_glossary_entries, settings, [entry.model_dump() for entry in payload.entries],
        payload.keep.model_dump(), payload.etag,
    )


async def _glossary_write(operation, *args) -> dict[str, Any]:
    import asyncio

    try:
        return await asyncio.to_thread(operation, *args)
    except ResourceModifiedError:
        raise HTTPException(
            status_code=409, detail="Another administrator updated the glossary. Reload and retry."
        ) from None
    except (ValueError, UnicodeError) as exc:
        logger.warning("Administrator glossary validation failed (%s)", type(exc).__name__)
        raise HTTPException(
            status_code=400, detail="Glossary validation failed. Check term and dictionary limits."
        ) from None
    except AzureError as exc:
        logger.warning("Administrator glossary write unavailable (%s)", type(exc).__name__)
        raise HTTPException(
            status_code=503,
            detail="Glossary save could not be confirmed. "
            "Reload the live glossary before retrying.",
        ) from None


@app.patch("/api/admin/questions/{question_id}")
async def review_question(
    question_id: str, payload: QuestionReviewRequest, _: dict = Depends(admin_user)
) -> dict[str, Any]:
    import asyncio

    answer = payload.answer
    if payload.markdown and payload.status == "answered" and answer.strip():
        answer = await asyncio.to_thread(markdown_answer, settings, answer)
    try:
        entry = await asyncio.to_thread(
            app.state.questions.update, question_id, payload.status, answer
        )
    except ValueError:
        raise HTTPException(status_code=404, detail="Question not found") from None
    if entry is None:
        raise HTTPException(status_code=404, detail="Question not found")
    indexed = (
        await asyncio.to_thread(app.state.answer_index.sync, entry)
        if app.state.answer_index.enabled
        else None
    )
    return {**entry, "indexed": indexed}


class MarkdownSuggestionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer: str = Field(min_length=1, max_length=20_000)


@app.post("/api/admin/answers/markdown")
async def suggest_markdown(
    payload: MarkdownSuggestionRequest, _: dict = Depends(admin_user)
) -> dict[str, Any]:
    import asyncio

    markdown = await asyncio.to_thread(markdown_answer, settings, payload.answer)
    return {"markdown": markdown, "changed": markdown != payload.answer}


@app.delete("/api/admin/questions/{question_id}")
async def delete_question(question_id: str, _: dict = Depends(admin_user)) -> dict[str, bool]:
    import asyncio

    try:
        deleted = await asyncio.to_thread(app.state.questions.delete, question_id)
    except ValueError:
        deleted = False
    if not deleted:
        raise HTTPException(status_code=404, detail="Question not found")
    await asyncio.to_thread(app.state.answer_index.remove, question_id)
    return {"ok": True}


@app.patch("/api/admin/runs/{run_id}")
async def review_run(
    run_id: str, payload: QuestionReviewRequest, _: dict = Depends(admin_user)
) -> dict[str, Any]:
    import asyncio

    try:
        entry = await asyncio.to_thread(
            app.state.questions.update, run_id, payload.status, payload.answer, RUN_PREFIX
        )
    except ValueError:
        raise HTTPException(status_code=404, detail="Run not found") from None
    if entry is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return entry


@app.delete("/api/admin/runs/{run_id}")
async def delete_run(run_id: str, _: dict = Depends(admin_user)) -> dict[str, bool]:
    import asyncio

    try:
        deleted = await asyncio.to_thread(app.state.questions.delete, run_id, RUN_PREFIX)
    except ValueError:
        deleted = False
    if not deleted:
        raise HTTPException(status_code=404, detail="Run not found")
    return {"ok": True}


@app.get("/api/assistants")
async def assistants(user: dict = Depends(api_user)) -> list[dict[str, Any]]:
    return [
        {
            "id": assistant.id,
            "title": assistant.title,
            "description": assistant.description,
            "suggested_questions": list(assistant.suggested_questions),
            "capabilities": [
                {"title": item.title, "description": item.description}
                for item in assistant.capabilities
            ],
        }
        for assistant in ASSISTANTS.values()
        if can_use_assistant(user, assistant.id)
    ]


@app.get("/api/schemas")
async def schemas(_: dict = Depends(api_user)) -> list[dict[str, str]]:
    return list_schemas()


@app.get("/api/catalog/sources")
async def approved_catalog_sources(_: dict = Depends(api_user)) -> dict[str, Any]:
    return {
        "repositories": catalog_sources(),
        "output_repository": "almbrand/landingzone-service-catalog",
        "escalation": ESCALATION,
        "mode": "read-only",
    }


@app.get("/api/schemas/{schema_id}")
async def schema_detail(schema_id: str, _: dict = Depends(api_user)) -> dict[str, Any]:
    try:
        schema, source = load_schema(schema_id)
    except SchemaNotFound as exc:
        raise HTTPException(
            status_code=404, detail="No approved schema is available for that request type"
        ) from exc
    return {"schema": schema, "required": required_fields(schema), "source": source}


@app.post("/api/schemas/validate")
async def schema_validate(payload: ValidateRequest, _: dict = Depends(api_user)) -> dict[str, Any]:
    try:
        result = validate_payload(payload.schema_id, payload.payload)
    except SchemaNotFound as exc:
        raise HTTPException(
            status_code=404, detail="No approved schema is available for that request type"
        ) from exc
    result["validated_at"] = datetime.now(UTC).isoformat()
    return result


@app.post("/api/schemas/generate")
async def schema_generate(payload: GenerateRequest, _: dict = Depends(api_user)) -> dict[str, Any]:
    try:
        schema, source = load_schema(payload.schema_id)
    except SchemaNotFound as exc:
        raise HTTPException(
            status_code=404, detail="No approved schema is available for that request type"
        ) from exc
    if schema.get("x-target-repository") != "almbrand/landingzone-service-catalog":
        raise HTTPException(status_code=400, detail="Only service-catalog JSON can be generated")
    missing = [
        field
        for field in required_fields(schema)
        if field not in payload.values or payload.values[field] in (None, "")
    ]
    questions = []
    for field in missing:
        definition = schema.get("properties", {}).get(field, {})
        label = definition.get("title", field.replace("_", " "))
        prompt = definition.get("description")
        questions.append(prompt or f"Please provide {label}.")
    if missing:
        return {
            "status": "missing_parameters",
            "missing": missing,
            "questions": questions,
            "source": source,
        }
    validation = validate_payload(payload.schema_id, payload.values)
    return {
        "status": "generated" if validation["valid"] else "invalid",
        "missing": [],
        "questions": [],
        "json": payload.values if validation["valid"] else None,
        "errors": validation["errors"],
        "source": source,
    }


@app.post("/api/errors/match")
async def error_match(payload: MatchRequest, _: dict = Depends(api_user)) -> dict[str, Any]:
    matches = match_error(payload.message)
    return {
        "status": "matched" if matches else "unknown",
        "matches": matches,
        "message": None
        if matches
        else "No documented match was found; investigate using the escalation route.",
    }


@app.post("/api/lookups/azure")
async def azure_lookup(
    payload: AzureLookupRequest, request: Request, user: dict = Depends(api_user)
) -> dict[str, Any]:
    from cloudops_hub.azure_audit import audit_scope

    azp = azure_user(request)
    with audit_scope(request.app.state.operations, settings, user, azp):
        event = begin("azure_resource_lookup")
        try:
            result = lookup_arm(payload.resource_id, settings)
        except Exception:
            finish(event, "failed")
            raise
        finish(event, result.get("status", "unknown"))
        return result


@app.post("/api/lookups/entra")
async def entra_lookup(
    payload: GraphLookupRequest, request: Request, user: dict = Depends(api_user)
) -> dict[str, Any]:
    from cloudops_hub.azure_audit import audit_scope

    azp = azure_user(request)
    with audit_scope(request.app.state.operations, settings, user, azp):
        event = begin("entra_object_lookup")
        try:
            result = await lookup_graph(payload.object_type, payload.object_id, settings)
        except Exception:
            finish(event, "failed")
            raise
        finish(event, result.get("status", "unknown"))
        return result


@app.get("/api/conversations")
async def conversations(request: Request, user: dict = Depends(api_user)) -> list[dict[str, Any]]:
    records = app.state.conversations.list(user["oid"])
    records = [
        record
        for record in records
        if not has_platform_history(record) or can_use_assistant(user, "platform-admin")
    ]
    try:
        firewall_user(request)
    except HTTPException:
        records = [record for record in records if not has_firewall_history(record)]
    from cloudops_hub.conversations import conversation_view

    return [
        conversation_view(record, admin=ADMIN_ROLE in user.get("roles", [])) for record in records
    ]


@app.get("/api/conversations/{conversation_id}")
async def conversation_detail(
    conversation_id: str, request: Request, user: dict = Depends(api_user)
) -> dict[str, Any]:
    result = app.state.conversations.get(user["oid"], conversation_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    require_conversation(user, result)
    if has_firewall_history(result):
        firewall_user(request)
    from cloudops_hub.conversations import conversation_view

    return conversation_view(result, admin=ADMIN_ROLE in user.get("roles", []))


def firewall_user(request: Request) -> dict:
    return firewall_ab_identity(request) or azure_user(request)


def has_firewall_history(conversation: dict) -> bool:
    return conversation.get("assistant_id") == "firewall" or any(
        message.get("assistant_id") == "firewall" for message in conversation.get("messages", [])
    )


def has_platform_history(conversation: dict) -> bool:
    return conversation.get("assistant_id") == "platform-admin" or any(
        message.get("assistant_id") == "platform-admin"
        for message in conversation.get("messages", [])
    )


def require_conversation(user: dict, conversation: dict) -> None:
    if has_platform_history(conversation):
        require_assistant(user, "platform-admin")


@app.delete("/api/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    request: Request,
    user: dict = Depends(api_user),
) -> dict[str, bool]:
    conversation = request.app.state.conversations.get(user["oid"], conversation_id)
    if conversation is not None:
        require_conversation(user, conversation)
    if not request.app.state.conversations.delete(user["oid"], conversation_id):
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"ok": True}


@app.post("/api/conversations/{conversation_id}/messages/{message_id}/feedback")
async def submit_feedback(
    conversation_id: str,
    message_id: str,
    payload: FeedbackRequest,
    request: Request,
    background: BackgroundTasks,
    user: dict = Depends(api_user),
) -> dict[str, bool]:
    store = request.app.state.conversations
    conversation = store.get(user["oid"], conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    require_conversation(user, conversation)
    if has_firewall_history(conversation):
        firewall_user(request)
    message_index = next(
        (
            index
            for index, item in enumerate(conversation["messages"])
            if item.get("id") == message_id and item.get("role") == "assistant"
        ),
        None,
    )
    if message_index is None:
        raise HTTPException(status_code=404, detail="Answer not found")
    message = conversation["messages"][message_index]
    question = next(
        (
            item["content"]
            for item in reversed(conversation["messages"][:message_index])
            if item.get("role") == "user" and isinstance(item.get("content"), str)
        ),
        None,
    )
    message["feedback"] = {
        "helpful": payload.helpful,
        "updated_at": datetime.now(UTC).isoformat(),
    }
    store.save(user["oid"], conversation)
    record_feedback(message_id, payload.helpful)
    if settings.question_log_enabled and not has_platform_history(conversation):
        background.add_task(
            request.app.state.questions.record_feedback,
            message_id,
            payload.helpful,
            message.get("assistant_id"),
            question,
        )
    return {"saved": True}


class AnswerTestRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    assistant_id: Literal["landing-zone", "firewall", "platform-admin"]
    question: str = Field(min_length=1, max_length=2_000)


class AnswerTestCaseRequest(AnswerTestRunRequest):
    expect: str = Field(default="", max_length=300)


@app.get("/api/admin/answer-tests")
async def answer_test_cases(request: Request, _: dict = Depends(admin_user)) -> dict[str, Any]:
    import asyncio

    try:
        custom = await asyncio.to_thread(answer_tests.custom_cases, request.app.state.operations)
        custom_error = ""
    except (OperationsUnavailable, AzureError):
        custom, custom_error = [], "Custom questions are unavailable: operations storage failed"
    return {
        "cases": [*answer_tests.DEFAULT_CASES, *custom],
        "custom_error": custom_error,
    }


@app.post("/api/admin/answer-tests/custom")
async def add_answer_test_case(
    payload: AnswerTestCaseRequest, request: Request, _: dict = Depends(admin_user)
) -> dict[str, Any]:
    import asyncio

    try:
        cases = await asyncio.to_thread(
            answer_tests.add_custom_case, request.app.state.operations,
            payload.assistant_id, payload.question.strip(), payload.expect.strip(),
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None
    except (OperationsUnavailable, AzureError):
        raise HTTPException(status_code=503, detail="Custom questions could not be saved") from None
    return {"custom": cases}


@app.delete("/api/admin/answer-tests/custom/{case_id}")
async def delete_answer_test_case(
    case_id: str, request: Request, _: dict = Depends(admin_user)
) -> dict[str, Any]:
    import asyncio

    try:
        cases = await asyncio.to_thread(
            answer_tests.delete_custom_case, request.app.state.operations, case_id
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="Custom question not found") from None
    except (OperationsUnavailable, AzureError):
        raise HTTPException(
            status_code=503, detail="Custom question could not be deleted"
        ) from None
    return {"custom": cases}


@app.post("/api/admin/answer-tests/run")
async def run_answer_test(
    payload: AnswerTestRunRequest,
    request: Request,
    background: BackgroundTasks,
    user: dict = Depends(admin_user),
) -> dict[str, Any]:
    """Run one question through the real chat path without saving history or logs."""
    started = time.monotonic()
    token = _answer_test.set(True)
    try:
        result = await chat(
            payload.assistant_id, ChatRequest(question=payload.question), request, background, user
        )
    finally:
        _answer_test.reset(token)
    request.app.state.conversations.delete(user["oid"], result["conversation_id"])
    return {
        key: result[key]
        for key in ("status", "answer", "sources", "confidence", "live_check")
        if key in result
    } | {"duration_ms": round((time.monotonic() - started) * 1000)}


@app.post("/api/chat/{assistant_id}")
async def chat(
    assistant_id: str,
    payload: ChatRequest,
    request: Request,
    background: BackgroundTasks,
    user: dict = Depends(api_user),
) -> dict[str, Any]:
    sensitive = assistant_id == "firewall"
    if payload.conversation_id:
        saved = request.app.state.conversations.get(user["oid"], payload.conversation_id)
        sensitive = sensitive or bool(saved and has_firewall_history(saved))
    if sensitive:
        import asyncio

        from cloudops_hub.firewall_model_data import firewall_boundary

        require_assistant(user, assistant_id)
        azp = firewall_user(request)
        if payload.conversation_id:
            if saved is None:
                raise HTTPException(status_code=404, detail="Conversation not found")
            require_conversation(user, saved)
        if len(payload.question) > settings.chat_max_input_chars:
            raise HTTPException(
                status_code=413, detail="Question exceeds the configured input limit"
            )
        with firewall_boundary(), firewall_audit.activity_scope(
            request.app.state.operations, user, azp, payload.conversation_id,
            managed_identity_client_id=settings.azure_client_id,
        ):
            operation_id = uuid.uuid4().hex
            started = time.monotonic()
            try:
                await asyncio.to_thread(
                    firewall_audit.emit, "chat", "started",
                    operation_id=operation_id, question=payload.question,
                )
                try:
                    result = await _chat(assistant_id, payload, request, background, user)
                except FirewallAuditUnavailable:
                    raise
                except Exception as error:
                    await asyncio.to_thread(
                        firewall_audit.emit, "chat", "failed", operation_id=operation_id,
                        error_type=type(error).__name__,
                        http_status=error.status_code if isinstance(error, HTTPException) else 500,
                        duration_ms=round((time.monotonic() - started) * 1000),
                    )
                    raise
                await asyncio.to_thread(
                    firewall_audit.emit, "chat", "completed", operation_id=operation_id,
                    conversation_id=result["conversation_id"],
                    message_id=result["message_id"], status=result["status"],
                    duration_ms=round((time.monotonic() - started) * 1000),
                )
                return result
            except FirewallAuditUnavailable as error:
                raise HTTPException(status_code=503, detail=str(error)) from None
    from cloudops_hub.firewall_model_data import internal_boundary

    with internal_boundary(settings):
        return await _chat(assistant_id, payload, request, background, user)


async def _chat(
    assistant_id: str,
    payload: ChatRequest,
    request: Request,
    background: BackgroundTasks,
    user: dict,
) -> dict[str, Any]:
    assistant = ASSISTANTS.get(assistant_id)
    if assistant is None:
        raise HTTPException(status_code=404, detail="Assistant not found")
    require_assistant(user, assistant_id)
    if assistant_id == "service-catalog":
        raise HTTPException(
            status_code=400,
            detail="Use the Service Catalog form and confirmed submission endpoints",
        )
    azp = firewall_user(request) if assistant_id == "firewall" else None
    if len(payload.question) > settings.chat_max_input_chars:
        raise HTTPException(status_code=413, detail="Question exceeds the configured input limit")
    if payload.schema_id and (
        assistant_id != "landing-zone"
        or (payload.catalog_team and not TEAM_PATTERN.fullmatch(payload.catalog_team))
    ):
        raise HTTPException(
            status_code=400,
            detail="Catalog drafts require Landing Zone and a valid team folder when supplied",
        )
    if payload.schema_id:
        try:
            schema, _ = load_schema(payload.schema_id)
        except SchemaNotFound as exc:
            raise HTTPException(status_code=404, detail="Unsupported catalog request type") from exc
        if schema.get("x-target-repository") != "almbrand/landingzone-service-catalog":
            raise HTTPException(status_code=400, detail="Only service-catalog JSON can be drafted")
    check_chat_rate(user["oid"])
    store: ConversationStore = request.app.state.conversations
    if payload.conversation_id:
        conversation = store.get(user["oid"], payload.conversation_id)
        if conversation is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        require_conversation(user, conversation)
        if has_platform_history(conversation) != (assistant_id == "platform-admin"):
            raise HTTPException(
                status_code=409,
                detail="Start a new conversation when switching restricted platform knowledge",
            )
        if (conversation.get("temporary") or assistant_id == "firewall") and (
            conversation.get("assistant_id") != assistant_id
        ):
            raise HTTPException(
                status_code=409,
                detail=(
                    "Start a new chat when switching to or from the temporary Firewall assistant."
                ),
            )
        if has_firewall_history(conversation):
            firewall_user(request)
        # Users may switch agents within one conversation; the latest agent owns it.
        conversation["assistant_id"] = assistant_id
    else:
        conversation = (
            store.create_temporary(user["oid"], assistant_id)
            if _answer_test.get()
            else store.create(user["oid"], assistant_id)
        )

    firewall_sensitive = assistant_id == "firewall" or has_firewall_history(conversation)
    if firewall_sensitive:
        from cloudops_hub.firewall_model_data import EVIDENCE_VERSION, active_references

        references = active_references()
        if references is None:
            raise RuntimeError("Firewall endpoint boundary is missing")
        state = conversation.get("firewall_state") or {}
        if state.get("evidence_version") == EVIDENCE_VERSION:
            references.values.update(state.get("endpoint_references") or {})
        else:
            conversation["firewall_state"] = {}

    if is_prompt_attack(payload.question):
        logger.warning("Blocked prompt-attack pattern before model call: %s", assistant_id)
        message_id = uuid.uuid4().hex
        blocked = {"status": "blocked", "answer": BLOCKED_ANSWER, "sources": []}
        conversation["messages"].extend(
            [
                {"role": "user", "content": payload.question},
                {
                    "role": "assistant",
                    "id": message_id,
                    "assistant_id": assistant_id,
                    "content": BLOCKED_ANSWER,
                    "status": "blocked",
                    "sources": [],
                    "evidence_type": "documentation",
                },
            ]
        )
        conversation["updated_at"] = datetime.now(UTC).isoformat()
        store.save(user["oid"], conversation)
        return {**blocked, "conversation_id": conversation["id"], "message_id": message_id}

    history = [
        {"role": item["role"], "content": item["content"]}
        for item in conversation["messages"][-8:]
        if item.get("role") in {"user", "assistant"} and isinstance(item.get("content"), str)
    ]
    history = recent_history(settings, history)
    if firewall_sensitive:
        from cloudops_hub.firewall_model_data import user_history

        history = user_history(history)
    parameter_context = (
        parameter_explanation_context(payload.question, conversation["messages"])
        if assistant_id == "landing-zone" and not payload.schema_id
        else None
    )
    previous = conversation["messages"][-1] if conversation["messages"] else {}
    pending_draft = (
        previous.get("draft")
        if previous.get("role") == "assistant"
        and (
            previous.get("status") == "missing_parameters"
            or (previous.get("draft") or {}).get("awaiting_parameter_scope")
        )
        else None
    )
    import asyncio

    draft = None
    try:
        from cloudops_hub.live_catalog import RUN_URL, connected_run_reader, diagnose_run
        from cloudops_hub.service_catalog import user_github

        if assistant_id == "landing-zone" and RUN_URL.search(payload.question):
            try:
                with user_github(request, user) as github:
                    with connected_run_reader(github) as reader:
                        draft = await asyncio.to_thread(
                            diagnose_run, settings, payload.question, run_reader=reader,
                        )
            except HTTPException as exc:
                if exc.status_code != 403:
                    raise
                draft = {
                    "status": "github_auth_required",
                    "answer": (
                        "Connect GitHub to read this workflow's live logs and check "
                        "the CloudOps known-error guide for a documented fix. "
                        + str(exc.detail)
                    ),
                    "sources": [],
                }
        elif assistant_id == "landing-zone" and (
            payload.schema_id or landing_zone_json_request(payload.question)
        ):
            draft = landing_zone_json_redirect()
        elif payload.schema_id:
            draft = service_catalog_redirect()
        elif assistant_id == "landing-zone" and not parameter_context and configured(settings):
            draft = await asyncio.to_thread(
                catalog_chat,
                settings,
                payload.question,
                history,
                **({"pending_draft": pending_draft} if pending_draft else {}),
                allow_json=False,
            )
        elif assistant_id == "firewall" and configured(settings):
            draft = await asyncio.to_thread(
                run_audited,
                request.app.state.operations,
                settings,
                user,
                azp,
                firewall_json_chat,
                settings,
                payload.question,
                history,
                **(
                    {"analysis_state": conversation["firewall_state"]}
                    if conversation.get("firewall_state")
                    else {}
                ),
            )
    except FirewallAuditUnavailable:
        raise
    except ModelRateLimited as exc:
        raise HTTPException(
            status_code=429,
            detail=str(exc),
            headers={"Retry-After": str(exc.retry_after)},
        ) from None
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Request unavailable: {exc}",
        ) from None
    unmatched_run = None
    search_hint = None
    if draft is not None and draft.get("intent") == "guidance":
        search_hint = draft.get("search_query")
        draft = None
    if draft is not None:
        unmatched_run = draft.pop("unmatched_run", None)
        analysis_state = draft.pop("analysis_state", None)
        if analysis_state is not None:
            conversation["firewall_state"] = analysis_state
        source_links = draft.get("sources", [])
        result = {
            "status": draft["status"],
            "answer": draft["answer"],
            "sources": [
                {
                    "title": item["path"],
                    "url": item["url"],
                    "version": item["commit"],
                    "evidence_type": "live_repository" if "live_check" in draft else "reference",
                }
                for item in source_links
            ],
            **({"live_check": draft["live_check"]} if "live_check" in draft else {}),
            **(
                {"clarification_choices": draft["clarification_choices"]}
                if draft.get("status") == "clarification" and draft.get("clarification_choices")
                else {}
            ),
        }
        if draft.get("questions"):
            result["answer"] += "\n\n" + "\n".join(draft["questions"])
        display_json = draft.get("display_json", draft.get("json"))
        if display_json is not None:
            import json

            dependency = result["answer"].strip()
            result["answer"] = "```json\n" + json.dumps(display_json, indent=2) + "\n```"
            if dependency:
                result["answer"] += "\n\n" + dependency
    else:
        result = await asyncio.to_thread(
            answer_question,
            settings,
            assistant_id,
            payload.question,
            history,
            match_error(payload.question) if assistant_id != "platform-admin" else None,
            **({"parameter_context": parameter_context} if parameter_context else {}),
            **({"search_hint": search_hint} if search_hint else {}),
        )
    exact_guide_answer = assistant_id != "firewall" and bool(result.get("answer_contract"))
    if assistant_id == "landing-zone" and not exact_guide_answer:
        result = catalog_guidance_paths(result, payload.question)
    # Glossary keywords are extracted later by the scheduled keyword scan job, not here.
    previous_questions = [
        item["content"]
        for item in conversation["messages"]
        if item.get("role") == "user" and isinstance(item.get("content"), str)
    ]
    language = conversation_language(payload.question, previous_questions)
    if not exact_guide_answer and not (
        assistant_id == "firewall" and draft and draft.get("backend_only")
    ):
        result["answer"] = await asyncio.to_thread(
            localize_answer, settings, result["answer"], language
        )
    if firewall_sensitive:
        from cloudops_hub.firewall_model_data import active_references

        references = active_references()
        if references is None:
            raise RuntimeError("Firewall endpoint boundary is missing")
        if draft:
            references.values.update(draft.pop("endpoint_references", {}) or {})
        result["answer"] = references.render(result["answer"])
        if "clarification_choices" in result:
            result["clarification_choices"] = [
                references.render(choice) for choice in result["clarification_choices"]
            ]
        # Append backend-only evidence after any model-based translation.
        result["answer"] += draft.get("backend_guidance", "") if draft else ""
        state = conversation.setdefault("firewall_state", {})
        state["evidence_version"] = EVIDENCE_VERSION
        state["endpoint_references"] = references.values
    log_question = (
        settings.question_log_enabled
        and assistant_id != "platform-admin"
        and not firewall_sensitive
        and not _answer_test.get()
    )
    gap = log_question and result["status"] in GAP_STATUSES
    if gap:
        result["answer"] += "\n\n" + LANGUAGE_NOTE[language]
    message_id = uuid.uuid4().hex
    conversation["messages"].extend(
        [
            {"role": "user", "content": payload.question},
            {
                "role": "assistant",
                "id": message_id,
                "assistant_id": assistant_id,
                "content": result["answer"],
                "status": result["status"],
                "sources": result["sources"],
                **({"confidence": result["confidence"]} if "confidence" in result else {}),
                **({"answer_contract": result["answer_contract"]} if exact_guide_answer else {}),
                "evidence_type": (
                    "catalog_draft"
                    if draft
                    and draft.get("schema_id")
                    and draft.get("output_scope") in {"block", "file"}
                    else "live_check"
                    if "live_check" in result
                    else "documentation"
                ),
                **({"live_check": result["live_check"]} if "live_check" in result else {}),
                **({"draft": draft} if draft is not None else {}),
                **{key: result[key] for key in ("diagram", "knowledge_gap") if key in result},
                **(
                    {"clarification_choices": result["clarification_choices"]}
                    if "clarification_choices" in result
                    else {}
                ),
            },
        ]
    )
    conversation["updated_at"] = datetime.now(UTC).isoformat()
    store.save(user["oid"], conversation)
    if assistant_id != "platform-admin" and not firewall_sensitive and not _answer_test.get():
        record_retrieval(settings, message_id, payload.question, result)
    if log_question and not is_short_reply(payload.question):
        background.add_task(
            request.app.state.questions.record,
            message_id,
            assistant_id,
            payload.question,
            result["status"],
            language,
            previous_questions[-2:],
            mentions_contact_cloudops(str(result.get("answer") or "")),
            confidence=result.get("confidence"),
            **({"enrich": False} if firewall_sensitive else {}),
        )
    if unmatched_run and settings.question_log_enabled and not _answer_test.get():
        background.add_task(request.app.state.questions.record_run, unmatched_run)
    if ADMIN_ROLE not in user.get("roles", []) and "live_check" in result:
        result["live_check"] = {
            key: value
            for key, value in result["live_check"].items()
            if key in {"kind", "checked_at", "read_only", "reused_evidence"}
        }
    result.pop("retrieval_trace", None)
    result.pop("_graph_selection", None)
    return {**result, "conversation_id": conversation["id"], "message_id": message_id}
