"""Content-free model accounting and bounded, authorized request-local history."""

import logging
import time
import uuid

from cloudops_hub import firewall_audit
from cloudops_hub.config import Settings
from cloudops_hub.firewall_model_data import active_references

logger = logging.getLogger(__name__)


def recent_history(settings: Settings, history: list[dict]) -> list[dict[str, str]]:
    if not settings.history_max_messages or not settings.history_max_chars:
        return []
    selected = []
    size = 0
    for item in reversed(history[-settings.history_max_messages :]):
        if item.get("role") not in {"user", "assistant"} or not isinstance(
            item.get("content"), str
        ):
            continue
        content = item["content"]
        if size + len(content) > settings.history_max_chars:
            break
        selected.append({"role": item["role"], "content": content})
        size += len(content)
    selected.reverse()
    return selected


def record_usage(settings: Settings, operation: str, model: str, response, started: float):
    usage = getattr(response, "usage", None)
    input_tokens = getattr(usage, "prompt_tokens", None)
    if input_tokens is None:
        input_tokens = getattr(usage, "input_tokens", None)
    if input_tokens is None:
        input_tokens = getattr(usage, "total_tokens", None) if operation == "embedding" else None
    output_tokens = 0 if operation == "embedding" else getattr(usage, "completion_tokens", None)
    if output_tokens is None:
        output_tokens = getattr(usage, "output_tokens", None)
    price = (
        settings.embedding_input_price_per_million
        if operation == "embedding"
        else settings.model_input_price_per_million
    )
    output_price = 0 if operation == "embedding" else settings.model_output_price_per_million
    cost = (
        (input_tokens * price + output_tokens * output_price) / 1_000_000
        if all(value is not None for value in (input_tokens, output_tokens, price, output_price))
        else None
    )
    logger.info(
        "model_usage operation=%s model=%s input_tokens=%s output_tokens=%s "
        "latency_ms=%d estimated_cost=%s currency=%s",
        operation,
        model,
        input_tokens,
        output_tokens,
        round((time.monotonic() - started) * 1000),
        cost,
        settings.model_price_currency,
    )


def complete(
    client, settings: Settings, operation: str, *,
    responses_api=False, embeddings_api=False, **kwargs
):
    if not embeddings_api:
        kwargs["store"] = False
    references = active_references()
    if "messages" in kwargs and references is not None:
        kwargs["messages"] = references.messages(kwargs["messages"])
    kwargs.setdefault("timeout", settings.model_request_timeout_seconds)
    started = time.monotonic()
    operation_id = uuid.uuid4().hex
    audit_request = kwargs
    if responses_api and isinstance(kwargs.get("input"), list):
        audit_request = {
            **kwargs,
            "input": [
                {"type": "reasoning", "content": "[omitted]"}
                if isinstance(item, dict) and item.get("type") == "reasoning"
                else item
                for item in kwargs["input"]
            ],
        }
    firewall_audit.emit(
        "model", "started", operation_id=operation_id,
        name=operation, model=kwargs.get("model", ""), responses_api=responses_api,
        embeddings_api=embeddings_api, request=audit_request,
    )
    try:
        response = (
            client.embeddings.create(**kwargs)
            if embeddings_api
            else client.responses.create(**kwargs)
            if responses_api
            else client.chat.completions.create(**kwargs)
        )
    except Exception as exc:
        logger.warning(
            "model_failed operation=%s model=%s latency_ms=%d error_type=%s",
            operation,
            kwargs.get("model", ""),
            round((time.monotonic() - started) * 1000),
            type(exc).__name__,
        )
        firewall_audit.emit(
            "model", "failed", operation_id=operation_id, name=operation,
            error_type=type(exc).__name__,
            duration_ms=round((time.monotonic() - started) * 1000),
        )
        raise
    record_usage(settings, operation, kwargs.get("model", ""), response, started)
    if firewall_audit.correlation_id() is not None:
        usage = getattr(response, "usage", None)
        firewall_audit.emit(
            "model", "completed", operation_id=operation_id, name=operation,
            model=kwargs.get("model", ""), response_id=getattr(response, "id", None),
            status=getattr(response, "status", None),
            duration_ms=round((time.monotonic() - started) * 1000),
            finish_reasons=[
                choice.finish_reason for choice in (getattr(response, "choices", None) or [])
            ],
            usage={
                key: value
                for key in ("prompt_tokens", "completion_tokens", "input_tokens",
                            "output_tokens", "total_tokens")
                if isinstance(value := getattr(usage, key, None), int)
            },
        )
    if references is not None and references.restore_outputs:
        restore_response(references, response)
    return response


def restore_response(references, response) -> None:
    for choice in getattr(response, "choices", None) or []:
        message = getattr(choice, "message", None)
        if message is None:
            continue
        if isinstance(getattr(message, "content", None), str):
            message.content = references.restore(message.content)
        for call in getattr(message, "tool_calls", None) or []:
            function = getattr(call, "function", None)
            if isinstance(getattr(function, "arguments", None), str):
                function.arguments = references.restore(function.arguments)
