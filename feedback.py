"""Privacy-minimised retrieval telemetry; raw questions remain in user-scoped history."""

import hashlib
import hmac
import json
import logging

logger = logging.getLogger(__name__)


def record_retrieval(settings, message_id: str, question: str, result: dict) -> None:
    fingerprint = hmac.new(
        settings.session_secret.encode(), question.encode(), hashlib.sha256
    ).hexdigest()
    logger.info(
        "retrieval_event %s",
        json.dumps(
            {
                "message_id": message_id.replace("\r", "").replace("\n", ""),
                "question_fingerprint": fingerprint,
                "question_chars": len(question),
                "status": result["status"],
                "evidence_based_score": result.get("confidence"),
                "hybrid": settings.azure_search_hybrid_enabled,
                "semantic": settings.azure_search_semantic_enabled,
                "index": settings.azure_search_index,
                "documents": result.get("retrieval_trace")
                or [
                    {
                        "id": source.get("document_id")
                        or hashlib.sha256(
                            (source.get("url", "") + source.get("path", "")).encode()
                        ).hexdigest(),
                        "score": source.get("score"),
                        "reranker_score": source.get("reranker_score"),
                    }
                    for source in result.get("sources", [])
                ],
            }
        ),
    )


def record_feedback(message_id: str, helpful: bool) -> None:
    logger.info(
        "retrieval_feedback %s",
        json.dumps(
            {
                "message_id": message_id.replace("\r", "").replace("\n", ""),
                "helpful": helpful,
            }
        ),
    )
