"""Keep administrator answers from Question insights live in the chat search index.

Saving an answer indexes it immediately; dismissing or deleting the question removes it.
"""

import hashlib
import logging
from typing import Any

from azure.core.exceptions import AzureError
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient

from cloudops_hub.config import Settings
from cloudops_hub.knowledge import embed_texts, identifiers

logger = logging.getLogger(__name__)

SOURCE_ID = "insights:answers"
ASSISTANTS = {"firewall", "landing-zone"}


def document_id(entry_id: str) -> str:
    return hashlib.sha256(f"{SOURCE_ID}:{entry_id}".encode()).hexdigest()


def is_live(entry: dict[str, Any]) -> bool:
    review = entry.get("review") or {}
    return review.get("status") == "answered" and bool((review.get("answer") or "").strip())


def build_document(entry: dict[str, Any]) -> dict[str, Any]:
    if entry.get("assistant_id") == "platform-admin":
        raise ValueError("Restricted platform answers cannot enter the general answer index")
    topic = " ".join((entry.get("topic") or "").split()) or "Answered question"
    question = " ".join((entry.get("question") or "").split())
    answer = entry["review"]["answer"].strip()
    content = f"## {topic}\n\n**Question:** {question}\n\n**Answer:** {answer}"
    assistant = entry.get("assistant_id")
    return {
        "id": document_id(entry["id"]),
        "content": content,
        "title": f"CloudOps answer: {topic}"[:250],
        "source_url": "",
        "source_path": f"insights/answers/{entry['id']}",
        "source_id": SOURCE_ID,
        "source_kind": "operations",
        "assistant": assistant if assistant in ASSISTANTS else "shared",
        "version": entry["review"].get("updated_at") or "",
        "last_updated": entry["review"].get("updated_at") or "",
        "section": topic,
        "chunk_number": 0,
        "technical_terms": identifiers(content, limit=512),
    }


class AnswerIndex:
    def __init__(self, settings: Settings):
        self.settings = settings

    @property
    def enabled(self) -> bool:
        return bool(self.settings.azure_search_endpoint)

    def _client(self, credential) -> SearchClient:
        return SearchClient(
            self.settings.azure_search_endpoint,
            self.settings.azure_search_index,
            credential=credential,
        )

    def _with_vectors(self, documents: list[dict[str, Any]], credential) -> None:
        if not self.settings.azure_search_hybrid_enabled or not documents:
            return
        from cloudops_hub.retrieval import create_openai_client

        with create_openai_client(self.settings, credential) as ai:
            for start in range(0, len(documents), 16):
                batch = documents[start : start + 16]
                vectors = embed_texts(
                    ai,
                    self.settings,
                    [f"{doc['title']}\n{doc['source_path']}\n{doc['content']}" for doc in batch],
                )
                for doc, vector in zip(batch, vectors, strict=True):
                    doc["content_vector"] = vector

    def sync(self, entry: dict[str, Any]) -> bool:
        """Index or remove one question's answer. Returns False if the index was not updated."""
        if not self.enabled:
            return False
        credential = DefaultAzureCredential()
        try:
            client = self._client(credential)
            if is_live(entry):
                document = build_document(entry)
                self._with_vectors([document], credential)
                result = client.merge_or_upload_documents([document])
            else:
                result = client.delete_documents([{"id": document_id(entry["id"])}])
            return all(item.succeeded for item in result)
        except (AzureError, OSError, ValueError, RuntimeError) as exc:
            logger.warning("Answer index update failed (%s)", type(exc).__name__)
            return False
        finally:
            credential.close()

    def remove(self, entry_id: str) -> bool:
        return self.sync({"id": entry_id, "review": {"status": "dismissed", "answer": ""}})

    def sync_all(self, entries: list[dict[str, Any]]) -> int:
        """Make the index match the answered questions exactly. Returns the indexed count."""
        if not self.enabled:
            return 0
        credential = DefaultAzureCredential()
        try:
            client = self._client(credential)
            documents = [
                build_document(entry)
                for entry in entries
                if is_live(entry) and entry.get("assistant_id") != "platform-admin"
            ]
            existing = {
                item["id"]
                for item in client.search(
                    search_text="*", filter=f"source_id eq '{SOURCE_ID}'", select=["id"]
                )
            }
            self._with_vectors(documents, credential)
            for start in range(0, len(documents), 100):
                client.merge_or_upload_documents(documents[start : start + 100])
            stale = existing - {doc["id"] for doc in documents}
            if stale:
                client.delete_documents([{"id": item} for item in stale])
            return len(documents)
        except (AzureError, OSError, ValueError, RuntimeError) as exc:
            logger.warning("Answer index sync failed (%s)", type(exc).__name__)
            return 0
        finally:
            credential.close()
