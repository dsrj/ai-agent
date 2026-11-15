import hashlib
import json
import math
import re
import uuid
from copy import deepcopy
from datetime import UTC, datetime
from threading import RLock, Timer
from time import monotonic
from typing import Any

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient

from cloudops_hub.config import Settings

CONVERSATION_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
_local_conversations: dict[str, dict[str, Any]] = {}
FIREWALL_IDLE_SECONDS = 30 * 60


def temporary_conversation(conversation: dict) -> bool:
    return (
        conversation.get("temporary") is True
        or conversation.get("assistant_id") == "firewall"
        or any(
            message.get("assistant_id") == "firewall"
            for message in conversation.get("messages", [])
        )
    )


def conversation_view(conversation: dict[str, Any], *, admin: bool) -> dict[str, Any]:
    """Expose chat display data, not persisted backend execution state."""
    if admin:
        return deepcopy(conversation)
    result = {
        key: deepcopy(value)
        for key, value in conversation.items()
        if key in {"id", "assistant_id", "updated_at", "temporary", "expires_at"}
    }
    result["messages"] = []
    for message in conversation.get("messages", []):
        if message.get("role") not in {"user", "assistant"}:
            continue
        display = {
            key: deepcopy(value)
            for key, value in message.items()
            if key
            in {
                "id",
                "role",
                "content",
                "assistant_id",
                "status",
                "sources",
                "evidence_type",
                "confidence",
                "answer_contract",
                "feedback",
                "diagram",
                "knowledge_gap",
                "clarification_choices",
            }
        }
        draft = message.get("draft")
        if isinstance(draft, dict):
            display["draft"] = {
                key: deepcopy(value)
                for key, value in draft.items()
                if key
                in {
                    "status",
                    "schema_id",
                    "output_scope",
                    "json",
                    "target_repository",
                    "target_path",
                    "warnings",
                }
            }
        result["messages"].append(display)
    return result


def _user_key(user_id: str) -> str:
    return hashlib.sha256(user_id.encode("utf-8")).hexdigest()


class ConversationStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._credential = None
        self._container = None
        self._temporary: dict[str, tuple[dict, float, Timer]] = {}
        self._temporary_lock = RLock()
        if settings.azure_storage_account_url:
            self._credential = DefaultAzureCredential()
            service = BlobServiceClient(
                settings.azure_storage_account_url, credential=self._credential
            )
            self._container = service.get_container_client(
                settings.azure_storage_conversation_container
            )

    def _blob_name(self, user_id: str, conversation_id: str) -> str:
        if not CONVERSATION_ID_PATTERN.fullmatch(conversation_id):
            raise ValueError("Invalid conversation ID")
        return f"{_user_key(user_id)}/{conversation_id}.json"

    def list(self, user_id: str) -> list[dict[str, Any]]:
        prefix = f"{_user_key(user_id)}/"
        if self._container:
            records = [
                json.loads(self._container.get_blob_client(blob.name).download_blob().readall())
                for blob in self._container.list_blobs(name_starts_with=prefix)
            ]
            return [record for record in records if not temporary_conversation(record)]
        return [
            deepcopy(value)
            for key, value in _local_conversations.items()
            if key.startswith(prefix) and not temporary_conversation(value)
        ]

    def _expire(self, name: str, deadline: float) -> None:
        with self._temporary_lock:
            current = self._temporary.get(name)
            if current and current[1] == deadline:
                self._temporary.pop(name)

    def _save_temporary(self, name: str, conversation: dict, *, create: bool = False) -> None:
        with self._temporary_lock:
            previous = self._temporary.get(name)
            if not create and (previous is None or previous[1] <= monotonic()):
                raise RuntimeError("Temporary Firewall chat ended or expired. Start a new chat.")
            if previous:
                previous[2].cancel()
            deadline = max(
                monotonic() + FIREWALL_IDLE_SECONDS,
                math.nextafter(previous[1], math.inf) if previous else 0,
            )
            timer = Timer(FIREWALL_IDLE_SECONDS, self._expire, args=(name, deadline))
            timer.daemon = True
            conversation["temporary"] = True
            conversation["expires_at"] = datetime.fromtimestamp(
                datetime.now(UTC).timestamp() + FIREWALL_IDLE_SECONDS, UTC
            ).isoformat()
            self._temporary[name] = (deepcopy(conversation), deadline, timer)
            timer.start()

    def get(self, user_id: str, conversation_id: str) -> dict[str, Any] | None:
        try:
            name = self._blob_name(user_id, conversation_id)
        except ValueError:
            return None
        with self._temporary_lock:
            current = self._temporary.get(name)
            if current:
                if current[1] <= monotonic():
                    current[2].cancel()
                    self._temporary.pop(name)
                    return None
                return deepcopy(current[0])
        if self._container:
            blob = self._container.get_blob_client(name)
            if not blob.exists():
                return None
            result = json.loads(blob.download_blob().readall())
            return None if temporary_conversation(result) else result
        value = _local_conversations.get(name)
        return deepcopy(value) if value and not temporary_conversation(value) else None

    def save(self, user_id: str, conversation: dict[str, Any]) -> None:
        name = self._blob_name(user_id, conversation["id"])
        if temporary_conversation(conversation):
            self._save_temporary(name, conversation, create=not conversation.get("temporary"))
            _local_conversations.pop(name, None)
            return
        body = json.dumps(conversation, ensure_ascii=False).encode("utf-8")
        if self._container:
            self._container.upload_blob(name, body, overwrite=True, content_type="application/json")
        else:
            _local_conversations[name] = conversation.copy()

    def delete(self, user_id: str, conversation_id: str) -> bool:
        try:
            name = self._blob_name(user_id, conversation_id)
        except ValueError:
            return False
        with self._temporary_lock:
            current = self._temporary.pop(name, None)
            if current:
                current[2].cancel()
                return True
        if self._container:
            blob = self._container.get_blob_client(name)
            if not blob.exists():
                return False
            blob.delete_blob()
            return True
        return _local_conversations.pop(name, None) is not None

    def create(self, user_id: str, assistant_id: str) -> dict[str, Any]:
        conversation = {
            "id": uuid.uuid4().hex,
            "assistant_id": assistant_id,
            "messages": [],
            "updated_at": datetime.now(UTC).isoformat(),
        }
        self.save(user_id, conversation)
        return conversation

    def create_temporary(self, user_id: str, assistant_id: str) -> dict[str, Any]:
        """In-memory only and hidden from history; expires like a temporary Firewall chat."""
        conversation = {
            "id": uuid.uuid4().hex,
            "assistant_id": assistant_id,
            "messages": [],
            "updated_at": datetime.now(UTC).isoformat(),
        }
        name = self._blob_name(user_id, conversation["id"])
        self._save_temporary(name, conversation, create=True)
        return conversation

    def close(self) -> None:
        with self._temporary_lock:
            for _, _, timer in self._temporary.values():
                timer.cancel()
            self._temporary.clear()
        if self._credential:
            self._credential.close()
