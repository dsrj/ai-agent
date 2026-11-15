"""Shared diagram limits and approved icon identifiers."""

import json
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
ICON_INDEX = Path(__file__).resolve().parent.parent / "data" / "azure-icons.json"
MAX_ZONES, MAX_NODES, MAX_EDGES = 6, 14, 20
ACTION_LABEL = re.compile(
    r"^(?:add|approve|ask|check|choose|confirm|create|delete|enable|request|"
    r"schedule|select|submit|use|verify|wait|step\s+\d+)\b",
    re.IGNORECASE,
)


def component_label(label: Any) -> bool:
    """Keep procedure instructions out of component diagrams."""
    return (
        isinstance(label, str) and bool(label.strip()) and len(label) <= 64
        and len(label.split()) <= 8 and not ACTION_LABEL.search(label.strip())
        and not re.search(r"[`\n]|[.!?](?:\s|$)", label)
    )


@lru_cache(maxsize=1)
def icon_ids() -> frozenset[str]:
    try:
        return frozenset(item["id"] for item in json.loads(ICON_INDEX.read_text("utf-8")))
    except (OSError, ValueError, KeyError, TypeError):
        logger.warning("Azure icon index unavailable")
        return frozenset()
