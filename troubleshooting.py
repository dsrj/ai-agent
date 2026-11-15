import json
import re
from pathlib import Path
from typing import Any

CATALOG_PATH = Path(__file__).resolve().parent.parent / "data" / "troubleshooting.json"


def load_catalog(path: Path = CATALOG_PATH) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as catalog_file:
        entries = json.load(catalog_file)
    if not isinstance(entries, list):
        raise ValueError("Troubleshooting catalog must be a JSON array")
    return entries


def match_error(message: str, entries: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    catalog = entries if entries is not None else load_catalog()
    normalized = message.casefold()
    matches = []
    for entry in catalog:
        signatures = entry.get("signatures", [])
        if any(
            isinstance(signature, str)
            and re.search(rf"(?<!\w){re.escape(signature.casefold())}(?!\w)", normalized)
            for signature in signatures
        ):
            matches.append(entry)
    return matches
