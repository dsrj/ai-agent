"""Versioned Azure vocabulary; runtime matching is local, never a support contract."""

import argparse
import hashlib
import json
import re
from datetime import UTC, datetime
from functools import lru_cache
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import urlopen

DATA = Path(__file__).resolve().parents[1] / "data"
CATALOG = DATA / "microsoft-resources.json"
TERMS = DATA / "resource-terminology.json"
SOURCE = (
    "https://learn.microsoft.com/en-us/azure/cloud-adoption-framework/"
    "ready/azure-best-practices/resource-abbreviations"
)


class ResourceTables(HTMLParser):
    def __init__(self):
        super().__init__()
        self.cell = None
        self.row = []
        self.rows = []

    def handle_starttag(self, tag, attrs):
        if tag in {"td", "th"}:
            self.cell = []
        if tag == "tr":
            self.row = []

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if tag in {"td", "th"} and self.cell is not None:
            self.row.append(" ".join("".join(self.cell).split()))
            self.cell = None
        if tag == "tr" and self.row:
            self.rows.append(self.row)


def parse_resources(html):
    parser = ResourceTables()
    parser.feed(html)
    entries = []
    seen = set()
    for row in parser.rows:
        if not any("Microsoft." in cell for cell in row):
            continue
        if len(row) != 3:
            raise ValueError("Microsoft resource table columns changed")
        name, provider, abbreviation = row
        match = re.fullmatch(r"(Microsoft\.[\w./-]+)(?:\s+(.+))?", provider)
        if not match:
            raise ValueError("Microsoft resource provider format changed")
        entry = {
            "name": name,
            "resource_type": match[1],
            "qualifier": match[2] or "",
            "abbreviation": abbreviation,
        }
        identity = tuple(entry.values())
        if identity not in seen:
            entries.append(entry)
            seen.add(identity)
    if not 100 <= len(entries) <= 2000:
        raise ValueError("Microsoft resource table is missing or unexpectedly sized")
    return entries


def validate_catalog(value):
    if (
        not isinstance(value, dict)
        or value.get("version") != 1
        or value.get("source_url") != SOURCE
        or not isinstance(value.get("resources"), list)
        or not 100 <= len(value["resources"]) <= 2000
        or not isinstance(value.get("source_sha256"), str)
        or not re.fullmatch(r"[a-f0-9]{64}", value["source_sha256"])
        or not isinstance(value.get("retrieved_at"), str)
    ):
        raise ValueError("Invalid Microsoft resource catalogue")
    if datetime.fromisoformat(value["retrieved_at"]).utcoffset() is None:
        raise ValueError("Microsoft resource catalogue timestamp requires a timezone")
    for entry in value["resources"]:
        if (
            not isinstance(entry, dict)
            or set(entry) != {"name", "resource_type", "qualifier", "abbreviation"}
            or any(not isinstance(text, str) or len(text) > 250 for text in entry.values())
            or not entry["name"].strip()
            or not re.fullmatch(r"Microsoft\.[\w./-]+", entry["resource_type"])
        ):
            raise ValueError("Invalid Microsoft resource catalogue entry")
    return value


@lru_cache(maxsize=1)
def vocabulary():
    if CATALOG.stat().st_size > 1_000_000 or TERMS.stat().st_size > 65_536:
        raise ValueError("Resource vocabulary exceeds its size limit")
    catalogue = validate_catalog(json.loads(CATALOG.read_text(encoding="utf-8")))
    terminology = json.loads(TERMS.read_text(encoding="utf-8"))
    if (
        not isinstance(terminology, dict)
        or terminology.get("version") != 1
        or not isinstance(terminology.get("groups"), list)
        or len(terminology["groups"]) > 200
    ):
        raise ValueError("Invalid conversational resource terminology")
    groups = []
    for group in terminology["groups"]:
        if (
            not isinstance(group, list)
            or not 2 <= len(group) <= 16
            or any(
                not isinstance(term, str)
                or not term.strip()
                or len(term) > 150
                or not re.fullmatch(r"[\w .:/-]+", term)
                for term in group
            )
        ):
            raise ValueError("Invalid conversational resource terminology group")
        groups.append(tuple(group))
    for item in catalogue["resources"]:
        name = item["name"]
        if item["qualifier"]:
            name += f" {item['qualifier']}"
        # Naming prefixes are suggestions, not all conversational aliases. Keep very
        # short prefixes out of free-text matching unless explicitly curated above.
        terms = [name, item["name"], item["resource_type"]]
        if len(item["abbreviation"]) >= 3:
            terms.append(item["abbreviation"])
        groups.append(tuple(dict.fromkeys(terms)))
    return tuple(groups)


@lru_cache(maxsize=4096)
def _term_pattern(term):
    return re.compile(rf"(?<![\w./:@-]){re.escape(term)}(?![\w/:@-]|\.[\w-])", re.I)


def resource_matches(text):
    matches = []
    for group in vocabulary():
        length = max((len(term) for term in group if _term_pattern(term).search(text)), default=0)
        if length:
            matches.append((length, group))
    # Prefer descriptive matches over generic short naming prefixes.
    matches.sort(key=lambda match: match[0], reverse=True)
    return [list(group) for _, group in matches[:12]]


def main():
    parser = argparse.ArgumentParser(description="Refresh reviewed Azure vocabulary snapshot")
    parser.add_argument("--refresh", action="store_true", required=True)
    parser.add_argument("--output", type=Path, default=CATALOG)
    args = parser.parse_args()
    with urlopen(SOURCE, timeout=45) as response:
        raw = response.read(2_000_001)
    if len(raw) > 2_000_000:
        raise ValueError("Microsoft resource page exceeded its size limit")
    resources = parse_resources(raw.decode("utf-8"))
    value = validate_catalog(
        {
            "version": 1,
            "source_url": SOURCE,
            "retrieved_at": datetime.now(UTC).isoformat(),
            "source_sha256": hashlib.sha256(raw).hexdigest(),
            "coverage": "Azure resources listed in Microsoft CAF, not all Microsoft products",
            "resources": resources,
        }
    )
    args.output.write_text(json.dumps(value, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(f"Written {len(resources)} Microsoft resource entries; review before publishing.")


if __name__ == "__main__":
    main()
