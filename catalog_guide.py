"""Package the reviewed service-catalog guide as an independent knowledge source."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from cloudops_hub.catalog import catalog_schemas
from cloudops_hub.catalog_contracts import RELEASE_COMMIT
from cloudops_hub.ingest import ingest

GUIDE_DIR = Path(__file__).resolve().parent.parent / "data" / "catalog-guide"
SOURCE_ID = "guide:service-catalog"
GUIDE_FILES = (
    "overview.md",
    "execution.md",
    "inputs.md",
    "workspaces.md",
    "troubleshooting.md",
    "parameters.md",
)
SOURCE_BASE = "https://github.com/almbrand/cloudops-openai-hub/blob/main/backend/data/catalog-guide"


def _cell(value: Any) -> str:
    return str(value).replace("|", r"\|").replace("\n", " ")


def _type(schema: dict) -> str:
    if "oneOf" in schema:
        return " or ".join(_type(option) for option in schema["oneOf"])
    if "const" in schema:
        return json.dumps(schema["const"])
    if "enum" in schema:
        return "enum: " + ", ".join(json.dumps(item) for item in schema["enum"])
    types = schema.get("type", "value")
    return " or ".join(types) if isinstance(types, list) else types


def _parameter_rows(schema: dict, path: str = "$", required: bool = True):
    notes = []
    if "default" in schema:
        notes.append("Default: " + json.dumps(schema["default"]))
    if schema.get("description"):
        notes.append(schema["description"])
    for key in ("pattern", "minimum", "maximum", "minItems", "maxItems", "uniqueItems", "not"):
        if key in schema:
            notes.append(f"{key}: {json.dumps(schema[key])}")
    for keyword in ("if", "then", "else", "allOf", "anyOf"):
        if keyword in schema:
            notes.append(f"{keyword}: {json.dumps(schema[keyword], separators=(',', ':'))}")
    if "properties" in schema:
        notes.append("Only the listed object properties are accepted.")
    yield (
        f"| `{_cell(path)}` | {_cell(_type(schema))} | "
        f"{'yes' if required else 'no'} | {_cell('; '.join(notes) or '-')} |"
    )
    for key, child in schema.get("properties", {}).items():
        yield from _parameter_rows(child, f"{path}.{key}", key in schema.get("required", []))
    if "items" in schema:
        yield from _parameter_rows(schema["items"], f"{path}[]")
    if isinstance(schema.get("additionalProperties"), dict):
        yield from _parameter_rows(schema["additionalProperties"], f"{path}.<map-key>", False)
    for index, option in enumerate(schema.get("oneOf", []), 1):
        yield from _parameter_rows(option, f"{path} (alternative {index})", required)


def parameter_reference() -> str:
    lines = [
        "# Service catalog JSON parameter reference",
        "",
        f"Consumer review: `release` commit `{RELEASE_COMMIT}`.",
        "",
        "Generated from the reviewed generation contracts by "
        "`python -m cloudops_hub.catalog_guide --write-reference` (run from backend). "
        "Do not edit the generated tables manually.",
        "",
        "These tables describe the Hub's reviewed JSON inputs, not every Azure/GitHub "
        "provider attribute. Current release consumers take precedence if they change. "
        "Required means required when the enclosing object/array entry is supplied; "
        "conditional requirements are stated in constraints. Missing defaults mean "
        "not specified here, not a promise that Terraform supplies a value. Map-key "
        "entries accept user-selected names with the displayed value type. No team "
        "name is needed to show an entry block.",
        "",
    ]
    for schema_id, schema in catalog_schemas().items():
        lines += [
            f"## {schema['title']} ({schema_id})",
            "",
            f"File: `{schema['x-target-path']}`. "
            f"Entry insertion path: `{'.'.join(schema['x-entry-path'])}`.",
            "",
            schema["description"],
            "",
            "Release consumers: "
            + ", ".join(f"`{item['path']}`" for item in schema["x-sources"])
            + ".",
            "",
            "| JSON path | Type | Required in parent | Defaults and constraints |",
            "|---|---|---|---|",
            *_parameter_rows(schema),
            "",
        ]
    return "\n".join(lines)


def ingest_guide(*, before_write=None) -> None:
    metadata = {}
    digest = hashlib.sha256()
    for filename in GUIDE_FILES:
        path = GUIDE_DIR / filename
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            raise ValueError(f"Catalog guide document is empty: {filename}")
        if filename == "parameters.md" and text != parameter_reference():
            raise ValueError("Catalog parameter guide is stale; regenerate it before importing")
        content = text.encode("utf-8")
        digest.update(filename.encode())
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")
        metadata[filename] = {
            "title": text.splitlines()[0].lstrip("# "),
            "url": f"{SOURCE_BASE}/{filename}",
            "assistant": "landing-zone",
        }
    actual = {path.name for path in GUIDE_DIR.iterdir() if path.is_file()}
    if actual != set(GUIDE_FILES) or any(path.is_dir() for path in GUIDE_DIR.iterdir()):
        raise ValueError("Catalog guide contains unreviewed or missing documents")
    ingest(
        GUIDE_DIR,
        source_id=SOURCE_ID,
        version=digest.hexdigest(),
        source_kind="operations",
        assistant="landing-zone",
        metadata=metadata,
        before_write=before_write,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-reference", action="store_true", required=True)
    parser.parse_args()
    (GUIDE_DIR / "parameters.md").write_text(parameter_reference(), encoding="utf-8", newline="\n")
