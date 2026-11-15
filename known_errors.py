"""Match failed service-catalog runs against the owner-reviewed known-errors knowledge base.

The knowledge base is the Markdown file maintained by CloudOps in the release branch of
almbrand/landingzone-service-catalog-code. Matching is fully local and deterministic:
normalized error signatures and the failed-run ledger select records, and the answer is the
record's own Solution. Run logs are never sent to a model.
"""

import re
from typing import Any

REPOSITORY = "almbrand/landingzone-service-catalog-code"
REF = "release"
PATH = "cloudops-hub-knowledge-source/service-catalog-known-errors.md"
MAX_BYTES = 3_000_000
MAX_RECORD_CHARS = 8_000
MAX_MATCHES = 3
MIN_LITERAL = 10
# KE-9xx records describe missing/generic evidence, not a diagnosis.
GENERIC_PREFIX = "KE-9"

_RECORD = re.compile(r"(?ms)^### (KE-\d+):[ \t]*([^\n]*)\n(.*?)(?=^### |^## |^<a id=|\Z)")
_SIGNATURE = re.compile(r"\*\*Search signature[^\n]*\*\*\s*```[a-z]*\n(.*?)```", re.S)
_SOLUTION_LABEL = re.compile(r"^\*\*Solution(?! validation)[^\n]*?:\*\*", re.I)
_SOLUTION_TABLE = re.compile(r"^\|[^\n]*(solution|recovery)", re.I)
MAX_SOLUTION_CHARS = 3_000
_VALIDATION = re.compile(r"\s*\*\*Solution validation:\*\*[^\n]*", re.I)
# Provenance notes such as "Owner-provided guidance recorded on 2026-10-24." are not shown to users.
_PROVENANCE = re.compile(
    r"[ \t]*Owner-(?:provided|confirmed)[^.\n]*?recorded on \d{4}-\d{2}-\d{2}\.?", re.I
)


def _solution(body: str) -> str:
    """Owner guidance: every 'Solution...:' block (any qualifier) and solution/recovery tables."""
    blocks = [
        _PROVENANCE.sub("", _VALIDATION.sub("", block)).strip()
        for block in re.split(r"\n\s*\n|\n(?=\*\*[^*\n]+\*\*)", body)
        if _SOLUTION_LABEL.match(block.strip()) or _SOLUTION_TABLE.match(block.strip())
    ]
    return "\n\n".join(blocks)[:MAX_SOLUTION_CHARS]


_LEDGER = re.compile(r"(?ms)^## Failed-run and workspace-evidence ledger\n(.*?)(?=^## |\Z)")
_LEDGER_ROW = re.compile(r"^\|\s*\[(\d+)\]\([^)]*\)\s*\|(.*)$", re.M)
_TIMESTAMP = re.compile(r"(?m)^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z ?")
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_BOX = re.compile(r"[â”‚â•·â•µâ”€â•­â•®â•°â•¯â”Œâ”â””â”˜â”œâ”¤]")
_PLACEHOLDER = re.compile(r"<[^<>\n]{1,60}>|\.\.\.|â€¦")


def _pattern(line: str) -> re.Pattern | None:
    parts = _PLACEHOLDER.split(line.strip())
    if sum(len(part.strip()) for part in parts) < MIN_LITERAL:
        return None
    body = ".{0,300}?".join(r"\s+".join(re.escape(word) for word in part.split()) for part in parts)
    return re.compile(body, re.I)


def parse(text: str) -> dict[str, Any]:
    """Return records with compiled signatures plus the run ledger (run ID -> KE IDs)."""
    records = []
    for match in _RECORD.finditer(text):
        error_id, title, body = match[1], match[2].strip(), match[3]
        signature = _SIGNATURE.search(body)
        lines = [line for line in (signature[1].splitlines() if signature else []) if line.strip()]
        patterns = [pattern for line in lines if (pattern := _pattern(line))]
        records.append(
            {
                "id": error_id,
                "title": title,
                "anchor": error_id.lower(),
                "generic": error_id.startswith(GENERIC_PREFIX),
                "signatures": lines,
                "patterns": patterns,
                "solution": _solution(body),
                "text": f"### {error_id}: {title}\n{body.strip()}"[:MAX_RECORD_CHARS],
            }
        )
    ledger: dict[str, list[str]] = {}
    section = _LEDGER.search(text)
    if section:
        for row in _LEDGER_ROW.finditer(section[1]):
            ids = list(dict.fromkeys(re.findall(r"\bKE-\d+\b", row[2])))
            if ids:
                ledger[row[1]] = ids
    return {"records": records, "ledger": ledger}


def normalize_log(text: str) -> str:
    text = _TIMESTAMP.sub("", _ANSI.sub("", text))
    return re.sub(r"\s+", " ", _BOX.sub(" ", text))


def match(knowledge: dict[str, Any], logs: list[str], run_id: str | None = None) -> list[dict]:
    """Specific (non-generic) records ranked by ledger evidence and matched signature lines."""
    log = normalize_log("\n".join(logs))
    ledger_ids = set(knowledge["ledger"].get(run_id or "", []))
    scored = []
    for record in knowledge["records"]:
        if record["generic"]:
            continue
        hits = [
            line
            for line, pattern in zip(
                [line for line in record["signatures"] if _pattern(line)],
                record["patterns"],
                strict=True,
            )
            if pattern.search(log)
        ]
        in_ledger = record["id"] in ledger_ids
        if not hits and not in_ledger:
            continue
        scored.append(
            (
                (in_ledger, len(hits) / max(len(record["patterns"]), 1), len(hits)),
                {**record, "matched_signatures": hits, "in_run_ledger": in_ledger},
            )
        )
    scored.sort(key=lambda item: item[0], reverse=True)
    return [record for _, record in scored[:MAX_MATCHES]]


def error_excerpt(logs: list[str], limit: int = 2_000) -> str:
    """Short list of distinct error lines for the administrator's review queue."""
    lines = []
    for raw in "\n".join(logs).splitlines():
        line = _BOX.sub(" ", _TIMESTAMP.sub("", _ANSI.sub("", raw))).strip()
        if (
            re.search(r"(?i)\berror\b|failed|not found|forbidden|denied|exception", line)
            and line not in lines
        ):
            lines.append(line[:300])
    return "\n".join(lines[-15:])[-limit:]
