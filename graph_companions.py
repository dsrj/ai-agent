"""Verify explicitly declared diagram companions against independently indexed evidence."""

import logging
import re
from typing import Any

from cloudops_hub.graph_diagram import _same_scope, _validate_sources, render_sequence
from cloudops_hub.graph_procedures import procedure_context

logger = logging.getLogger(__name__)
POLICY = "explicit-process-companion-evidence-v1"
SOURCE = "knowledge:github:almbrand/cloudops-openai-hub-knowledge-source"


def _record_sources(record: dict[str, Any], records: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    _validate_sources(record, records)
    evidence = [records[identity] for identity in record["source"]]
    if any(
        not re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256", "")))
        or not isinstance(item.get("path"), str) or not item["path"]
        or not isinstance(item.get("section"), str) or not item["section"]
        for item in evidence
    ):
        raise ValueError("Companion evidence provenance is incomplete")
    return evidence


def _read_sources(search: Any, scope: str, paths: set[str]) -> dict[str, list[dict[str, Any]]]:
    result = {}
    for path in sorted(paths):
        escaped = path.replace("'", "''")
        rows = list(search.search(
            search_text="*",
            filter=(f"{scope} and source_id eq '{SOURCE}' and source_path eq '{escaped}' "
                    "and review_status eq 'approved'"),
            select=["source_id", "source_path", "section", "chunk_number", "version",
                    "doc_id", "review_status", "answer_section_body"],
            top=201,
        ))
        if not rows or len(rows) > 200:
            raise ValueError("Companion source is unavailable or exceeds its reconstruction limit")
        if any(
            row.get("source_id") != SOURCE or row.get("source_path") != path
            or row.get("review_status") != "approved" or not row.get("doc_id")
            or not re.fullmatch(r"[0-9a-f]{40}", str(row.get("version", "")))
            or type(row.get("chunk_number")) is not int
            or not isinstance(row.get("section"), str)
            or not isinstance(row.get("answer_section_body"), str)
            for row in rows
        ):
            raise ValueError("Companion source metadata is incomplete")
        rows.sort(key=lambda row: row["chunk_number"])
        if (
            [row["chunk_number"] for row in rows] != list(range(len(rows)))
            or len({row["version"] for row in rows}) != 1
            or len({row["doc_id"] for row in rows}) != 1
            or sum(len(row["answer_section_body"]) for row in rows) > 200_000
        ):
            raise ValueError("Companion source reconstruction is incomplete")
        result[path] = rows
    return result


def _quote_matches(quote: str, evidence: dict[str, Any], rows: dict[str, list[dict[str, Any]]]) -> bool:
    section_rows = [
        row for row in rows[evidence["path"]] if row["section"] == evidence["section"]
    ]
    bodies = [row["answer_section_body"] for row in section_rows if row["answer_section_body"]]
    if len(bodies) != 1:
        return False
    body = bodies[0].strip()
    text = quote.strip()
    first, separator, remainder = text.partition("\n")
    heading = re.fullmatch(r"#{1,6}\s+(.+?)\s*#*", first)
    if heading:
        if heading.group(1) != evidence["section"].rsplit(" > ", 1)[-1]:
            return False
        text = remainder.strip() if separator else ""
    return bool(text) and text in body


def companion_for_process(
    records: dict[str, dict[str, Any]], process_id: str, search: Any, scope: str,
) -> dict[str, Any] | None:
    """Return one verified companion; missing or conflicting evidence never implies a link."""
    links = [
        row for row in records.values()
        if row.get("@type") == "Relationship" and row.get("kind") == "process-diagram-companion"
        and row.get("subject") == process_id
    ]
    if not links:
        return None
    if len(links) != 1:
        raise ValueError("Companion selection is ambiguous")
    link = links[0]
    process = records.get(process_id, {})
    target = records.get(link.get("object"), {})
    procedure_context(records, process)
    if (
        link.get("predicate") != "has-companion-diagram" or link.get("companionPolicy") != POLICY
        or link.get("status") != "unverified-copilot-interpretation"
        or link.get("interpretationVerificationStatus") != "not-assessed"
        or link.get("liveVerificationStatus") != "not-assessed"
        or link.get("sourceVerificationStatus") != "verified"
        or link.get("conditions") != process.get("conditions")
        or not _same_scope(link, process) or not _same_scope(link, target)
        or target.get("sourceSnapshotCommit") != link.get("sourceSnapshotCommit")
        or any(key in link for key in ("diagramId", "edgeId", "fromNode", "toNode"))
    ):
        raise ValueError("Companion contract is inconsistent")
    link_evidence = _record_sources(link, records)
    if (
        len(link_evidence) != 1 or len(link["sourceText"]) != 1
        or link["sourceText"][0] != link_evidence[0]["text"]
    ):
        raise ValueError("Companion applicability evidence is incomplete")
    diagram = render_sequence(records, link["object"])
    members = [target, *(
        row for row in records.values()
        if row.get("diagramId") == link["object"] and row.get("kind") in {"diagram-node", "diagram-edge"}
    )]
    member_evidence = [(record, _record_sources(record, records)) for record in members]
    paths = {item["path"] for item in link_evidence}
    paths.update(item["path"] for _, evidence in member_evidence for item in evidence)
    rows = _read_sources(search, scope, paths)
    if not _quote_matches(link["sourceText"][0], link_evidence[0], rows):
        raise ValueError("Companion applicability was not corroborated")
    for record, evidence in member_evidence:
        for quote in record["sourceText"]:
            if not any(
                quote in item["text"] and _quote_matches(quote, item, rows) for item in evidence
            ):
                raise ValueError("Companion sequence evidence was not corroborated")
    diagram["notes"] = list(dict.fromkeys([
        *diagram.get("notes", []), *link["conditions"], link["sourceText"][0],
    ]))
    logger.info(
        "Companion corroborated graph_snapshot=%s indexed_versions=%s",
        link["sourceSnapshotCommit"],
        sorted({row["version"] for source_rows in rows.values() for row in source_rows}),
    )
    return diagram
