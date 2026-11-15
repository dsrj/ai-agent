"""Guide-owned answer contracts; no inferred resource or task labels."""

import json
import re

from cloudops_hub.sections import section_chunks

VERSION = 1
FIELDS = [
    "answer_id", "answer_resource", "answer_task", "answer_support",
    "answer_aliases", "answer_task_aliases", "answer_conditions", "scope_version",
]
PRESENTATION_FIELDS = [
    "answer_presentation_version", "answer_layout", "answer_disposition",
    "answer_canonical_id",
]
KEYS = {
    "id", "resource", "task", "support", "aliases", "task_aliases", "section", "conditions",
}
SLUG = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*")


def question_sections(body: str) -> set[str]:
    return {
        " > ".join(section.split(" > ")[:index + 1])
        for section, _ in section_chunks(body)
        for index, heading in enumerate(section.split(" > "))
        if heading.endswith("?") or heading.casefold().startswith("q:")
    }


def presentations(fields: dict, body: str) -> list[dict]:
    """Require exhaustive presentation coverage for an opted-in document."""
    if "answer_presentation_version" not in fields and "answer_presentations" not in fields:
        return []
    if (
        type(fields.get("answer_presentation_version")) is not int
        or fields["answer_presentation_version"] != VERSION
    ):
        raise ValueError("Unsupported answer_presentation_version")
    entries = fields.get("answer_presentations")
    if not isinstance(entries, list) or len(entries) > 500:
        raise ValueError("answer_presentations must be a list of at most 500 entries")
    seen = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {
            "section", "layout", "disposition", "canonical_answer_id",
        }:
            raise ValueError("Presentation has missing or unknown fields")
        if not isinstance(entry["section"], str) or entry["section"] in seen:
            raise ValueError("Invalid or duplicate presentation section")
        if not isinstance(entry["layout"], str) or entry["layout"] not in {
            "controls", "procedure", "explanation", "troubleshooting", "reference",
        }:
            raise ValueError("Invalid answer layout")
        if not isinstance(entry["disposition"], str) or entry["disposition"] not in {
            "direct", "duplicate", "composite", "reference-only", "incomplete", "over-budget",
        }:
            raise ValueError("Invalid answer disposition")
        canonical = entry["canonical_answer_id"]
        if not isinstance(canonical, str) or (
            entry["disposition"] == "duplicate"
            and (not SLUG.fullmatch(canonical) or len(canonical) > 120)
        ) or (entry["disposition"] != "duplicate" and canonical):
            raise ValueError("Only duplicate presentations reference a canonical answer id")
        seen.add(entry["section"])
    if seen != question_sections(body):
        raise ValueError("Presentation coverage must match every question section exactly")
    return entries


def contracts(fields: dict, body: str) -> list[dict]:
    if "answer_schema_version" not in fields and "answers" not in fields:
        return []
    if (
        type(fields.get("answer_schema_version")) is not int
        or fields["answer_schema_version"] != VERSION
    ):
        raise ValueError("Unsupported answer_schema_version")
    entries = fields.get("answers")
    if not isinstance(entries, list) or not entries or len(entries) > 200:
        raise ValueError("answers must contain 1 to 200 answer contracts")
    sections = question_sections(body)
    seen_ids, seen_routes, seen_sections = set(), set(), set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != KEYS:
            raise ValueError("Answer contract has missing or unknown fields")
        for key in ("id", "resource", "task"):
            if (
                not isinstance(entry[key], str) or not SLUG.fullmatch(entry[key])
                or len(entry[key]) > 120
            ):
                raise ValueError(f"Invalid answer {key}")
        if not isinstance(entry["support"], str) or entry["support"] not in {
            "supported", "conditional", "unsupported", "generic",
        }:
            raise ValueError("Invalid answer support")
        for key in ("aliases", "task_aliases", "conditions"):
            values = entry[key]
            if (
                not isinstance(values, list) or len(values) > 40
                or any(
                    not isinstance(value, str) or not value.strip() or len(value) > 250
                    or value != value.strip()
                    or re.search(r"[\x00-\x1f\x7f]", value)
                    for value in values
                )
                or len({value.casefold() for value in values}) != len(values)
            ):
                raise ValueError(f"Invalid answer {key}")
        if not entry["aliases"] or not entry["task_aliases"]:
            raise ValueError("Answer resource and task aliases must be explicit")
        section = entry["section"]
        if not isinstance(section, str) or section not in sections:
            raise ValueError("Answer section must exactly match an existing heading path")
        heading = section.rsplit(" > ", 1)[-1]
        if not (heading.endswith("?") or heading.casefold().startswith("q:")):
            raise ValueError("Answer contract must target a question heading")
        if entry["support"] == "conditional" and not entry["conditions"]:
            raise ValueError("Conditional answer requires conditions")
        route = (entry["resource"], entry["task"])
        if entry["id"] in seen_ids or route in seen_routes or section in seen_sections:
            raise ValueError("Duplicate answer id, route or section")
        seen_ids.add(entry["id"])
        seen_routes.add(route)
        seen_sections.add(section)
    return entries


def owning_question(section: str) -> str:
    parents = [
        " > ".join(section.split(" > ")[:index + 1])
        for index, heading in enumerate(section.split(" > "))
        if heading.endswith("?") or heading.casefold().startswith("q:")
    ]
    return parents[-1] if parents else ""


def presentation_metadata(section: str, entries: list[dict]) -> dict:
    entry = next((item for item in entries if item["section"] == owning_question(section)), None)
    return {
        "answer_presentation_version": VERSION if entry else 0,
        "answer_layout": entry["layout"] if entry else "",
        "answer_disposition": entry["disposition"] if entry else "",
        "answer_canonical_id": entry["canonical_answer_id"] if entry else "",
    }


def metadata(section: str, entries: list[dict]) -> dict:
    # Deepest question owns its chunks. Never leak a parent's route into a nested FAQ.
    target = owning_question(section)
    entry = next((entry for entry in entries if entry["section"] == target), None)
    if entry is None:
        return {
            "scope_version": 0, "answer_id": "", "answer_resource": "",
            "answer_task": "", "answer_support": "", "answer_aliases": [],
            "answer_task_aliases": [], "answer_conditions": [],
        }
    return {
        "scope_version": VERSION,
        **{
            f"answer_{name}": entry[key]
            for name, key in (
                ("id", "id"), ("resource", "resource"), ("task", "task"),
                ("support", "support"), ("aliases", "aliases"),
                ("task_aliases", "task_aliases"), ("conditions", "conditions"),
            )
        },
    }


def _phrase_length(text: str, phrases: list[str]) -> int:
    return max((
        len(phrase) for phrase in phrases
        if re.search(rf"(?<![\w./:@-]){re.escape(phrase)}(?![\w/:@-]|\.[\w-])", text, re.I)
    ), default=0)


class AmbiguousAnswerRoute(RuntimeError):
    """Multiple explicitly authored answers match the requested resource/task."""

    def __init__(
        self, message: str = "", *, meanings: list[str] | None = None,
        clarification: str = "",
    ):
        super().__init__(message)
        self.meanings = meanings or []
        self.clarification = clarification


class AnswerContractUnavailable(RuntimeError):
    """A declared answer cannot be served safely; never substitute synthesis."""


def routing_registry(search, scope: str, *, allow_empty: bool = False) -> list[dict]:
    records = []
    identities = {}
    for record in search.search(
        search_text="*",
        filter=f"{scope} and scope_version eq 1 and review_status eq 'approved'",
        select=FIELDS + PRESENTATION_FIELDS + [
            "source_id", "source_path", "version", "doc_id", "review_status",
        ],
    ):
        records.append(record)
        if len(records) > 10_000:
            raise AnswerContractUnavailable("Answer routing registry exceeded its safety limit")
        if (
            record.get("answer_presentation_version") != VERSION
            or record.get("answer_layout") not in {
                "controls", "procedure", "explanation", "troubleshooting", "reference",
            }
            or record.get("answer_disposition") not in {
                "direct", "duplicate", "composite", "reference-only", "incomplete", "over-budget",
            }
        ):
            raise AnswerContractUnavailable(
                "Answer routing presentation metadata requires a reimport",
            )
        if record.get("answer_support") != "generic" and record["answer_disposition"] != "direct":
            raise AnswerContractUnavailable("Declared answer route is not a complete direct answer")
        identifier = record.get("answer_id")
        if not isinstance(identifier, str) or not SLUG.fullmatch(identifier):
            raise AnswerContractUnavailable("Answer routing registry contains an invalid identity")
        identity = json.dumps({
            key: record.get(key) for key in [
                *FIELDS, *PRESENTATION_FIELDS, "source_id", "source_path", "version", "doc_id",
            ]
        }, sort_keys=True)
        if identifier in identities and identities[identifier] != identity:
            raise AnswerContractUnavailable(
                "Answer routing identity has conflicting indexed definitions",
            )
        identities[identifier] = identity
    if not records and not allow_empty:
        raise AnswerContractUnavailable("Answer routing registry is empty; refresh knowledge first")
    return records


def choose_route(
    question: str, records: list[dict], *, clarify_ambiguity: bool = False,
) -> dict | None:
    """Find an authored candidate; whole-request applicability still requires selection."""
    ranked = []
    for record in records:
        if record.get("answer_support") == "generic":
            continue
        resource = _phrase_length(question, record.get("answer_aliases") or [])
        task = _phrase_length(question, record.get("answer_task_aliases") or [])
        if resource and task:
            ranked.append(((resource, task), record))
    if not ranked:
        return None
    facets = re.findall(
        r"\b(?:why|rationale|gdpr|compliance|frameworks?|threats?|risks?|"
        r"terraform|bicep|provenance|history|exceptions?|troubleshoot\w*)\b",
        question, re.I,
    )
    if any(
        not _phrase_length(" ".join(record.get("answer_task_aliases") or []), [facet])
        for _, record in ranked for facet in facets
    ):
        return None
    choices = {record["answer_id"]: record for _, record in ranked}
    if not choices:
        return None
    # Ambiguous requests remain on the existing evidence path, never arbitrarily pick a route.
    if len(choices) != 1:
        if clarify_ambiguity:
            raise AmbiguousAnswerRoute("Multiple answer routes match the request")
        return None
    selected = next(iter(choices.values()))
    return selected
