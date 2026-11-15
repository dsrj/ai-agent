"""Exact discovery and bounded navigation of an approved knowledge snapshot.

The graph is reference data, not a source of instructions or live Azure state.
"""

import json
import re
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

DISCOVERABLE_KINDS = frozenset(
    {
        "concept",
        "environment",
        "resource-type",
        "platform-component-type",
        "author-declared-route-resource",
        "resource-knowledge-bundle",
        "shared-topic-bundle",
    }
)
_load_lock = threading.Lock()
_validation_lock = threading.Lock()


def normalize(value: str) -> str:
    return " ".join(value.split()).casefold()


class GraphIndex:
    """Index source labels and aliases without inferring equivalent resources."""

    def __init__(self, snapshot: Any, ontology: dict[str, Any] | None = None):
        self.snapshot = snapshot
        self.ontology = ontology
        if ontology is not None:
            required_classes = {"Entity", "KnowledgeProfile", "Claim", "Relationship", "Evidence"}
            required_properties = {
                "aliases", "label", "kind", "taskAliases", "predicate", "subject",
                "object", "source", "path", "section", "text",
            }
            if not required_classes.issubset(ontology["classes"]) or not (
                required_properties.issubset(ontology["properties"])
            ):
                raise ValueError("Graph ontology does not support navigation")
        self._engine: Any = None
        self._lock = threading.Lock()
        self.aliases: dict[str, list[str]] = {}
        self.evidence_by_section: dict[tuple[str, str], list[dict[str, Any]]] = {}
        self.records_by_evidence: dict[str, list[dict[str, Any]]] = {}
        self.route_tasks: dict[str, list[str]] = {}
        self.ontology_inverse_index: dict[str, dict[str, list[dict[str, Any]]]] = {}
        if ontology is not None:
            for field in ("subject", "object"):
                inverse = ontology.get("inverses", {}).get(field)
                if inverse and inverse in ontology["properties"]:
                    self.ontology_inverse_index[inverse] = {}
        for identity, record in snapshot.records.items():
            if record["@type"] == "Relationship":
                for field, inverse in (ontology or {}).get("inverses", {}).items():
                    if inverse in self.ontology_inverse_index:
                        target = record.get(field)
                        if isinstance(target, str):
                            self.ontology_inverse_index[inverse].setdefault(target, []).append(
                                record
                            )
            if record["@type"] == "Evidence":
                key = (record.get("path", ""), record.get("section", ""))
                if all(key):
                    self.evidence_by_section.setdefault(key, []).append(record)
            if record["@type"] in ("Claim", "Relationship"):
                source = record.get("source", [])
                for evidence_id in source if isinstance(source, list) else [source]:
                    self.records_by_evidence.setdefault(evidence_id, []).append(record)
            if record["@type"] == "Claim" and record.get("routeId"):
                self.route_tasks.setdefault(record.get("subject", ""), []).extend(
                    record.get("taskAliases", [])
                )
            if record["@type"] != "KnowledgeProfile" and not (
                record["@type"] == "Entity" and record.get("kind") in DISCOVERABLE_KINDS
            ):
                continue
            for value in (record.get("label", ""), *record.get("aliases", [])):
                alias = normalize(value)
                if alias and identity not in self.aliases.setdefault(alias, []):
                    self.aliases[alias].append(identity)
        self.patterns = [
            (alias, re.compile(r"(?<!\w)" + re.escape(alias) + r"(?!\w)"))
            for alias in self.aliases
            if len(alias) >= 3
        ]

    def resolve(self, phrase: str) -> dict[str, Any]:
        identities = sorted(self.aliases.get(normalize(phrase), []))
        return {
            "status": "unmatched" if not identities else "resolved" if len(identities) == 1
            else "ambiguous",
            "matches": [
                {
                    "id": identity,
                    "label": self.snapshot.records[identity]["label"],
                    "type": self.snapshot.records[identity]["@type"],
                    "kind": self.snapshot.records[identity].get(
                        "profileKind", self.snapshot.records[identity].get("kind", "")
                    ),
                }
                for identity in identities
            ],
        }

    def phrases(self, question: str) -> list[str]:
        """Find exact aliases inside a question; never substitute fuzzy matches."""
        text = normalize(question)
        matches = []
        for alias, pattern in self.patterns:
            for match in pattern.finditer(text):
                matches.append((match.start(), match.end(), alias))
        matches.sort(key=lambda item: (-(item[1] - item[0]), item[0], item[2]))
        selected: list[tuple[int, int, str]] = []
        for match in matches:
            if not any(match[0] < other[1] and other[0] < match[1] for other in selected):
                selected.append(match)
        return list(dict.fromkeys(item[2] for item in sorted(selected)))

    def semantic_vocabulary(self) -> dict[str, list[str]]:
        """Return only graph-declared aliases and task phrases as interpretation candidates."""
        return {
            "entities": sorted(alias for alias, _ in self.patterns),
            "tasks": sorted({
                normalize(task)
                for tasks in self.route_tasks.values() for task in tasks
                if isinstance(task, str) and len(normalize(task)) >= 3
            }),
            "relationships": sorted({
                record["predicate"] for record in self.snapshot.records.values()
                if record["@type"] == "Relationship"
                and isinstance(record.get("predicate"), str) and record["predicate"]
            }),
            **({"ontology": self.ontology} if self.ontology else {}),
        }

    def evidence_context(
        self,
        documents: list[dict[str, Any]],
        *,
        offset: int = 0,
        snapshot_sha256: str | None = None,
    ) -> dict[str, Any]:
        """Find graph assertions supported by exact retrieved source passages.

        A search hit anchors graph navigation only when its path, heading and evidence
        text all agree. An indexed search hit alone is not proof of graph applicability.
        """
        from cloudops_knowledge_mcp.knowledge import Knowledge, Page

        source_ids = {
            "knowledge:github:almbrand/cloudops-openai-hub-knowledge-source",
            "knowledge:github:almbrand/cloudops-openai-hub-knowledge-source:admin",
        }
        found: dict[str, dict[str, Any]] = {}
        for document in documents:
            if document.get("source_id") not in source_ids:
                continue
            key = (document.get("source_path", ""), document.get("section", ""))
            content = " ".join(str(document.get("content", "")).split())
            for evidence in self.evidence_by_section.get(key, []):
                text = " ".join(str(evidence.get("text", "")).split())
                if len(text) < 12 or text not in content:
                    continue
                for record in self.records_by_evidence.get(evidence["@id"], []):
                    found[record["@id"]] = record
        with self._lock:
            if self._engine is None:
                self._engine = Knowledge(self.snapshot, "active")
            self._engine.content_truncated = False
            rows = [
                self._engine.evidenced(record)
                for _, record in sorted(found.items())
                if record["@id"] in self._engine.visible
            ]
            return self._engine.result(
                rows,
                Page(limit=50, offset=offset, snapshot_sha256=snapshot_sha256),
                status="ok" if rows else "not_found",
                warnings=[
                    "Only assertions anchored to exact retrieved source passages are included; "
                    "this is not a complete graph traversal or a live Azure check."
                ],
            ).model_dump()

    def _route_context(self, question: str) -> tuple[set[str], set[str]]:
        from cloudops_hub.answer_scope import _phrase_length
        route_ids = {
            identity for identity, record in self.snapshot.records.items()
            if record["@type"] == "Entity"
            and record.get("kind") == "author-declared-route-resource"
        }
        applicable_routes = {
            identity for identity in route_ids
            if _phrase_length(question, self.route_tasks.get(identity, []))
            and _phrase_length(
                question,
                [
                    self.snapshot.records[identity]["label"],
                    *self.snapshot.records[identity].get("aliases", []),
                ],
            )
        }
        return route_ids, applicable_routes

    def discover_question_context(self, question: str) -> dict[str, Any]:
        """Discover bounded graph assertions without depending on Search results."""
        route_ids, applicable_routes = self._route_context(question)
        discoveries = []
        for phrase in self.phrases(question):
            result = self.resolve(phrase)
            matches = [
                item for item in result["matches"]
                if item["id"] not in route_ids or item["id"] in applicable_routes
            ]
            discoveries.append((phrase, {
                "status": "unmatched" if not matches else "resolved" if len(matches) == 1
                else "ambiguous",
                "matches": matches,
            }))
        if len(discoveries) > 12:
            raise ValueError("Knowledge discovery exceeded its context budget")
        ambiguous_routes = {
            item["id"]
            for _, result in discoveries if result["status"] == "ambiguous"
            for item in result["matches"] if item["id"] in route_ids
        }
        applicable_routes -= ambiguous_routes
        results = []
        for _, match in discoveries:
            if match["status"] != "resolved":
                continue
            item = match["matches"][0]
            if item["type"] == "KnowledgeProfile":
                results.append(self.guidance(item["id"]))
            else:
                results.append(self.entity_context(item["id"]))
        rows: dict[str, dict[str, Any]] = {}
        for result in results:
            for item in result["items"]:
                if any(
                    item.get(field) in route_ids and item[field] not in applicable_routes
                    for field in ("subject", "object")
                ):
                    continue
                enriched = dict(item)
                for field in ("subject", "object"):
                    target = self.snapshot.records.get(item.get(field))
                    if target and target.get("label"):
                        enriched[f"{field}_label"] = target["label"]
                rows[item["@id"]] = enriched
        return {
            "items": list(rows.values()),
            "discoveries": [
                {"phrase": phrase, **result} for phrase, result in discoveries
            ],
            "truncated": any(result["truncated"] for result in results),
            "snapshot_commit": self.snapshot.commit,
            "snapshot_sha256": self.snapshot.digest,
            "knowledge_status": self.snapshot.knowledge_status,
        }

    def discover_ontology_relationships(
        self, discovery: dict[str, Any], predicates: list[str],
    ) -> dict[str, Any]:
        """Query existing relationship assertions using declared inverse properties."""
        from cloudops_knowledge_mcp.knowledge import Knowledge, Page

        if not predicates or not self.ontology_inverse_index:
            return {**discovery, "items": []}
        selected = {
            match["id"]
            for item in discovery["discoveries"] if item["status"] == "resolved"
            for match in item["matches"]
            if match["type"] == "Entity"
        }
        found = {
            row["@id"]: row
            for inverse in self.ontology_inverse_index.values()
            for identity in selected
            for row in inverse.get(identity, [])
            if row.get("predicate") in predicates
        }
        with self._lock:
            if self._engine is None:
                self._engine = Knowledge(self.snapshot, "active")
            self._engine.content_truncated = False
            rows = [
                self._engine.evidenced(row)
                for _, row in sorted(found.items()) if row["@id"] in self._engine.visible
            ]
            result = self._engine.result(rows, Page(limit=50)).model_dump()
        return {
            **discovery, "items": result["items"],
            "truncated": discovery["truncated"] or result["truncated"],
        }

    def question_context(self, question: str, documents: list[dict[str, Any]]) -> dict[str, Any]:
        """Gather bounded exact-match and source-anchored assertions without guessing identity."""
        return self.integrate_question_context(
            question, self.discover_question_context(question), documents,
        )

    def integrate_question_context(
        self, question: str, discovery: dict[str, Any], documents: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Add retrieved evidence to independently discovered graph assertions."""
        route_ids, applicable_routes = self._route_context(question)
        applicable_routes -= {
            item["id"]
            for result in discovery["discoveries"] if result["status"] == "ambiguous"
            for item in result["matches"] if item["id"] in route_ids
        }
        evidence = self.evidence_context(documents)
        rows: dict[str, dict[str, Any]] = {}
        for item in [*evidence["items"], *discovery["items"]]:
            if any(
                item.get(field) in route_ids and item[field] not in applicable_routes
                for field in ("subject", "object")
            ):
                continue
            enriched = dict(item)
            for field in ("subject", "object"):
                target = self.snapshot.records.get(item.get(field))
                if target and target.get("label"):
                    enriched[f"{field}_label"] = target["label"]
            rows[item["@id"]] = enriched
        return {
            **discovery, "items": list(rows.values()),
            "truncated": discovery["truncated"] or evidence["truncated"],
        }

    def integrate_semantic_context(
        self, original_question: str, semantic_question: str,
        original: dict[str, Any], semantic: dict[str, Any], documents: list[dict[str, Any]],
        relationship_types: list[str],
    ) -> dict[str, Any]:
        """Corroborate semantic-only graph matches against the retrieved source passages."""
        eligible = [
            doc for doc in documents
            if doc.get("review_status") == "approved"
            and doc.get("version") == self.snapshot.commit
        ]
        supported_ids = {row["@id"] for row in self.evidence_context(eligible)["items"]}
        supported = {
            **semantic,
            "items": [row for row in semantic["items"] if row["@id"] in supported_ids],
        }
        result = self.integrate_question_context(original_question, original, documents)
        supplemental = self.integrate_question_context(semantic_question, supported, eligible)
        combined = {row["@id"]: row for row in result["items"]}
        combined.update({row["@id"]: row for row in supplemental["items"]})
        result["items"] = list(combined.values())
        result["discoveries"] = supplemental["discoveries"]
        result["truncated"] = result["truncated"] or supplemental["truncated"]
        if relationship_types:
            result["items"].sort(key=lambda row: row.get("predicate") not in relationship_types)
        return result

    def guidance(
        self,
        profile_id: str,
        *,
        resource_id: str | None = None,
        offset: int = 0,
        snapshot_sha256: str | None = None,
    ) -> dict[str, Any]:
        from cloudops_knowledge_mcp.knowledge import Knowledge, Page, Selection

        with self._lock:
            if self._engine is None:
                self._engine = Knowledge(self.snapshot, "active")
            self._engine.content_truncated = False
            result = self._engine.guidance(
                Selection(profile_id=profile_id, resource_id=resource_id),
                Page(limit=50, offset=offset, snapshot_sha256=snapshot_sha256),
            )
            return result.model_dump()

    def entity_context(
        self, identity: str, *, offset: int = 0, snapshot_sha256: str | None = None
    ) -> dict[str, Any]:
        from cloudops_knowledge_mcp.knowledge import Knowledge, Page

        record = self.snapshot.records.get(identity)
        if not record or record["@type"] != "Entity" or record.get("kind") not in (
            DISCOVERABLE_KINDS
        ):
            raise ValueError("Knowledge entity is unavailable")
        with self._lock:
            if self._engine is None:
                self._engine = Knowledge(self.snapshot, "active")
            engine = self._engine
            engine.content_truncated = False
            claims = [engine.evidenced(row) for row in engine.claims({identity}, set())]
            adjacent = {
                row["@id"]: row
                for row in engine.outgoing.get(identity, []) + engine.incoming.get(identity, [])
            }
            edges = [engine.evidenced(row) for row in adjacent.values()]
            return engine.result(
                claims + edges,
                Page(limit=50, offset=offset, snapshot_sha256=snapshot_sha256),
                status="ok" if claims or edges else "not_found",
                warnings=[
                    "Only direct documented claims and relationships are included; "
                    "conditions and source status are not inferred.",
                ],
            ).model_dump()


def load_graph(directory: str) -> GraphIndex:
    # Avoid concurrent cold-start parses of the same large snapshot.
    with _load_lock:
        return _load_graph(directory)


@lru_cache(maxsize=1)
def _load_snapshot(directory: str) -> Any:
    from cloudops_knowledge_mcp.snapshot import parse_snapshot

    root = Path(directory)
    pin = json.loads((root / "snapshot-pin.json").read_bytes())
    snapshot = parse_snapshot((root / "snapshot.json").read_bytes(), pin["snapshot_sha256"])
    if snapshot.commit != pin["commit"] or snapshot.manifest_sha256 != pin["manifest_sha256"]:
        raise ValueError("Knowledge graph does not match the approved pin")
    return snapshot


@lru_cache(maxsize=1)
def load_semantic_graph(directory: str) -> GraphIndex:
    """Prepare source-declared terminology without applying the ontology."""
    with _load_lock:
        return GraphIndex(_load_snapshot(directory))


@lru_cache(maxsize=1)
def _load_graph(directory: str) -> GraphIndex:
    root = Path(directory)
    snapshot = _load_snapshot(directory)
    from cloudops_hub.graph_schema import load_validated_schema

    ontology = load_validated_schema(root, snapshot, validate_graph=False)
    return GraphIndex(snapshot, ontology)


def validate_retrieved_graph(directory: str, index: GraphIndex) -> None:
    """Validate the pinned graph after discovery and before using it as evidence."""
    with _validation_lock:
        _validate_retrieved_graph(directory, index)


@lru_cache(maxsize=1)
def _validate_retrieved_graph(directory: str, index: GraphIndex) -> None:
    from cloudops_hub.graph_schema import load_validated_schema

    load_validated_schema(Path(directory), index.snapshot)
