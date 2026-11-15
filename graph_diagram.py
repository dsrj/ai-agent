"""Select and project source-backed sequences without generating connections."""

import re
import unicodedata
from typing import Any

from cloudops_hub.diagram_contract import MAX_EDGES, MAX_NODES, component_label, icon_ids

SUPPORTED_SELECTION_POLICY = "exact-query-entity-task-source-unique-v1"
SOURCE_TOPIC_SELECTION_POLICY = "exact-source-topic-task-unique-v1"
SEQUENCE_PREDICATES = frozenset({
    "precedes",
    "grants-azp-access-through-subscription-group",
    "is-read-by-hourly-mapping-job",
    "maps-azp-account-to-ab-account",
    "added-to-environment-matched-gsa-sign-in-group",
    "assigned-to-matching-enterprise-application",
    "contains-profile",
    "associated-with-profile",
    "request-rule-from",
})
UNSUPPORTED_PROJECTION_INTENT = re.compile(
    r"\b(?:actual|currently?|deployed|deployment|existing|live|real|"
    r"enforced|enforcement|packet|topology|timing|when|ready|readiness)\b"
    r"|\b(?:how long|network path|traffic path)\b",
    re.IGNORECASE,
)
def _same_scope(left: dict[str, Any], right: dict[str, Any]) -> bool:
    a, b = left.get("assistants"), right.get("assistants")
    return (
        left.get("audience") == right.get("audience")
        and isinstance(a, list) and isinstance(b, list)
        and all(isinstance(item, str) for item in [*a, *b])
        and set(a) == set(b)
    )


def _contains_term(question: str, term: str) -> bool:
    if not isinstance(term, str) or not term.strip():
        return False
    phrase = " ".join(unicodedata.normalize("NFKC", term).casefold().split())
    return bool(re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", question))


def _transposed_alias_windows(question: str, alias: str) -> list[tuple[int, int]]:
    phrase = " ".join(unicodedata.normalize("NFKC", alias).casefold().split())
    if not re.fullmatch(r"\w+(?: \w+)+", phrase):
        return []
    words = phrase.split()
    tokens = re.findall(r"\w+", question)
    matches = []
    for start in range(len(tokens) - len(words) + 1):
        actual = tokens[start:start + len(words)]
        differences = [(a, b) for a, b in zip(actual, words, strict=True) if a != b]
        if len(differences) != 1 or not any(
            a == b and len(b) >= 3 for a, b in zip(actual, words, strict=True)
        ):
            continue
        observed, expected = differences[0]
        if len(expected) < 5 or len(observed) != len(expected):
            continue
        if any(
            observed == expected[:i] + expected[i + 1] + expected[i] + expected[i + 2:]
            for i in range(len(expected) - 1)
            if expected[i] != expected[i + 1]
        ):
            matches.append((start, start + len(words)))
    return matches


def _safe_transposed_alias(question: str, aliases: list[str], tasks: list[str]) -> bool:
    tokens = re.findall(r"\w+", question)
    filler = {
        "a", "an", "the", "how", "to", "do", "i", "can", "you", "please",
        "what", "is", "are", "steps", "step", "for", "of", "draw", "show",
        "create", "diagram", "add", "enable", "connect", "through", "access",
    }
    task_words = set(re.findall(r"\w+", " ".join(tasks).casefold()))
    for alias in aliases:
        for start, end in _transposed_alias_windows(question, alias):
            if all(word in filler or word in task_words for word in tokens[:start] + tokens[end:]):
                return True
    return False


def select_sequence(
    records: dict[str, dict[str, Any]], question: str, documents: list[dict[str, Any]]
) -> dict[str, Any]:
    """Resolve only uniquely matched, explicitly declared discovery policies."""
    normalized = " ".join(unicodedata.normalize("NFKC", question).casefold().split())
    if UNSUPPORTED_PROJECTION_INTENT.search(normalized):
        return {"status": "unsupported", "selection": None}
    candidates = []
    matching_contexts = 0
    matching_topics = 0
    for sequence in records.values():
        if (
            sequence.get("@type") != "Entity"
            or sequence.get("kind") != "source-backed-sequence"
            or sequence.get("audience") not in {"guide", "admin"}
            or not isinstance(sequence.get("assistants"), list)
            or "landing-zone" not in sequence["assistants"]
            or sequence.get("status") != "unverified-copilot-interpretation"
            or sequence.get("sourceVerificationStatus") != "verified"
            or sequence.get("interpretationVerificationStatus") != "not-assessed"
            or sequence.get("liveVerificationStatus") != "not-assessed"
        ):
            continue
        if sequence.get("queryResolverPolicy") == SOURCE_TOPIC_SELECTION_POLICY:
            evidence_id = sequence.get("querySourceEvidence")
            sources = sequence.get("source")
            evidence = records.get(evidence_id, {}) if isinstance(evidence_id, str) else {}
            topics, tasks = sequence.get("queryTopicPhrases"), sequence.get("queryTaskPhrases")
            if (
                not isinstance(sources, list) or evidence_id not in sources
                or not isinstance(topics, list) or not topics
                or not isinstance(tasks, list) or not tasks
                or not all(isinstance(item, str) and item.strip() for item in [*topics, *tasks])
                or not isinstance(evidence.get("text"), str)
                or evidence.get("@type") != "Evidence"
                or not _same_scope(evidence, sequence)
                or evidence.get("snapshotCommit") != sequence.get("sourceSnapshotCommit")
                or evidence.get("sourceVerificationStatus") != "verified"
                or not all(isinstance(evidence.get(key), str) and evidence[key] for key in (
                    "path", "section",
                ))
                or any(key in sequence for key in (
                    "queryEntity", "queryAliasSource", "queryAliases", "queryTaskAliases",
                    "queryProfileId", "queryRouteId", "queryRouteCanonicalResource",
                    "queryAmbiguousTerms", "queryMatchPolicy", "queryAmbiguityPolicy",
                ))
            ):
                continue
            phrases = [" ".join(unicodedata.normalize("NFKC", item).casefold().split())
                       for item in [*topics, *tasks]]
            evidence_text = " ".join(
                unicodedata.normalize("NFKC", evidence["text"]).casefold().split()
            )
            if (
                len(set(phrases)) != len(phrases)
                or any(
                    _contains_term(topic, task) or _contains_term(task, topic)
                    for topic in phrases[:len(topics)] for task in phrases[len(topics):]
                )
                or not all(_contains_term(evidence_text, phrase) for phrase in phrases)
            ):
                continue
            if not any(_contains_term(normalized, phrase) for phrase in topics):
                continue
            matching_topics += 1
            if not any(_contains_term(normalized, phrase) for phrase in tasks):
                continue
            matching_contexts += 1
            quotes = sequence.get("sourceText")
            if (
                isinstance(quotes, list)
                and any(
                    isinstance(quote, str) and len(quote.strip()) >= 12
                    and quote in evidence["text"]
                    and any(
                        document.get("source_id") in {
                            "knowledge:github:almbrand/cloudops-openai-hub-knowledge-source",
                            "knowledge:github:almbrand/cloudops-openai-hub-knowledge-source:admin",
                        }
                        and document.get("source_path") == evidence.get("path")
                        and document.get("section") == evidence.get("section")
                        and quote in str(document.get("content", ""))
                        for document in documents
                    )
                    for quote in quotes
                )
            ):
                candidates.append({"sequence_id": sequence["@id"]})
            continue
        canonical_id = sequence.get("queryRouteCanonicalResource")
        canonical = records.get(canonical_id)
        query_entity = records.get(sequence.get("queryEntity"))
        alias_source = records.get(sequence.get("queryAliasSource"))
        aliases = sequence.get("queryAliases")
        tasks = sequence.get("queryTaskAliases")
        profile_id = sequence.get("queryProfileId")
        route_id = sequence.get("queryRouteId")
        if (
            not canonical or canonical.get("@type") != "Entity"
            or not query_entity or query_entity.get("@type") != "Entity"
            or not alias_source or alias_source.get("@type") != "Entity"
            or alias_source.get("kind") != "author-declared-route-resource"
            or not isinstance(query_entity.get("label"), str)
            or not isinstance(aliases, list) or not isinstance(tasks, list)
            or not isinstance(profile_id, str) or not isinstance(route_id, str)
            or sequence.get("queryResolverPolicy") != SUPPORTED_SELECTION_POLICY
            or any(key in sequence for key in (
                "querySourceEvidence", "queryTopicPhrases", "queryTaskPhrases",
            ))
        ):
            continue
        context_label = query_entity["label"]
        if not _contains_term(normalized, context_label):
            continue
        profiles = [
            item for item in records.values()
            if item.get("@type") == "KnowledgeProfile" and item.get("profileId") == profile_id
        ]
        routes = [
            item for item in records.values()
            if item.get("@type") == "Claim" and item.get("routeId") == route_id
        ]
        if len(profiles) != 1 or len(routes) != 1:
            continue
        route = routes[0]
        links = [
            item for item in records.values()
            if item.get("@type") == "Relationship"
            and item.get("predicate") == "declares-canonical-resource"
            and item.get("subject") == alias_source.get("@id")
            and item.get("object") == canonical_id
        ]
        if (
            route.get("subject") != alias_source.get("@id")
            or aliases != alias_source.get("aliases")
            or tasks != route.get("taskAliases")
            or len(links) != 1
        ):
            continue
        if any(not _same_scope(item, sequence) for item in (
            canonical, query_entity, alias_source, profiles[0], route, links[0]
        )):
            continue
        matching_topics += 1
        if not any(_contains_term(normalized, task) for task in tasks):
            continue
        matching_contexts += 1
        anchored = False
        quotes = sequence.get("sourceText", [])
        if not isinstance(quotes, list):
            continue
        for evidence_id in sequence.get("source", []):
            evidence = records.get(evidence_id, {})
            text = evidence.get("text", "")
            if (
                not isinstance(text, str) or len(text) < 12
                or evidence.get("@type") != "Evidence"
                or not _same_scope(evidence, sequence)
                or evidence.get("snapshotCommit") != sequence.get("sourceSnapshotCommit")
                or evidence.get("sourceVerificationStatus") != sequence.get(
                    "sourceVerificationStatus"
                )
            ):
                continue
            supported_quotes = [
                quote for quote in quotes
                if isinstance(quote, str) and len(quote.strip()) >= 12 and quote in text
            ]
            for document in documents:
                if (
                    document.get("source_id") in {
                        "knowledge:github:almbrand/cloudops-openai-hub-knowledge-source",
                        "knowledge:github:almbrand/cloudops-openai-hub-knowledge-source:admin",
                    }
                    and document.get("source_path") == evidence.get("path")
                    and document.get("section") == evidence.get("section")
                    and any(
                        quote in str(document.get("content", ""))
                        for quote in supported_quotes
                    )
                ):
                    anchored = True
                    break
            if anchored:
                break
        if anchored:
            candidates.append({
                "sequence_id": sequence["@id"],
                "profile_id": profile_id,
                "route_id": route_id,
                "canonical_resource_id": canonical_id,
            })
    if matching_contexts > 1 or (matching_contexts and matching_topics > 1):
        return {"status": "ambiguous", "selection": None}
    if not candidates:
        return {
            "status": "unmatched", "selection": None,
            **({"context_present": True} if matching_contexts else {}),
        }
    return {"status": "resolved", "selection": candidates[0]}


def _procedure(
    records: dict[str, dict[str, Any]], process: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    from cloudops_hub.graph_procedures import procedure_context

    context = procedure_context(records, process)
    if len(context[2]) > MAX_NODES or len(context[2]) - 1 > MAX_EDGES:
        raise ValueError("Procedure exceeds the diagram budget")
    return context


def select_procedure(
    records: dict[str, dict[str, Any]], question: str, documents: list[dict[str, Any]]
) -> dict[str, Any]:
    """Select a uniquely matched published route and exact retrieved procedure evidence."""
    normalized = " ".join(unicodedata.normalize("NFKC", question).casefold().split())
    candidates = []
    matching_contexts = 0
    for process in records.values():
        if (
            process.get("@type") != "Entity"
            or process.get("kind") != "documented-resource-onboarding-process"
            or process.get("status") != "source-explicit-procedure"
            or process.get("audience") not in {"guide", "admin"}
            or not isinstance(process.get("assistants"), list)
            or "landing-zone" not in process["assistants"]
        ):
            continue
        try:
            alias_source, route, steps, evidence = _procedure(records, process)
        except ValueError:
            continue
        aliases = alias_source["aliases"]
        exact = any(_contains_term(normalized, alias) for alias in aliases)
        transposed = any(_transposed_alias_windows(normalized, alias) for alias in aliases)
        if not ((exact or transposed) and any(
            _contains_term(normalized, task) for task in route["taskAliases"]
        )):
            continue
        matching_contexts += 1
        if not exact and not _safe_transposed_alias(normalized, aliases, route["taskAliases"]):
            continue
        anchored = any(
            document.get("source_id") in {
                "knowledge:github:almbrand/cloudops-openai-hub-knowledge-source",
                "knowledge:github:almbrand/cloudops-openai-hub-knowledge-source:admin",
            }
            and document.get("source_path") == item["path"]
            and document.get("section") == item["section"]
            and any(
                step["text"] in item["text"]
                and step["text"] in str(document.get("content", ""))
                for step in steps
            )
            for item in evidence for document in documents
        )
        if anchored:
            canonical = next(
                row["object"] for row in records.values()
                if row.get("@type") == "Relationship"
                and row.get("predicate") == "declares-canonical-resource"
                and row.get("subject") == alias_source["@id"]
            )
            candidates.append({
                "process_id": process["@id"], "route_id": route["routeId"],
                "canonical_resource_id": canonical,
            })
    if matching_contexts > 1 or len(candidates) > 1:
        return {"status": "ambiguous", "selection": None}
    if not candidates:
        return {
            "status": "unmatched", "selection": None,
            **({"context_present": True} if matching_contexts else {}),
        }
    return {"status": "resolved", "selection": candidates[0]}


def render_procedure(records: dict[str, dict[str, Any]], process_id: str) -> dict[str, Any]:
    """Project consecutive documented steps, not graph edges or deployed infrastructure."""
    process = records.get(process_id, {})
    if (
        process.get("@type") != "Entity"
        or process.get("kind") != "documented-resource-onboarding-process"
        or process.get("status") != "source-explicit-procedure"
    ):
        raise ValueError("Documented procedure is unavailable")
    alias_source, route, steps, _ = _procedure(records, process)
    notes = [
        "Arrows show the order of documented procedure steps, not a network topology.",
        "This is source guidance, not a check of the current Azure or identity state.",
    ]
    for item in [process, route, *steps]:
        conditions = item.get("conditions", [])
        if not isinstance(conditions, list) or not all(isinstance(c, str) for c in conditions):
            raise ValueError("Procedure conditions are invalid")
        notes.extend(conditions)
    labels = []
    for step in steps:
        heading = re.match(r"^\d+\.\s+\*\*(.+?)\*\*", step["text"])
        labels.append(heading.group(1).rstrip(".") if heading else f'Step {step["order"]}')
        notes.append(step["text"])
    return {
        "title": f'Documented procedure: {alias_source["aliases"][0]} - {route["taskAliases"][0]}',
        "kind": "procedure",
        "zones": [],
        "nodes": [
            {"id": f"n{i}", "label": label, "sublabel": "", "role": "component",
             "zone": "", "icon": ""}
            for i, label in enumerate(labels, 1)
        ],
        "edges": [
            {"from": f"n{i}", "to": f"n{i + 1}", "label": "Next documented step"}
            for i in range(1, len(steps))
        ],
        "notes": list(dict.fromkeys(notes)),
    }


def render_relationships(
    records: dict[str, dict[str, Any]], result: dict[str, Any], *, dependencies_only: bool = True
) -> dict[str, Any]:
    """Render explicit graph relationships, never as an inferred traffic or procedure flow."""
    if result.get("truncated"):
        raise ValueError("The selected graph relationships are incomplete")
    relationships = [
        row for row in result.get("items", [])
        if row.get("@type") == "Relationship"
        and (not dependencies_only or row.get("predicate") == "depends-on")
    ]
    if not relationships or len(relationships) > MAX_EDGES:
        raise ValueError("No complete bounded dependency graph is available")
    identities: dict[str, str] = {}
    nodes = []
    edges = []
    notes = [(
        "Each arrow means the source depends on the target; it is not a traffic flow "
        "or a live infrastructure check."
        if dependencies_only else
        "Each arrow shows the source-declared relationship and direction, not a traffic flow "
        "or a live infrastructure check."
    )]
    for row in relationships:
        source = records.get(row.get("@id"), {})
        source_refs = source.get("source", [])
        if isinstance(source_refs, str):
            source_refs = [source_refs]
        evidence = row.get("evidence")
        if (
            not isinstance(source_refs, list) or not source_refs
            or not isinstance(evidence, list) or len(evidence) != len(source_refs)
            or any(
                not isinstance(item, dict)
                or item != {
                    key: value for key, value in records.get(identity, {}).items()
                    if key not in {"assistants", "members"}
                }
                for item, identity in zip(evidence, source_refs, strict=False)
            )
            or not isinstance(row.get("status"), str)
            or not isinstance(row.get("predicate"), str) or not row["predicate"]
            or any(source.get(field) != row.get(field) for field in (
                "@type", "subject", "object", "predicate", "status", "condition", "conditions",
            ))
        ):
            raise ValueError("Dependency evidence or status is unavailable")
        for field in ("subject", "object"):
            identity = row.get(field)
            entity = records.get(identity)
            if not entity or entity.get("@type") != "Entity" or not component_label(
                entity.get("label")
            ):
                raise ValueError("Dependency endpoint is unavailable")
            if identity not in identities:
                if len(nodes) == MAX_NODES:
                    raise ValueError("Dependency graph exceeds its node budget")
                identities[identity] = f"n{len(nodes) + 1}"
                nodes.append({
                    "id": identities[identity], "label": entity["label"], "sublabel": "",
                    "role": "component", "zone": "", "icon": "",
                })
        if row["subject"] == row["object"]:
            raise ValueError("Dependency edge cannot point to itself")
        edges.append({
            "from": identities[row["subject"]], "to": identities[row["object"]],
            "label": row["predicate"],
        })
        label = (
            f'{records[row["subject"]]["label"]} {row["predicate"]} '
            f'{records[row["object"]]["label"]}'
        )
        if row["status"] != "verified":
            notes.append(f"{label}: interpretation is not verified.")
        conditions = row.get("conditions", [])
        if not isinstance(conditions, list) or not all(
            isinstance(item, str) for item in conditions
        ):
            raise ValueError("Dependency conditions are invalid")
        if isinstance(row.get("condition"), str):
            conditions = [*conditions, row["condition"]]
        elif row.get("condition") is not None:
            raise ValueError("Dependency conditions are invalid")
        notes.extend(f"{label}: {condition}" for condition in conditions)
    if len(set((edge["from"], edge["to"]) for edge in edges)) != len(edges):
        raise ValueError("Dependency graph contains duplicate edges")
    return {
        "title": "Documented dependencies" if dependencies_only else "Documented relationships",
        "zones": [], "nodes": nodes,
        "edges": edges, "notes": list(dict.fromkeys(notes)),
    }


def _validate_sources(record: dict[str, Any], records: dict[str, dict[str, Any]]) -> None:
    sources = record.get("source")
    quotes = record.get("sourceText")
    if not isinstance(sources, list) or not sources or not isinstance(quotes, list) or not quotes:
        raise ValueError("Diagram evidence is incomplete")
    evidence = []
    for identity in sources:
        item = records.get(identity)
        if (
            not item
            or item.get("@type") != "Evidence"
            or not _same_scope(item, record)
            or item.get("snapshotCommit") != record.get("sourceSnapshotCommit")
            or item.get("sourceVerificationStatus") != record.get("sourceVerificationStatus")
            or not isinstance(item.get("text"), str)
        ):
            raise ValueError("Diagram evidence does not match the sequence scope")
        evidence.append(item["text"])
    if (
        not all(isinstance(quote, str) and quote for quote in quotes)
        or any(not any(quote in text for text in evidence) for quote in quotes)
        or any(not any(quote in text for quote in quotes) for text in evidence)
    ):
        raise ValueError("Diagram evidence quotations do not match")


def render_sequence(records: dict[str, dict[str, Any]], sequence_id: str) -> dict[str, Any]:
    """Render one selected sequence exactly; reject broken links rather than repairing them."""
    sequence = records.get(sequence_id)
    if (
        not sequence
        or sequence.get("@type") != "Entity"
        or sequence.get("kind") != "source-backed-sequence"
        or sequence.get("diagramId") != sequence_id
    ):
        raise ValueError("Diagram sequence is unavailable")
    members = [record for record in records.values() if record.get("diagramId") == sequence_id]
    nodes = [item for item in members if item.get("kind") == "diagram-node"]
    edges = [item for item in members if item.get("kind") == "diagram-edge"]
    if not 2 <= len(nodes) <= MAX_NODES or not 1 <= len(edges) <= MAX_EDGES:
        raise ValueError("Diagram sequence exceeds the supported size")
    for entries in (nodes, edges):
        if any(type(item.get("order")) is not int for item in entries) or sorted(
            item["order"] for item in entries
        ) != list(range(1, len(entries) + 1)):
            raise ValueError("Diagram sequence ordering is invalid")
        entries.sort(key=lambda item: item["order"])
    for record in [sequence, *nodes, *edges]:
        if (
            not _same_scope(record, sequence)
            or record.get("status") != sequence.get("status")
            or record.get("liveVerificationStatus") != sequence.get("liveVerificationStatus")
            or record.get("interpretationVerificationStatus") != sequence.get(
                "interpretationVerificationStatus"
            )
            or record.get("sourceSnapshotCommit") != sequence.get("sourceSnapshotCommit")
            or record.get("sourceVerificationStatus") != sequence.get("sourceVerificationStatus")
            or not isinstance(record.get("status"), str) or not record["status"]
            or not isinstance(record.get("liveVerificationStatus"), str)
            or not record["liveVerificationStatus"]
            or not isinstance(record.get("conditions", []), list)
            or not all(isinstance(item, str) for item in record.get("conditions", []))
        ):
            raise ValueError("Diagram sequence scope or status is inconsistent")
        _validate_sources(record, records)
        if not set(record["source"]).issubset(set(sequence["source"])):
            raise ValueError("Diagram member cites evidence outside the sequence")
    if any(item.get("@type") != "Entity" for item in nodes) or any(
        item.get("@type") != "Relationship" for item in edges
    ):
        raise ValueError("Diagram member types are invalid")
    node_ids = {item["@id"]: f"n{index}" for index, item in enumerate(nodes, 1)}
    node_order = {item["@id"]: item["order"] for item in nodes}
    if len(node_ids) != len(nodes):
        raise ValueError("Diagram sequence contains duplicate nodes")
    visible_nodes = []
    notes = []
    for item in nodes:
        label = item.get("label")
        icon = item.get("iconId", "")
        if not component_label(label) or (icon and icon not in icon_ids()):
            raise ValueError("Diagram node presentation is invalid")
        visible_nodes.append({
            "id": node_ids[item["@id"]], "label": label, "sublabel": "",
            "role": "component", "zone": "", "icon": icon,
        })
        notes.extend(f"{label}: {condition}" for condition in item.get("conditions", []))
    visible_edges = []
    for item in edges:
        start, end = item.get("subject"), item.get("object")
        label = item.get("label")
        if (
            start not in node_ids or end not in node_ids or start == end
            or node_order.get(start, 0) >= node_order.get(end, 0)
            or item.get("fromNode") != start or item.get("toNode") != end
            or not isinstance(label, str) or not label
            or item.get("predicate") not in SEQUENCE_PREDICATES
        ):
            raise ValueError("Diagram edge is invalid")
        visible_edges.append({"from": node_ids[start], "to": node_ids[end], "label": label})
        notes.extend(f"{label}: {condition}" for condition in item.get("conditions", []))
    for key in ("label", "context", "orderingSemantics"):
        if not isinstance(sequence.get(key), str) or not sequence[key]:
            raise ValueError("Diagram sequence context is incomplete")
    notes = [
        sequence["context"], sequence["orderingSemantics"], *sequence.get("conditions", []), *notes,
    ]
    if sequence["status"] != "verified":
        notes.insert(0, "This interpretation has not been verified.")
    if sequence["liveVerificationStatus"] != "verified":
        notes.insert(1, "The current Azure or identity state has not been checked.")
    return {
        "title": sequence["label"], "zones": [], "nodes": visible_nodes,
        "edges": visible_edges, "notes": list(dict.fromkeys(notes)),
    }
