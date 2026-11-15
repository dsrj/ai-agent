"""Validate published source-backed procedures without changing the MCP contract."""

from typing import Any


def procedure_context(records: dict[str, dict[str, Any]], process: dict[str, Any]) -> tuple:
    def same_scope(item: dict[str, Any]) -> bool:
        left, right = process.get("assistants"), item.get("assistants")
        return (
            process.get("audience") == item.get("audience")
            and isinstance(left, list) and isinstance(right, list)
            and all(isinstance(value, str) for value in [*left, *right])
            and set(left) == set(right)
        )

    def evidence(record: dict[str, Any]) -> list[dict[str, Any]]:
        identities = record.get("source", [])
        if isinstance(identities, str):
            identities = [identities]
        if not isinstance(identities, list) or not identities:
            raise ValueError("Procedure source evidence is missing")
        result = []
        for identity in identities:
            item = records.get(identity, {})
            if (
                item.get("@type") != "Evidence" or not same_scope(item)
                or item.get("sourceVerificationStatus") != "verified"
                or not all(isinstance(item.get(key), str) and item[key] for key in (
                    "snapshotCommit", "path", "section", "text"
                ))
            ):
                raise ValueError("Procedure source evidence is invalid")
            result.append(item)
        return result

    if (
        process.get("@type") != "Entity"
        or process.get("kind") != "documented-resource-onboarding-process"
        or process.get("status") != "source-explicit-procedure"
    ):
        raise ValueError("Documented procedure is unavailable")
    links = [
        row for row in records.values()
        if row.get("@type") == "Relationship"
        and row.get("predicate") == "documents-resource-onboarding"
        and row.get("subject") == process.get("@id")
    ]
    if len(links) != 1:
        raise ValueError("Procedure route is not unique")
    alias_source = records.get(links[0].get("object"), {})
    routes = [
        row for row in records.values()
        if row.get("@type") == "Claim" and row.get("subject") == alias_source.get("@id")
        and isinstance(row.get("routeId"), str) and row["routeId"]
    ]
    canonical_links = [
        row for row in records.values()
        if row.get("@type") == "Relationship"
        and row.get("predicate") == "declares-canonical-resource"
        and row.get("subject") == alias_source.get("@id")
    ]
    if (
        alias_source.get("@type") != "Entity"
        or alias_source.get("kind") != "author-declared-route-resource"
        or len(routes) != 1 or len(canonical_links) != 1
    ):
        raise ValueError("Procedure route context is incomplete")
    canonical = records.get(canonical_links[0].get("object"), {})
    aliases, tasks = alias_source.get("aliases"), routes[0].get("taskAliases")
    if (
        canonical.get("@type") != "Entity"
        or not isinstance(aliases, list) or not aliases
        or not all(isinstance(item, str) and item.strip() for item in aliases)
        or not isinstance(tasks, list) or not tasks
        or not all(isinstance(item, str) and item.strip() for item in tasks)
        or any(not same_scope(item) for item in (
            links[0], alias_source, routes[0], canonical_links[0], canonical
        ))
    ):
        raise ValueError("Procedure route scope or aliases are invalid")
    steps = [
        row for row in records.values()
        if row.get("@type") == "Claim" and row.get("subject") == process.get("@id")
        and type(row.get("order")) is int
    ]
    if len(steps) < 2 or sorted(row["order"] for row in steps) != list(range(1, len(steps) + 1)):
        raise ValueError("Procedure order is incomplete")
    steps.sort(key=lambda row: row["order"])
    commits = {item["snapshotCommit"] for item in evidence(process)}
    if len(commits) != 1 or any(
        item["snapshotCommit"] not in commits for item in evidence(links[0])
    ):
        raise ValueError("Procedure route is not supported by the same source snapshot")
    step_evidence = []
    for step in steps:
        if (
            not same_scope(step) or step.get("status") != "source-explicit-procedure"
            or not isinstance(step.get("text"), str) or not step["text"]
        ):
            raise ValueError("Procedure step is invalid")
        sources = evidence(step)
        if any(item["snapshotCommit"] not in commits for item in sources) or not all(
            step["text"] in item["text"] for item in sources
        ):
            raise ValueError("Procedure step does not match pinned source evidence")
        step_evidence.extend(sources)
    return alias_source, routes[0], steps, step_evidence
