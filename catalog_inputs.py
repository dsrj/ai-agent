"""Complete reviewed input inventories and their Landing Zone answer layout."""

import json
import re

from cloudops_hub.catalog import catalog_schemas
from cloudops_hub.catalog_contracts import RELEASE_COMMIT

RESOURCE_PATTERNS = (
    ("catalog-whitelist", r"\bwhitelist\b"),
    ("catalog-rulesets", r"\brulesets?\b"),
    ("catalog-group-members", r"\b(?:group[ -]members(?:hips?)?|existing[ -]groups?)\b"),
    ("catalog-sftp-network-rules", r"\bsftp\b.*\bnetwork\b|\bnetwork\b.*\bsftp\b"),
    ("catalog-tags", r"\btags?\b"),
    ("catalog-repositories", r"\b(?:oidc|repositor(?:y|ies)|githubenvs|github.environments?)\b"),
    ("catalog-groups", r"\b(?:ad.groups?|entra.groups?|groups?|groups\.json)\b"),
    ("catalog-applications", r"\b(?:spns?|applications?|enterprise.apps?|app.registrations?)\b"),
    ("catalog-identities", r"\b(?:idns|managed.identit(?:y|ies))\b"),
    ("catalog-certificates", r"\b(?:certs?|certificates?)\b"),
    ("catalog-onprem", r"\bon[ -]?prem(?:ises)?\b"),
    ("catalog-sftp", r"\bsftp\b"),
    ("catalog-nsp", r"\b(?:nsp|network.security.perimeter)\b"),
    ("catalog-public-network-access", r"\bpublic.network.access\b"),
    ("catalog-gsa", r"\b(?:gsa|global.secure.access)\b"),
    ("catalog-pim", r"\b(?:pim|privileged.identity)\b"),
)


def input_schema(question: str, history: list[dict]) -> dict | None:
    input_request = re.search(
        r"\b(?:create|configure|set.up|provision|inputs?|parameters?|fields?|properties|"
        r"opret|parametre|felter)\b", question, re.I,
    )
    if not input_request and not re.search(r"\badd\b", question, re.I):
        return None
    schemas = catalog_schemas()
    for text in [question, *(
        str(item.get("content", "")) for item in reversed(history)
        if item.get("role") == "user"
    )]:
        for schema_id, pattern in RESOURCE_PATTERNS:
            if not input_request and not re.search(
                rf"\badd\s+(?:(?:an?|new|another|the|my|our|team)\s+)*(?:{pattern})",
                question, re.I,
            ):
                continue
            if re.search(pattern, text, re.I):
                return {"id": schema_id, **schemas[schema_id]}
    return None


def input_fields(schema: dict) -> list[dict]:
    fields = []
    entry_path = ".".join(schema["x-entry-path"])

    seen = set()

    def walk(node, path="", parent_required=True):
        for variant in node.get("oneOf", []):
            walk(variant, path, False)
        for name, child in node.get("properties", {}).items():
            pointer = f"{path}.{name}" if path else name
            required_here = name in node.get("required", [])
            required = parent_required and required_here
            properties = child.get("properties") or child.get("items", {}).get("properties")
            wrapper = pointer == entry_path or entry_path.startswith(pointer + ".")
            if not wrapper and pointer not in seen:
                seen.add(pointer)
                fields.append({
                    "path": pointer, "name": name, "required": required,
                    "required_in_parent": required_here,
                    "type": child.get("type", "value"),
                    "parent_constraints": {
                        key: node[key] for key in ("if", "then", "else", "allOf", "anyOf", "oneOf")
                        if key in node
                    },
                    **{key: child[key] for key in (
                        "default", "description", "enum", "if", "then", "else", "allOf", "anyOf",
                        "oneOf", "not", "minimum", "maximum", "minItems", "maxItems",
                    ) if key in child},
                })
            nested = child.get("items", child)
            if properties or "oneOf" in nested:
                # Entries in a supplied collection follow their own required-field contract.
                walk(nested, pointer, True if wrapper else required)
            additional = child.get("additionalProperties")
            if isinstance(additional, dict) and additional.get("properties"):
                walk(additional, pointer + ".<map-key>", False)

    walk(schema)
    return fields


def contract_document(schema: dict) -> dict:
    """Supply the whole parent contract, independently of top-k search truncation."""
    return {
        "id": f"reviewed-contract:{schema['id']}:{RELEASE_COMMIT}",
        "title": schema["title"] + " - complete reviewed input contract",
        "assistant": "landing-zone",
        "source_id": "reviewed:service-catalog-contract",
        "source_path": schema["x-sources"][0]["path"],
        "source_url": schema["x-sources"][0]["url"],
        "source_kind": "reviewed_contract",
        "version": RELEASE_COMMIT,
        "content": (
            "Reviewed repository snapshot, not live verification. Required-in-parent means "
            "required only when that parent object is supplied. Optional-parent prerequisites "
            "and all conditional requirements must be explained, never promoted to unconditional "
            "requirements. Complete parent resource: " + schema["title"] + ".\n"
            + schema["description"] + "\n"
            + json.dumps(input_fields(schema), ensure_ascii=False)
        ),
    }


def input_prompt(schema: dict) -> str:
    return (
        "\nCOMPLETE RESOURCE INPUTS: Use the complete reviewed parent contract supplied in "
        "context, including related repository inputs for a GitHub environment question, "
        "not just githubenvs. Return one input table under the exact heading "
        "'### Catalog input inventory'. Its columns MUST be 'Parameter | Explanation'. "
        "For application validation, Parameter must use each exact path in the supplied "
        "contract, with one row per path, including every optional field. The application "
        "will display leaf names and Required: Yes/No, so do not prefix explanations with "
        "requirement labels. Explain purpose, defaults and conditional requirements from "
        "evidence; if purpose is undocumented, explicitly say that rather than inventing it. "
        "Do not omit or merge same-named fields from different parent blocks. "
        "This complete inventory overrides mandatory-only or brevity limits for internal "
        "output; the application controls which rows are visible. For a creation how-to, "
        "also give the full documented numbered procedure, with step 1 opening the team file "
        "and step 2 adding the resource definition. Do not write a separate input-review step; "
        "the application inserts step 3. Do not generate any JSON or code examples. "
        f"Resource contract ID: {schema['id']}."
    )


def format_inputs(answer: str, schema: dict, *, procedure: bool) -> str:
    section = re.search(
        r"(?ms)^### Catalog input inventory\s*\n(.*?)(?=^#{1,3} |\Z)", answer,
    )
    if not section:
        raise ValueError("Complete catalog input inventory is missing")
    explanations = {}
    for line in section[1].splitlines():
        # Structured grounding appends its citation markers after the final table row.
        line = re.sub(r"\s+(?:\[\d+\]\s*)+$", "", line)
        match = re.fullmatch(r"\s*\|\s*`?([^|`]+)`?\s*\|\s*(.*?)\s*\|\s*", line)
        if match and match[1].strip() not in {"Parameter", "---"}:
            key = match[1].strip()
            if key in explanations or not match[2].strip():
                raise ValueError("Duplicate or empty catalog parameter explanation")
            explanations[key] = match[2].strip()
    fields = input_fields(schema)
    if set(explanations) != {field["path"] for field in fields}:
        raise ValueError("Catalog input inventory does not cover the complete resource")

    def table(rows, *, possible=False):
        lines = ["| Parameter | Explanation |", "| --- | --- |"]
        for field in rows:
            explanation = explanations[field["path"]]
            if schema["id"] == "catalog-groups" and field["name"] == "group_name":
                explanation = (
                    field["description"]
                    if possible
                    else "Names the group."
                )
            if field["required_in_parent"] and not field["required"]:
                explanation += " Required when its parent is supplied."
            parent = field["path"].rsplit(".", 1)[0] if "." in field["path"] else ""
            entry_path = ".".join(schema["x-entry-path"])
            if parent == entry_path:
                parent = ""
            elif parent.startswith(entry_path + "."):
                parent = parent[len(entry_path) + 1:]
            meaning = f"{explanation} **Required:** {'Yes' if field['required'] else 'No'}"
            if parent:
                meaning = f"Parent: `{parent}`. " + meaning
            lines.append(f"| `{field['name']}` | {meaning} |")
        return "\n".join(lines)

    required = table([field for field in fields if field["required"]])
    possible = table(fields, possible=True)
    body = (answer[:section.start()] + answer[section.end():]).strip()
    if not procedure:
        return body + "\n\n" + possible
    steps = list(re.finditer(r"(?m)^\d+\.\s+", body))
    if len(steps) < 2:
        raise ValueError("Catalog creation procedure is missing its first two steps")
    insertion = steps[2].start() if len(steps) > 2 else len(body)
    before, after = body[:insertion].rstrip(), body[insertion:]
    before += "\n\n" + "\n".join("   " + line for line in required.splitlines())
    disclosure = (
        "3. **Review possible inputs:** Expand to see the available parameters and what "
        "they mean.\n\n   ```possible-inputs\n"
        + "\n".join("   " + line for line in possible.splitlines()) + "\n   ```\n\n"
    )
    after = re.sub(r"(?m)^(\d+)\.\s+", lambda match: f"{int(match[1]) + 1}. ", after)
    return before + "\n\n" + disclosure + after
