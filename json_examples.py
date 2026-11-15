"""Schema-valid illustrative values; never look up or infer actual resources."""

import math
from copy import deepcopy
from typing import Any

from jsonschema import FormatChecker

from cloudops_hub.schemas import schema_validator

_MISSING = object()


def example_values(schema: dict, values: dict, *, all_parameters: bool = False) -> dict:
    validator = schema_validator(schema)
    formats = FormatChecker()

    def valid(node: dict, value: Any) -> bool:
        return validator.evolve(schema=node).is_valid(value)

    def combine(base: dict, extra: dict) -> dict:
        merged = {**base, **extra}
        merged["required"] = list(
            dict.fromkeys([*base.get("required", []), *extra.get("required", [])])
        )
        if "properties" in base or "properties" in extra:
            merged["properties"] = deepcopy(base.get("properties", {}))
            for key, child in extra.get("properties", {}).items():
                merged["properties"][key] = combine(merged["properties"].get(key, {}), child)
        return merged

    def fill(node: dict, value: Any = _MISSING, name: str = "value", depth: int = 0) -> Any:
        if depth > 40:
            raise ValueError("Example schema exceeds the supported nesting limit")
        if "$ref" in node:
            reference = node["$ref"]
            if not reference.startswith("#/"):
                raise ValueError("Example schema requires a local JSON pointer reference")
            target = schema
            for part in reference[2:].split("/"):
                target = target[part.replace("~1", "/").replace("~0", "~")]
            node = combine(target, {key: item for key, item in node.items() if key != "$ref"})
        for operator in ("oneOf", "anyOf"):
            if operator in node:
                base = {key: item for key, item in node.items() if key != operator}
                for branch in node[operator]:
                    try:
                        candidate = fill(combine(base, branch), value, name, depth + 1)
                    except ValueError:
                        continue
                    if valid(node, candidate):
                        return candidate
                raise ValueError(f"No compatible example alternative for {name}")
        kind = node.get("type")
        if isinstance(kind, list):
            kind = next((item for item in kind if item != "null"), "null")
        if kind == "object" or "properties" in node:
            if value is not _MISSING and not isinstance(value, dict):
                return deepcopy(value)
            result = deepcopy(value) if value is not _MISSING else {}
            properties = node.get("properties", {})
            for key, child in properties.items():
                if key in result or all_parameters or key in node.get("required", []):
                    result[key] = fill(child, result.get(key, _MISSING), key, depth + 1)
            additional = node.get("additionalProperties")
            if isinstance(additional, dict):
                for key in result.keys() - properties.keys():
                    result[key] = fill(additional, result[key], key, depth + 1)
                minimum = node.get("minProperties", 0)
                if all_parameters and not properties:
                    minimum = max(1, minimum)
                while len(result) < minimum:
                    key = "example-parameter" if not result else f"example-parameter-{len(result)}"
                    result[key] = fill(additional, name=name, depth=depth + 1)
            for condition in node.get("allOf", []):
                if "if" not in condition:
                    result = fill(
                        combine(node_without_conditions(node), condition), result, name, depth + 1
                    )
                    continue
                branch = condition.get("then" if valid(condition["if"], result) else "else", {})
                if branch:
                    result = fill(
                        combine(node_without_conditions(node), branch), result, name, depth + 1
                    )
            return result
        if kind == "array":
            if value is not _MISSING and not isinstance(value, list):
                return deepcopy(value)
            items = node.get("items", {})
            result = deepcopy(value) if value is not _MISSING else []
            minimum = node.get("minItems", 0)
            if all_parameters:
                minimum = max(1, minimum)
            minimum = min(minimum, node.get("maxItems", minimum))
            while len(result) < minimum:
                result.append(_MISSING)
            return [fill(items, item, name, depth + 1) for item in result]
        if value is not _MISSING and value != "":
            return deepcopy(value)
        if "const" in node:
            return deepcopy(node["const"])
        if "enum" in node:
            choices = node["enum"]
            for preferred in (False, "false", "Deny", "Reader", "disabled"):
                if any(type(item) is type(preferred) and item == preferred for item in choices):
                    return preferred
            return deepcopy(choices[0])
        if kind == "boolean":
            return False
        if kind in {"integer", "number"}:
            minimum = node.get("minimum", node.get("exclusiveMinimum", 0))
            multiple = node.get("multipleOf", 1)
            number = math.ceil(minimum / multiple) * multiple
            if number <= node.get("exclusiveMinimum", -math.inf):
                number += multiple
            return int(number) if kind == "integer" else number
        if kind == "null":
            return None
        if kind == "string":
            format_examples = {
                "uri": "https://example.invalid",
                "uri-reference": "https://example.invalid",
                "hostname": "example.invalid",
                "email": "example@example.invalid",
                "ipv4": "192.0.2.1",
                "ipv6": "2001:db8::1",
                "uuid": "00000000-0000-4000-8000-000000000000",
                "date": "2000-01-01",
                "date-time": "2000-01-01T00:00:00Z",
            }
            candidates = [
                f"example-{name}",
                f"ab-example-{name.replace('_', '-')}",
                "00000000-0000-4000-8000-000000000000",
                "1",
                "0",
                "example.invalid",
                "https://example.invalid",
                "example@example.invalid",
                "192.0.2.1",
                "2001:db8::1",
                "2000-01-01",
                "2000-01-01T00:00:00Z",
            ]
            if node.get("format") in format_examples:
                candidates.insert(0, format_examples[node["format"]])
            for candidate in candidates:
                if valid(node, candidate) and (
                    not node.get("format") or formats.conforms(candidate, node["format"])
                ):
                    return candidate
            raise ValueError(f"No schema-valid example string for {name}")
        raise ValueError(f"No supported example type for {name}")

    result = fill(schema, deepcopy(values))
    validator.validate(result)
    return result


def node_without_conditions(node: dict) -> dict:
    return {key: value for key, value in node.items() if key != "allOf"}
