import json
import logging
import re
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from referencing import Registry

from cloudops_hub.catalog import catalog_schemas, duplicate_errors

SCHEMA_DIR = Path(__file__).resolve().parent.parent / "data" / "schemas"
SCHEMA_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
logger = logging.getLogger(__name__)
EXAMPLE_JSON_INSTRUCTIONS = (
    "Generate the requested JSON immediately; do not ask for parameter values. "
    "Preserve explicit user-supplied values exactly. For every missing string value use "
    "'example-' followed by its parameter name, for example group_name='example-group_name' "
    "or app_name='example-app_name'. These are illustrative placeholders, not real resources "
    "or verified permissions. Respect schema types, enums, formats, patterns and constraints: "
    "if that literal placeholder is invalid, use a clearly illustrative schema-valid example "
    "(such as ab-example-subscription, example.invalid or a documentation-only IP address). "
    "Use schema-valid enum members, numeric examples and false booleans where applicable; "
    "do not replace typed values with strings. Do not invent actual IDs, credentials, "
    "approved access, existing resource names or real defaults. Missing values do not make "
    "a supported template unsupported. Return no missing-value questions. "
    "Default to mandatory fields and requested conditional prerequisites only; include "
    "optional fields only when requested, including all supported fields for 'all parameters'. "
)


class SchemaNotFound(Exception):
    pass


def schema_validator(schema: dict) -> Draft202012Validator:
    """Allow local references only, with no network retrieval during validation."""

    def check_refs(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in {"$ref", "$dynamicRef"} and (
                    not isinstance(item, str) or not item.startswith("#")
                ):
                    raise ValueError("External schema references are not allowed")
                check_refs(item)
        elif isinstance(value, list):
            for item in value:
                check_refs(item)

    check_refs(schema)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema, registry=Registry())


def validated_fragment(schema: dict, values: dict, pointer: str) -> tuple[Any, str]:
    """Validate only the requested subtree, retaining local schema reference scope."""
    if not isinstance(pointer, str) or not pointer.startswith("/") or len(pointer) > 512:
        raise ValueError("A valid fragment pointer is required")
    validator = schema_validator(schema)
    node = schema
    value: Any = values
    tokens = pointer[1:].split("/")
    key = ""
    parent_is_array = False
    for raw in tokens:
        if re.search(r"~(?![01])", raw):
            raise ValueError("Invalid fragment pointer escape")
        key = raw.replace("~1", "/").replace("~0", "~")
        visited = set()
        while "$ref" in node:
            ref = node["$ref"]
            if ref in visited or not ref.startswith("#/") or len(node) != 1:
                raise ValueError("Unsupported schema reference; validate the enclosing object")
            visited.add(ref)
            node = schema
            for part in ref[2:].split("/"):
                node = node[part.replace("~1", "/").replace("~0", "~")]
        if any(
            name in node
            for name in (
                "allOf",
                "anyOf",
                "oneOf",
                "if",
                "$dynamicRef",
                "dependentSchemas",
                "dependentRequired",
                "not",
            )
        ):
            raise ValueError("Validate the enclosing object for this conditional schema")
        types = node.get("type", [])
        types = [types] if isinstance(types, str) else types
        if "object" in types:
            properties = node.get("properties", {})
            additional = node.get("additionalProperties")
            if (
                not isinstance(value, dict)
                or (key not in properties and not isinstance(additional, dict))
                or (
                    key not in properties
                    and any(
                        constraint in node for constraint in ("patternProperties", "propertyNames")
                    )
                )
            ):
                raise ValueError("Fragment property is not declared in the schema")
            parent_is_array = False
            node = properties[key] if key in properties else additional
            value = value[key]
        elif "array" in types:
            if not re.fullmatch(r"0|[1-9][0-9]*", key) or not isinstance(value, list):
                raise ValueError("Invalid fragment array index")
            parent_is_array = True
            node = node["items"]
            value = value[int(key)]
        else:
            raise ValueError("Fragment pointer extends beyond the declared schema")
    validator.evolve(schema=node).validate(value)
    if isinstance(value, (dict, list)):
        return value, pointer
    # A scalar alone omits the property name needed to apply the requested change.
    return ([value] if parent_is_array else {key: value}), pointer.rsplit("/", 1)[0] or "/"


def list_schemas() -> list[dict[str, Any]]:
    return [
        {
            "id": schema_id,
            "title": schema["title"],
            "source": schema["x-source"],
        }
        for schema_id, schema in catalog_schemas().items()
    ]


def load_schema(schema_id: str) -> tuple[dict[str, Any], str]:
    if not SCHEMA_ID_PATTERN.fullmatch(schema_id):
        raise SchemaNotFound(schema_id)
    catalog = catalog_schemas()
    if schema_id in catalog:
        schema = catalog[schema_id]
        return schema, schema["x-source"]
    files = {path.name: path for path in SCHEMA_DIR.glob("*.schema.json") if path.is_file()}
    path = files.get(f"{schema_id}.schema.json")
    if path is None:
        raise SchemaNotFound(schema_id)
    try:
        return json.loads(path.read_text(encoding="utf-8")), path.name
    except (OSError, json.JSONDecodeError) as exc:
        logger.exception("Unable to load request schema %s", path.name)
        raise RuntimeError(f"Unable to load request schema {path.name}") from exc


def required_fields(schema: dict[str, Any]) -> list[str]:
    return list(schema.get("required", []))


def validate_payload(schema_id: str, payload: Any) -> dict[str, Any]:
    schema, source = load_schema(schema_id)
    errors = sorted(
        Draft202012Validator(schema).iter_errors(payload),
        key=lambda error: (list(map(str, error.absolute_path)), error.message),
    )
    formatted_errors = []
    for error in errors:
        path = "/" + "/".join(map(str, error.absolute_path))
        if error.validator == "required" and isinstance(error.instance, dict):
            missing = [key for key in error.validator_value if key not in error.instance]
            formatted_errors.extend(
                {"path": f"{path.rstrip('/')}/{key}", "message": f"{key} is required"}
                for key in missing
            )
        else:
            formatted_errors.append({"path": path, "message": error.message})
    if not errors and schema_id.startswith("catalog-"):
        formatted_errors.extend(duplicate_errors(schema_id, payload))
    return {
        "valid": not formatted_errors,
        "errors": formatted_errors,
        "source": source,
        "warnings": schema.get("x-warnings", []),
        "target_repository": schema.get("x-target-repository", ""),
        "target_path": schema.get("x-target-path", ""),
        "sources": schema.get("x-sources", []),
    }
