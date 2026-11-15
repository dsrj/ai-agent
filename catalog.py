"""Reviewed subsets of repository input contracts, not inferred schemas from team examples."""

import copy
import json
import re
from pathlib import Path
from typing import Any

from cloudops_hub.catalog_contracts import expand_contracts

MANIFEST = Path(__file__).resolve().parent.parent / "data" / "catalog-sources.json"
SOURCE_REPOSITORIES = (
    "almbrand/landingzone-service-catalog",
    "almbrand/landingzone-service-catalog-code",
    "almbrand/corp-service-catalog-infrastructure",
    "almbrand/corp-level0-landingzone",
    "almbrand/devops-landingzone-service-catalog",
)
TEXT = {"type": "string", "minLength": 1, "pattern": r"^(?!.*[<>]).*\S.*$"}
FLAG = {"type": "string", "enum": ["true", "false"]}
NAME = {**TEXT, "pattern": r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$"}
SUBSCRIPTION = {**TEXT, "pattern": r"^ab-[a-z0-9]+(?:-[a-z0-9]+)*$"}


def obj(properties: dict, required: list[str]) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def array(items: dict, minimum: int = 1) -> dict:
    return {"type": "array", "items": items, "minItems": minimum, "uniqueItems": True}


def sources() -> list[dict]:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))["repositories"]


def contract_sources(repository: str, paths: list[str]) -> list[dict[str, str]]:
    entry = next(item for item in sources() if item["repository"] == repository)
    return [
        {
            "repository": repository,
            "ref": entry["ref"],
            "commit": entry["commit"],
            "path": path,
            "url": f"https://github.com/{repository}/blob/{entry['commit']}/{path}",
        }
        for path in paths
    ]


def catalog_schemas() -> dict[str, dict[str, Any]]:
    api_access = array(
        obj(
            {
                "api": TEXT,
                "roles": array(
                    obj(
                        {"type": {"enum": ["Application", "Delegated"]}, "id": TEXT},
                        ["type", "id"],
                    )
                ),
            },
            ["api", "roles"],
        ),
        0,
    )
    environment = obj(
        {
            "name": NAME,
            "subscription": SUBSCRIPTION,
            "spn_secret": FLAG,
            "app_registration_Secret": FLAG,
            "spn_secret_validity_in_days": {
                "type": "string",
                "pattern": r"^[1-9][0-9]*$",
            },
            "api_access": api_access,
        },
        ["name", "subscription"],
    )
    oidc = obj(
        {
            "OwnerIdentifier": TEXT,
            "github_windows_runners": FLAG,
            "instances": array(
                obj(
                    {
                        "github_repo_name": NAME,
                        "description": TEXT,
                        "template": NAME,
                        "Environment": array(environment),
                    },
                    ["github_repo_name", "Environment"],
                )
            ),
        },
        ["instances"],
    )
    groups = obj(
        {
            "instances": array(
                obj(
                    {
                        "group_name": NAME,
                        "description": TEXT,
                        "create_github_team": FLAG,
                    },
                    ["group_name"],
                )
            )
        },
        ["instances"],
    )
    application = obj(
        {
            "app_name": NAME,
            "app_registration_Secret": FLAG,
            "app_role_assignment_required": {"type": "boolean"},
            "api_access": api_access,
        },
        ["app_name"],
    )
    code = "almbrand/landingzone-service-catalog-code"
    definitions = {
        "catalog-repositories": (
            "Repository and environment request",
            obj({"OIDC": oidc}, ["OIDC"]),
            "<team>/<team>.json",
            {
                "OIDC": {
                    "instances": [
                        {
                            "github_repo_name": "",
                            "Environment": [{"name": "", "subscription": ""}],
                        }
                    ]
                }
            },
            contract_sources(
                code,
                [
                    "code/common/github.common.tf",
                    "code/githubspns/githubspns.locals.tf",
                ],
            ),
        ),
        "catalog-groups": (
            "Entra group request",
            obj({"groups": groups}, ["groups"]),
            "<team>/<team>_groups.json",
            {"groups": {"instances": [{"group_name": "", "description": ""}]}},
            contract_sources(code, ["code/common/groups.common.tf"]),
        ),
        "catalog-applications": (
            "Application / SPN request",
            obj({"auth": obj({"applications": array(application)}, ["applications"])}, ["auth"]),
            "<team>/<team>_spns.json",
            {"auth": {"applications": [{"app_name": "", "api_access": []}]}},
            contract_sources(code, ["code/common/webapp.common.tf"]),
        ),
    }
    result = {}
    for schema_id, (title, schema, target, template, provenance) in definitions.items():
        result[schema_id] = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            **schema,
            "title": f"{title} (reviewed subset)",
            "description": (
                "Repository-derived draft, not a deployment approval. Unsupported fields are "
                "rejected; consult the pinned consumer before extending this subset."
            ),
            "x-source": provenance[0]["url"],
            "x-sources": provenance,
            "x-target-repository": "almbrand/landingzone-service-catalog",
            "x-target-path": target,
            "x-template": template,
            "x-warnings": [
                "Draft only. Structural validation does not verify live names, permissions, "
                "quota, existing resources, tags, or organizational approval.",
                "Merge the entries into the existing target file manually; do not replace "
                "other teams' entries. The Hub never uploads, commits, or deploys this draft.",
                "Keep Environment, app_registration_Secret and string flags exactly as shown. "
                "Do not substitute JSON booleans for string flags.",
                "Supply actual values; no team identifiers, subscription names, role grants, "
                "or secrets are inferred from another team's example.",
            ],
        }
    return copy.deepcopy(expand_contracts(result))


def duplicate_errors(schema_id: str, payload: dict) -> list[dict[str, str]]:
    """Terraform map keys must remain unique even when array items differ."""
    errors = []

    def check(items: list, key: str, path: str) -> None:
        seen = set()
        for index, item in enumerate(items):
            value = str(item[key]).casefold()
            if schema_id == "catalog-applications" and key == "app_name":
                value = re.sub(r"\s+", "-", value)
            elif schema_id == "catalog-sftp" and key == "name":
                value = value.replace("-", "")
            if value in seen:
                errors.append({"path": f"{path}/{index}/{key}", "message": f"Duplicate {key}"})
            seen.add(value)

    if schema_id == "catalog-repositories":
        items = payload["OIDC"]["instances"]
        check(items, "github_repo_name", "/OIDC/instances")
        for index, item in enumerate(items):
            check(item["Environment"], "name", f"/OIDC/instances/{index}/Environment")
    else:
        schema = catalog_schemas().get(schema_id, {})
        path = schema.get("x-entry-path", [])
        identity = schema.get("x-identity-key")
        if identity:
            items = payload
            for part in path:
                items = items[part]
            check(items, identity, "/" + "/".join(path))
            if schema_id == "catalog-gsa":
                for index, item in enumerate(items):
                    check(
                        item["segments"], "azure_resource_name", f"/gsa/instances/{index}/segments"
                    )
            elif schema_id == "catalog-public-network-access":
                for index, item in enumerate(items):
                    check(
                        item["resources"],
                        "resource_type",
                        f"/publicnetworkaccess/instances/{index}/resources",
                    )
            elif schema_id == "catalog-pim":
                for index, item in enumerate(items):
                    keys = [
                        {
                            "terraform_key": role["name"]
                            + "-"
                            + (
                                role.get("scope", item["subscription"])
                                .rstrip("/")
                                .rsplit("/", 1)[-1]
                            )
                        }
                        for role in item["roles"]
                    ]
                    check(keys, "terraform_key", f"/policies/{index}/roles")
    return errors
