"""Input contracts reviewed against the service catalog's release Terraform consumers."""

from typing import Any

RELEASE_COMMIT = "eaf932defa4cc11c7ba226adebff229cd27c0d18"
CODE_REPOSITORY = "almbrand/landingzone-service-catalog-code"
EXAMPLES = {
    "catalog-group-members": "appgw/appgw_group_members.json",
    "catalog-identities": "appgw/appgw_idns.json",
    "catalog-certificates": "appgw/appgw_certs.json",
    "catalog-tags": "appgw/appgw_tags.json",
    "catalog-onprem": "ai/ai_onprem_access.json",
    "catalog-sftp": "batch/batch_sftp.json",
    "catalog-sftp-network-rules": "cloudops/cloudops_sftp_network_rules.json",
    "catalog-nsp": "cloudops/cloudops_nsp.json",
    "catalog-public-network-access": "cloudops/cloudops_publicnetworkaccess.json",
    "catalog-gsa": "intgr/intgr_gsa.json",
    "catalog-pim": "intgr/intgr_pim.json",
    "catalog-rulesets": "ai/ai_github_rulesets.json",
    "catalog-whitelist": "cloudops/whitelist_adgroups.json",
}


def expand_contracts(schemas: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    from cloudops_hub.catalog import FLAG, NAME, SUBSCRIPTION, TEXT, array, obj

    boolean = {"type": "boolean"}
    strings = array(TEXT, 0)
    number = {"type": "integer", "minimum": 1}
    days = {"oneOf": [number, {"type": "string", "pattern": "^[1-9][0-9]*$"}]}

    def fields(names: str, value: dict) -> dict:
        return {name: dict(value) for name in names.split()}

    def default(value: dict, fallback: Any) -> dict:
        return {**value, "default": fallback}

    def collection(root: str, key: str, item: dict, settings: dict | None = None) -> dict:
        return obj({root: obj({key: array(item), **(settings or {})}, [key])}, [root])

    def scaffold(schema: dict) -> Any:
        if schema.get("type") == "object":
            return {key: scaffold(schema["properties"][key]) for key in schema.get("required", [])}
        if schema.get("type") == "array":
            return [scaffold(schema["items"])] if schema.get("minItems", 0) else []
        return ""

    def add(
        schema_id: str,
        title: str,
        schema: dict,
        suffix: str,
        path: tuple[str, ...],
        consumers: list[str],
        key: str | None = None,
        note: str = "",
    ) -> None:
        provenance = [
            {
                "repository": CODE_REPOSITORY,
                "ref": "release",
                "commit": RELEASE_COMMIT,
                "path": source,
                "url": f"https://github.com/{CODE_REPOSITORY}/blob/{RELEASE_COMMIT}/{source}",
            }
            for source in consumers
        ]
        existing = schemas.get(schema_id, {})
        schemas[schema_id] = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            **schema,
            "title": title,
            "description": (
                "Reviewed JSON input fields, not derived Terraform resource attributes. "
                "Live consumers are checked before generation. " + note
            ),
            "x-source": provenance[0]["url"],
            "x-sources": provenance,
            "x-target-repository": "almbrand/landingzone-service-catalog",
            "x-target-path": (suffix if "/" in suffix else f"<team>/<team>{suffix}.json"),
            "x-entry-path": list(path),
            "x-identity-key": key,
            "x-template": existing.get("x-template", scaffold(schema)),
            "x-example-path": EXAMPLES.get(schema_id),
            "x-warnings": existing.get(
                "x-warnings",
                [
                    "Draft only; live permissions, existing resources and deployment approval "
                    "are not verified. Merge manually into the existing file.",
                    "The Hub never uploads, commits or deploys generated JSON.",
                ],
            ),
        }

    repo = schemas["catalog-repositories"]["properties"]["OIDC"]
    instance = repo["properties"]["instances"]["items"]
    instance["properties"]["Environment"]["minItems"] = 0
    environment = instance["properties"]["Environment"]["items"]
    environment["properties"].update(
        {
            "OwnerIdentifier": TEXT,
            "app_registration_Secret_validity_in_days": default(days, "720"),
            "spn_secret_validity_in_days": default(days, "720"),
            "prevent_self_review": default(boolean, False),
            "deployment_branches_protected": boolean,
            **fields("reviewers_users reviewers_groups keyvault_secrets custom_secrets", strings),
        }
    )
    gate = obj(
        {
            "name": TEXT,
            "include_admin_team": default(FLAG, "false"),
            **fields("reviewers_users reviewers_groups", strings),
        },
        ["name"],
    )
    shared = {
        **fields(
            "reviewers_users reviewers_groups custom_branch_protection_name "
            "custom_repo_secrets github_app_to_repos",
            strings,
        ),
        "manual_gate_githubenv": array(gate, 0),
        "ruleset_config": TEXT,
        "OwnerIdentifier": TEXT,
    }
    instance["properties"].update(
        {
            **shared,
            "skip_custom_branch_protection": default(FLAG, "false"),
            "storage_container_name": TEXT,
            "access_to_other_teams": array(
                obj(
                    {
                        "role": {"enum": ["pull", "triage", "push", "maintain", "admin"]},
                        "teams": array(TEXT),
                    },
                    ["role", "teams"],
                ),
                0,
            ),
        }
    )
    reviews = obj(
        {
            "required_approving_review_count": {
                "oneOf": [
                    {"type": "integer", "minimum": 0},
                    {"type": "string", "pattern": "^[0-9]+$"},
                ]
            },
            **fields(
                "require_last_push_approval dismiss_stale_reviews require_code_owner_reviews",
                boolean,
            ),
        },
        [],
    )
    repo["properties"].update(
        {
            **shared,
            "default_template": NAME,
            "default_backend_config": default(FLAG, "false"),
            "default_backend_config_use_OIDC": default(FLAG, "false"),
            "github_windows_runners": default(FLAG, "false"),
            "custom_branch_protection": array(
                obj(
                    {
                        "name": TEXT,
                        "pattern": TEXT,
                        **fields(
                            "allows_deletions enforce_admins restrict_pushes_blocks_creations", FLAG
                        ),
                        "required_pull_request_reviews": reviews,
                    },
                    ["name", "pattern"],
                ),
                0,
            ),
        }
    )
    add(
        "catalog-repositories",
        "GitHub repositories and environments",
        obj({"OIDC": repo}, ["OIDC"]),
        "",
        ("OIDC", "instances"),
        [
            "code/common/github.common.tf",
            "code/githubspns/githubspns.locals.tf",
            "code/githubenvs/githubenvs.locals.tf",
            "code/branchprotection/branchprotection.locals.tf",
            "code/branchprotection/githubbranchprotection.resources.tf",
            "code/backend/backend.locals.tf",
            "code/devops/devops.locals.tf",
        ],
        "github_repo_name",
        "Environment subscription names refer to existing landing-zone subscriptions. "
        "Linux runners are hard-coded enabled by release; github_linux_runners is not an "
        "effective input. Repository visibility is fixed internal, not a JSON parameter.",
    )

    members = {
        **fields(
            "include_explicit_idns include_explicit_app_spns include_explicit_github_spns "
            "filter_github_spns_azure_subscription_name",
            strings,
        ),
        **fields("include_all_idns include_github_spns include_app_spns", FLAG),
        "include_other_spns": {
            **strings,
            "description": "Service-principal display names; object IDs are resolved at apply.",
        },
    }
    group = obj(
        {
            "group_name": {
                **NAME,
                "description": (
                    "The supplied group_name is always prefixed with ab-<team>- "
                    "when forming the Entra group display name."
                ),
            },
            "description": default(TEXT, "default"),
            "create_github_team": default(FLAG, "false"),
            "access_level": default(TEXT, "restricted"),
            "allowed_teams": strings,
            "databricks_scim_Provisioning": default(FLAG, "false"),
            **members,
        },
        ["group_name"],
    )
    add(
        "catalog-groups",
        "Entra groups",
        collection("groups", "instances", group),
        "_groups",
        ("groups", "instances"),
        ["code/common/groups.common.tf", "code/members/members.locals.tf"],
        "group_name",
    )
    add(
        "catalog-group-members",
        "Membership of existing Entra groups",
        collection("groups", "instances", obj({"group_name": TEXT, **members}, ["group_name"])),
        "_group_members",
        ("groups", "instances"),
        ["code/members/members.locals.tf"],
        "group_name",
        "Use the existing full group display name; membership requires owner-team permission.",
    )

    app = schemas["catalog-applications"]["properties"]["auth"]["properties"]["applications"][
        "items"
    ]
    redirects = obj({"redirect_uris": strings}, ["redirect_uris"])
    scope = obj(
        {
            **fields(
                "admin_consent_description admin_consent_display_name id type "
                "user_consent_description user_consent_display_name value",
                TEXT,
            ),
            "enabled": boolean,
        },
        [],
    )
    app["properties"].update(
        {
            "app_name": TEXT,
            "identifier_uris": strings,
            "Allow_public_client_flows": default(boolean, False),
            "app_role_assignment_required": default(boolean, True),
            "app_registration_Secret": default(FLAG, "false"),
            "spn_secret_validity_in_days": default(days, "720"),
            "app_registration_Secret_validity_in_days": default(days, "720"),
            "group_membership_claims": strings,
            "AADgroups": strings,
            "web": array(
                obj(
                    {
                        **redirects["properties"],
                        "homepage_url": TEXT,
                        "implicit_grant": obj(
                            fields(
                                "access_token_issuance_enabled id_token_issuance_enabled", boolean
                            ),
                            [],
                        ),
                    },
                    ["redirect_uris"],
                ),
                0,
            ),
            "public_client": array(redirects, 0),
            "single_page_application": array(redirects, 0),
            "approles": array(
                obj(
                    {
                        "default_role": FLAG,
                        "enabled": boolean,
                        "allowed_member_types": array({"enum": ["User", "Application"]}),
                        **fields("description display_name value id", TEXT),
                        "AADgroups": strings,
                    },
                    [],
                ),
                0,
            ),
            "api": {
                **array(
                    obj(
                        {
                            "known_client_applications": strings,
                            "mapped_claims_enabled": boolean,
                            "requested_access_token_version": {"enum": [1, 2]},
                            "oauth2_permission_scope": array(scope, 0),
                        },
                        ["oauth2_permission_scope"],
                    )
                ),
                "maxItems": 1,
            },
            "optional_claims": array(
                obj(
                    {
                        "saml2_token": obj(
                            {
                                "additional_properties": strings,
                                "essential": boolean,
                                "name": TEXT,
                                "source": {"type": ["string", "null"]},
                            },
                            ["additional_properties", "essential", "name", "source"],
                        )
                    },
                    ["saml2_token"],
                ),
                0,
            ),
            "feature_tags": array(
                obj(
                    {
                        "feature_tags": obj(
                            fields("enterprise gallery custom_single_sign_on hide", boolean), []
                        )
                    },
                    ["feature_tags"],
                ),
                0,
            ),
        }
    )
    add(
        "catalog-applications",
        "Applications and service principals",
        collection("auth", "applications", app),
        "_spns",
        ("auth", "applications"),
        [
            "code/common/webapp.common.tf",
            "code/webapp/webapp.resources.tf",
            "code/approles/approles.locals.tf",
            "code/approles/approles.resources.tf",
        ],
        "app_name",
    )
    add(
        "catalog-identities",
        "Managed identities",
        collection("idns", "instances", obj({"name": NAME, "allowed_teams": strings}, ["name"])),
        "_idns",
        ("idns", "instances"),
        ["code/common/idns.common.tf", "code/idns/idns.locals.tf"],
        "name",
    )
    add(
        "catalog-certificates",
        "Certificates",
        collection(
            "certs",
            "instances",
            obj(
                {
                    "name": NAME,
                    "subject": TEXT,
                    "dns_names": strings,
                    "validity_in_months": default(number, 12),
                    "allowed_teams": strings,
                    "server_cert": default(boolean, True),
                    "client_cert": default(boolean, True),
                },
                ["name"],
            ),
        ),
        "_certs",
        ("certs", "instances"),
        ["code/certs/certs.locals.tf"],
        "name",
    )
    add(
        "catalog-tags",
        "Owner identifier and resource tags",
        collection(
            "config",
            "tags",
            obj(
                {
                    "OwnerIdentifier": TEXT,
                    "tags": {
                        **array(
                            {"type": "object", "minProperties": 1, "additionalProperties": TEXT}
                        ),
                        "maxItems": 1,
                    },
                },
                ["OwnerIdentifier", "tags"],
            ),
        ),
        "_tags",
        ("config", "tags"),
        ["code/githubspns/githubspns.locals.tf"],
        "OwnerIdentifier",
        "Tag names are map keys; GITpath and Environment are overwritten for GitHub SPNs.",
    )
    add(
        "catalog-onprem",
        "On-premises IP-group access",
        obj(
            {
                "onprem": obj(
                    {
                        "description": TEXT,
                        "cidrs": {
                            **array(
                                {
                                    "type": "object",
                                    "minProperties": 1,
                                    "additionalProperties": strings,
                                    "description": (
                                        "Existing IP-group name mapped to IP addresses/CIDRs."
                                    ),
                                }
                            ),
                            "maxItems": 1,
                        },
                    },
                    ["cidrs"],
                )
            },
            ["onprem"],
        ),
        "_onprem_access",
        ("onprem",),
        [
            "code/common/firewall.common.tf",
            "code/onpremaccess/onpremaccess.locals.tf",
            "code/onpremaccess/onpremaccess.resources.tf",
        ],
        note="IP-group names must already exist; bare IP addresses are treated as /32.",
    )
    access = array(
        obj(
            {
                "role": {
                    "enum": [
                        "Reader",
                        "Storage Blob Delegator",
                        "Storage Blob Data Reader",
                        "Storage Blob Data Owner",
                        "Storage Blob Data Contributor",
                    ]
                },
                "ad_groups": strings,
                "spns": strings,
            },
            ["role"],
        ),
        0,
    )
    add(
        "catalog-sftp",
        "SFTP users and containers",
        collection(
            "sftp",
            "instances",
            obj(
                {
                    "name": NAME,
                    "ssh_public_key": strings,
                    "password": default(FLAG, "true"),
                    "blob_container_access": access,
                },
                ["name"],
            ),
            {"blob_container_access": access},
        ),
        "_sftp",
        ("sftp", "instances"),
        ["code/sftp/sftp.locals.tf"],
        "name",
        "ssh_public_key contains existing vault key names or generate-new, not private keys. "
        "password is an enable flag, never a password value.",
    )
    add(
        "catalog-sftp-network-rules",
        "Shared SFTP network rules",
        obj(
            {
                "network_rules": obj(
                    {
                        "bypass": strings,
                        "default_action": {"enum": ["Allow", "Deny"]},
                        "ip_list": array(
                            obj(
                                {
                                    "name": TEXT,
                                    "ips": {
                                        "type": ["array", "null"],
                                        "items": {"type": "string"},
                                    },
                                    "subnets": {
                                        "type": ["array", "null"],
                                        "items": {"type": "string"},
                                    },
                                    "private_link_access": {
                                        **array(
                                            obj(
                                                {"resource_id": TEXT, "tenant_id": TEXT},
                                                ["resource_id"],
                                            ),
                                            0,
                                        ),
                                        "type": ["array", "null"],
                                    },
                                },
                                ["ips", "subnets", "private_link_access"],
                            )
                        ),
                    },
                    ["bypass", "default_action", "ip_list"],
                )
            },
            ["network_rules"],
        ),
        "cloudops/cloudops_sftp_network_rules.json",
        ("network_rules",),
        ["code/sftp/networkrules.tf"],
        note="Shared cloudops-only storage account network policy, not a per-team setting.",
    )
    add(
        "catalog-nsp",
        "Network security perimeter rules",
        collection(
            "nsp",
            "instances",
            obj(
                {
                    "name": NAME,
                    "fqdns": strings,
                    "ip_cidrs": strings,
                    "service_tags": strings,
                    "environment": array(TEXT),
                },
                ["name", "fqdns", "environment"],
            ),
        ),
        "_nsp",
        ("nsp", "instances"),
        ["code/nsp/nsp.locals.tf", "code/nsp/nsp.resources.tf"],
        "name",
        "fqdns creates outbound rules; ip_cidrs and service_tags create inbound rules. "
        "Environment names must resolve to existing perimeter profiles.",
    )
    add(
        "catalog-public-network-access",
        "Public network access resource exceptions",
        collection(
            "publicnetworkaccess",
            "instances",
            obj(
                {
                    "subscription_name": SUBSCRIPTION,
                    "resources": array(
                        obj(
                            {"resource_type": TEXT, "resource_names": strings},
                            ["resource_type", "resource_names"],
                        )
                    ),
                },
                ["subscription_name", "resources"],
            ),
        ),
        "_publicnetworkaccess",
        ("publicnetworkaccess", "instances"),
        ["code/publicnetworkaccess/publicnetworkaccess.locals.tf"],
        "subscription_name",
    )
    add(
        "catalog-gsa",
        "Global Secure Access application segments",
        collection(
            "gsa",
            "instances",
            obj(
                {
                    "subscription_name": SUBSCRIPTION,
                    "type": {"enum": ["fqdn"]},
                    "segments": array(
                        obj(
                            {
                                "azure_resource_name": TEXT,
                                "destinationHost": TEXT,
                                "ports": array({"enum": ["80-80", "443-443", "1433-1433"]}),
                            },
                            ["azure_resource_name", "destinationHost", "ports"],
                        )
                    ),
                },
                ["subscription_name", "type", "segments"],
            ),
        ),
        "_gsa",
        ("gsa", "instances"),
        ["code/gsa/gsa.locals.tf", "code/gsa/gsa.resources.tf"],
        "subscription_name",
        "Requires an existing team-owned subscription with the GSA landing-zone setup.",
    )
    notification = obj(
        {
            "notification_level": default(TEXT, "Critical"),
            "default_recipients": default(boolean, True),
        },
        [],
    )
    add(
        "catalog-pim",
        "Privileged Identity Management role policies",
        obj(
            {
                "policies": array(
                    obj(
                        {
                            "subscription": SUBSCRIPTION,
                            "roles": array(
                                obj(
                                    {
                                        "name": TEXT,
                                        "scope": TEXT,
                                        "role_assignment_expiry": {
                                            "oneOf": [TEXT, {"const": False}]
                                        },
                                        "maximum_duration": default(TEXT, "PT8H"),
                                        "require_approval": default(boolean, False),
                                        "require_justification": default(boolean, False),
                                        "approvers": array(
                                            obj(
                                                {
                                                    "type": {"enum": ["Group", "User"]},
                                                    "name": TEXT,
                                                    "object_id": TEXT,
                                                },
                                                ["type"],
                                            ),
                                            0,
                                        ),
                                        "approver_notifications": notification,
                                        "assignee_notifications": obj(
                                            {
                                                **notification["properties"],
                                                "additional_recipients": strings,
                                            },
                                            [],
                                        ),
                                    },
                                    ["name"],
                                )
                            ),
                        },
                        ["subscription", "roles"],
                    )
                )
            },
            ["policies"],
        ),
        "_pim",
        ("policies",),
        ["code/pim/pim.locals.tf", "code/pim/pim.resources.tf"],
        "subscription",
        "Restricted roles: Owner, Contributor, User Access Administrator, and Role Based "
        "Access Control Administrator. Approver groups use name; users use object_id.",
    )
    pattern = obj(
        {"name": TEXT, "negate": boolean, "operator": TEXT, "pattern": TEXT},
        [],
    )
    rules = obj(
        {
            **fields(
                "creation update deletion required_linear_history required_signatures "
                "non_fast_forward",
                boolean,
            ),
            "required_deployment_environments": strings,
            "pull_request": obj(
                {
                    **fields(
                        "dismiss_stale_reviews_on_push require_code_owner_review "
                        "require_last_push_approval required_review_thread_resolution",
                        boolean,
                    ),
                    "required_approving_review_count": {"type": "integer", "minimum": 0},
                },
                [],
            ),
            "required_status_checks": obj(
                {
                    "required_check": array(
                        obj(
                            {"context": TEXT, "integration_id": {"type": "integer"}},
                            ["context"],
                        ),
                        0,
                    ),
                    **fields(
                        "strict_required_status_checks_policy do_not_enforce_on_create", boolean
                    ),
                },
                [],
            ),
            **fields(
                "commit_message_pattern commit_author_email_pattern "
                "committer_email_pattern branch_name_pattern tag_name_pattern",
                pattern,
            ),
        },
        [],
    )
    add(
        "catalog-rulesets",
        "GitHub repository rulesets",
        collection(
            "config",
            "rulesets",
            obj(
                {
                    "ruleset_config": TEXT,
                    "ruleset": obj(
                        {
                            "name": TEXT,
                            "target": {"enum": ["branch", "tag"]},
                            "enforcement": {"enum": ["active", "disabled", "evaluate"]},
                            "rules": rules,
                            "conditions": obj(
                                {
                                    "ref_name": obj(
                                        {"include": strings, "exclude": strings},
                                        [],
                                    )
                                },
                                ["ref_name"],
                            ),
                            "bypass_actors": array(
                                obj(
                                    {
                                        "actor_id": {"type": "integer"},
                                        "actor_type": TEXT,
                                        "bypass_mode": TEXT,
                                    },
                                    [],
                                ),
                                0,
                            ),
                        },
                        ["name", "target", "enforcement", "rules"],
                    ),
                },
                ["ruleset_config", "ruleset"],
            ),
        ),
        "_github_rulesets",
        ("config", "rulesets"),
        ["code/githubenvs/githubenvs.locals.tf", "code/githubenvs/github_ruleset.tf"],
        "ruleset_config",
        "Select this ruleset_config in the repository's OIDC instance or OIDC defaults.",
    )
    add(
        "catalog-whitelist",
        "Existing Entra group cross-team permission whitelist",
        collection(
            "whitelist",
            "groups",
            obj(
                {"group_name": TEXT, "allowed_teams": array(TEXT)},
                ["group_name", "allowed_teams"],
            ),
        ),
        "cloudops/whitelist_adgroups.json",
        ("whitelist", "groups"),
        ["code/cloudops/whitelist.tf"],
        "group_name",
        "Cloudops-owned shared permission whitelist; does not create the group.",
    )
    role = schemas["catalog-pim"]["properties"]["policies"]["items"]["properties"]["roles"]["items"]
    role["properties"]["name"] = {
        **TEXT,
        "not": {
            "enum": [
                "Owner",
                "Contributor",
                "User Access Administrator",
                "Role Based Access Control Administrator",
            ]
        },
    }
    role["properties"]["approvers"]["items"] = {
        "oneOf": [
            obj({"type": {"const": "Group"}, "name": TEXT}, ["type", "name"]),
            obj({"type": {"const": "User"}, "object_id": TEXT}, ["type", "object_id"]),
        ]
    }
    role["allOf"] = [
        {
            "if": {
                "properties": {"require_approval": {"const": True}},
                "required": ["require_approval"],
            },
            "then": {"required": ["approvers"], "properties": {"approvers": {"minItems": 1}}},
        }
    ]
    nsp = schemas["catalog-nsp"]["properties"]["nsp"]["properties"]["instances"]["items"]
    nsp["anyOf"] = [
        {"required": [key], "properties": {key: {"minItems": 1}}}
        for key in ("fqdns", "ip_cidrs", "service_tags")
    ]
    application_role = app["properties"]["approles"]["items"]
    application_role["allOf"] = [
        {
            "if": {
                "properties": {"default_role": {"const": "true"}},
                "required": ["default_role"],
            },
            "then": {},
            "else": {"required": ["allowed_member_types", "display_name"]},
        }
    ]
    application_role["properties"]["enabled"]["description"] = (
        "Read into release locals, but not applied by the current app-role resource."
    )
    scope["properties"]["id"] = {
        "type": "string",
        "pattern": "^[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}$",
        "description": "Supply the permission-scope UUID; release does not safely default it.",
    }
    scope["properties"]["type"] = {"enum": ["User", "Admin"]}
    scope["required"] = [
        "id",
        "value",
        "admin_consent_description",
        "admin_consent_display_name",
    ]
    return schemas
