import logging
import re
from datetime import UTC, datetime
from typing import Any

import httpx
from azure.identity import DefaultAzureCredential
from azure.mgmt.resource import ResourceManagementClient, SubscriptionClient

from cloudops_hub.config import Settings

ARM_API_VERSIONS = {
    "Microsoft.Resources/resourceGroups": "2021-04-01",
    "Microsoft.Network/virtualNetworks": "2024-01-01",
    "Microsoft.Network/virtualNetworks/subnets": "2024-01-01",
    "Microsoft.Network/azureFirewalls": "2024-01-01",
    "Microsoft.Network/firewallPolicies": "2024-01-01",
    "Microsoft.Network/privateEndpoints": "2024-01-01",
    "Microsoft.Network/privateDnsZones": "2020-06-01",
    "Microsoft.KeyVault/vaults": "2023-07-01",
}
GRAPH_COLLECTIONS = {
    "group": ("groups", "id,displayName,securityEnabled,mailEnabled"),
    "application": ("applications", "id,appId,displayName"),
    "service_principal": ("servicePrincipals", "id,appId,displayName,accountEnabled"),
}
GUID_PATTERN = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
ARM_ID_PATTERN = re.compile(
    r"^/subscriptions/([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})/"
    r"resourceGroups/[^/]+/providers/(.+)$"
)
RESOURCE_GROUP_ID_PATTERN = re.compile(
    r"^/subscriptions/([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})/resourceGroups/([^/]+)$"
)
SUBSCRIPTION_ID_PATTERN = re.compile(
    r"^/subscriptions/([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})$"
)
logger = logging.getLogger(__name__)


def _result(resource_type: str, status: str, **extra: Any) -> dict[str, Any]:
    return {
        "resource_type": resource_type,
        "status": status,
        "checked_at": datetime.now(UTC).isoformat(),
        **extra,
    }


def lookup_arm(resource_id: str, settings: Settings) -> dict[str, Any]:
    if not settings.arm_lookups_enabled:
        return _result("azure_resource", "not_checked", message="Live Azure lookups are disabled.")
    subscription_match = SUBSCRIPTION_ID_PATTERN.fullmatch(resource_id)
    match = ARM_ID_PATTERN.fullmatch(resource_id)
    group_match = RESOURCE_GROUP_ID_PATTERN.fullmatch(resource_id)
    if not match and not group_match and not subscription_match:
        return _result(
            "azure_resource", "invalid", message="The resource ID format is not supported."
        )
    if subscription_match:
        subscription_id = subscription_match.group(1)
        provider_type = "Microsoft.Resources/subscriptions"
    elif match:
        subscription_id, provider_path = match.groups()
        segments = provider_path.split("/")
        if len(segments) < 3 or len(segments) % 2 == 0 or any(not segment for segment in segments):
            return _result(
                "azure_resource", "invalid", message="The resource ID format is not supported."
            )
        provider_type = f"{segments[0]}/" + "/".join(segments[1:-1:2])
    else:
        subscription_id = group_match.group(1)
        provider_type = "Microsoft.Resources/resourceGroups"
    api_version = ARM_API_VERSIONS.get(provider_type)
    if not subscription_match and not api_version:
        return _result(
            "azure_resource", "unsupported", message="This Azure resource type is not allowlisted."
        )
    if subscription_id.casefold() not in {
        item.casefold() for item in settings.arm_subscription_ids
    }:
        return _result(
            "azure_resource", "not_checked", message="The subscription is not allowlisted."
        )
    credential = DefaultAzureCredential()
    try:
        if subscription_match:
            resource = SubscriptionClient(credential).subscriptions.get(subscription_id)
            return _result(
                provider_type,
                "found",
                resource_id=resource.id,
                metadata={"display_name": resource.display_name, "state": resource.state},
                source="Azure Resource Manager",
            )
        client = ResourceManagementClient(credential, subscription_id)
        if group_match:
            resource = client.resource_groups.get(group_match.group(2))
        else:
            resource = client.resources.get_by_id(resource_id, api_version)
        return _result(
            provider_type,
            "found",
            resource_id=resource.id,
            metadata={
                "name": resource.name,
                "type": resource.type,
                "location": resource.location,
            },
            source="Azure Resource Manager",
        )
    except Exception as exc:
        status_code = getattr(exc, "status_code", None)
        if status_code == 404:
            return _result(
                provider_type, "not_found", resource_id=resource_id, source="Azure Resource Manager"
            )
        if status_code in (401, 403):
            return _result(provider_type, "access_denied", source="Azure Resource Manager")
        logger.warning("Azure lookup failed (%s)", type(exc).__name__)
        return _result(
            provider_type, "query_failed", message="Azure Resource Manager query failed."
        )
    finally:
        credential.close()


async def lookup_graph(object_type: str, object_id: str, settings: Settings) -> dict[str, Any]:
    if not settings.graph_lookups_enabled:
        return _result(
            object_type, "not_checked", message="Live Microsoft Graph lookups are disabled."
        )
    if object_type not in GRAPH_COLLECTIONS or object_type not in settings.graph_allowed_types:
        return _result(
            object_type, "unsupported", message="This Entra object type is not allowlisted."
        )
    if not GUID_PATTERN.fullmatch(object_id):
        return _result(object_type, "invalid", message="A valid object ID is required.")
    credential = DefaultAzureCredential()
    try:
        token = credential.get_token("https://graph.microsoft.com/.default")
        collection, selected = GRAPH_COLLECTIONS[object_type]
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(
                f"https://graph.microsoft.com/v1.0/{collection}/{object_id}",
                params={"$select": selected},
                headers={"Authorization": f"Bearer {token.token}"},
            )
        if response.status_code == 404:
            return _result(object_type, "not_found", object_id=object_id, source="Microsoft Graph")
        if response.status_code in (401, 403):
            return _result(object_type, "access_denied", source="Microsoft Graph")
        if response.status_code >= 400:
            logger.warning("Graph lookup failed (HTTP %s)", response.status_code)
            return _result(object_type, "query_failed", source="Microsoft Graph")
        data = response.json()
        safe_fields = set(selected.split(","))
        return _result(
            object_type,
            "found",
            object_id=object_id,
            metadata={key: value for key, value in data.items() if key in safe_fields},
            source="Microsoft Graph",
        )
    except Exception as exc:
        logger.warning("Graph lookup unavailable (%s)", type(exc).__name__)
        return _result(
            object_type,
            "unavailable",
            message="Microsoft Graph connectivity or credentials are unavailable.",
        )
    finally:
        credential.close()
