"""Allowlisted traffic evidence for the Hub's model boundary, not the Azure readers."""

import ipaddress
import json
import re
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from urllib.parse import urlsplit

EVIDENCE_VERSION = 4
_ACTIVE_REFERENCES: ContextVar["TrafficReferences | None"] = ContextVar(
    "firewall_endpoint_references", default=None
)
_REFERENCE = r"fw_endpoint_[a-f0-9]{32}"
_HOSTNAME = (
    r"(?<![\w-])(?:[\w](?:[\w-]{0,61}[\w])?\.)+"
    r"[^\W\d_](?:[\w-]{0,61}[\w])?\.?(?![\w-])"
)
TRAFFIC_TOOLS = {
    "check_onprem_runner_rules",
    "check_onprem_runner_logs",
    "check_connection",
    "check_github_runner_connection",
    "analyze_firewall_rules",
    "search_firewall_logs",
    "find_denied_sources",
    "search_ip_activity",
    "search_runner_activity",
    "plan_rules_from_denied_logs",
}
# Endpoint values inside projected classification/ticket data become opaque references.
_ENDPOINT_FIELDS = {
    "endpoint",
    "destination",
    "destination_ip",
    "cname_chain",
    "ip_groups",
    "resolved_ips",
    "source_ips",
    "destination_ips",
    "destination_fqdn",
    "peer",
    "matched_ips",
}


def _destination(value):
    if not isinstance(value, str):
        return None
    try:
        ipaddress.ip_network(value, strict=False)
        return value
    except ValueError:
        pass
    # URLs in logs can contain credentials, paths and query parameters.
    return urlsplit(value).hostname if "://" in value else value.split("/")[0].split("?")[0]


def _endpoint(value):
    if isinstance(value, dict):
        return value.get("fqdn") or value.get("ip") or value.get("input")
    return value if isinstance(value, str) else None


def _action(value):
    return {
        "allow": "Allow",
        "allowed": "Allow",
        "deny": "Deny",
        "denied": "Deny",
        "default_deny": "Deny",
        "allowed_and_denied": "AllowAndDeny",
        "no_logs": "NoEvidence",
        "not_checked": "NotChecked",
        "dnat": "DNAT",
    }.get(str(value).casefold(), "Unknown")


def _flow(source, destination, port, protocol, action):
    return {
        "source": _endpoint(source),
        "destination": _destination(_endpoint(destination)),
        "port": port if isinstance(port, int) and not isinstance(port, bool) else None,
        "protocol": (
            protocol.upper()
            if isinstance(protocol, str)
            and protocol.upper() in {"TCP", "UDP", "ICMP", "ANY", "HTTP", "HTTPS", "MSSQL", "TLS"}
            else None
        ),
        "action": _action(action),
    }


def _log(row):
    return _flow(
        row.get("source_ip"),
        row.get("fqdn") or row.get("destination_ip") or row.get("target_url"),
        row.get("destination_port"),
        row.get("protocol"),
        row.get("action"),
    )


_LOCATIONS = {"onprem", "azure", "internet", "unknown"}
_TEAM = re.compile(r"[A-Za-z0-9][A-Za-z0-9 _-]{0,40}")
_TICKET_TEAMS = {"CloudOps team", "On-prem network team"}
_SEGMENT_GROUPS = {
    "ipg-lz-prod-all-vnets": "Production segment",
    "ipg-lz-nonprod-all-vnets": "Nonproduction segment",
}
_COMMON_GROUPS = {"ipg-all-azure-vnets", *_SEGMENT_GROUPS}
_RECOMMENDATION_KINDS = {"info", "create_ticket", "remove_fqdn", "contact_cloudops", "warning"}


def _teams(values):
    return [value for value in values or () if isinstance(value, str) and _TEAM.fullmatch(value)]


def _strings(values):
    return [value for value in values or () if isinstance(value, str) and value]


def _location(endpoint):
    """Classification and ownership only; reasons are free text and stay backend-side."""
    if not isinstance(endpoint, dict):
        return None
    location = endpoint.get("location") or {}
    kind = location.get("location")
    groups = _strings(location.get("ip_groups"))
    group_names = {group.rsplit("/", 1)[-1].casefold() for group in groups}
    value = endpoint.get("input") or _endpoint(endpoint)
    input_kind = "unknown"
    if isinstance(value, str) and value:
        try:
            ipaddress.ip_network(value, strict=False)
        except ValueError:
            input_kind = "hostname"
        else:
            input_kind = "cidr" if "/" in value else "ip"
    return {
        "endpoint": _endpoint(endpoint),
        "input_kind": input_kind,
        "location": kind if kind in _LOCATIONS else "unknown",
        "teams": _teams(location.get("teams")),
        "ip_groups": [
            group for group in groups if group.rsplit("/", 1)[-1].casefold() not in _COMMON_GROUPS
        ],
        "segment_types": [label for name, label in _SEGMENT_GROUPS.items() if name in group_names],
        "resolved_ips": _strings(endpoint.get("ips")),
    }


def _ticket(ticket, destination_location):
    if not isinstance(ticket, dict):
        return None
    fqdn = ticket.get("destination_fqdn")
    return {
        "team": ticket.get("team") if ticket.get("team") in _TICKET_TEAMS else None,
        "direction": ticket.get("direction")
        if ticket.get("direction") in {"inbound", "outbound"}
        else "",
        "azure_firewall": ticket.get("azure_firewall")
        if ticket.get("azure_firewall") in {"denied", "allowed", "no_logs"}
        else "",
        "rule_type": "application rule"
        if fqdn and destination_location == "internet"
        else "network rule",
        "source_ips": _strings(ticket.get("source")),
        "destination_ips": _strings(ticket.get("destination")),
        "destination_fqdn": fqdn if isinstance(fqdn, str) else None,
        "source_teams": _teams(ticket.get("source_teams")),
        "destination_teams": _teams(ticket.get("destination_teams")),
        "flows": [
            {
                "port": flow.get("port") if isinstance(flow.get("port"), int) else None,
                "protocol": flow.get("protocol") if isinstance(flow.get("protocol"), str) else None,
            }
            for flow in ticket.get("flows") or ()
            if isinstance(flow, dict)
        ],
        "fqdn_tag": ticket.get("fqdn_tag") if isinstance(ticket.get("fqdn_tag"), str) else None,
    }


_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:?\d{2})?")


def _count(value):
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _timestamp(value):
    if hasattr(value, "isoformat"):
        value = value.isoformat()
    return value if isinstance(value, str) and _TIMESTAMP.fullmatch(value) else None


def _connection(result):
    logs = result.get("logs") or {}
    source = _location(result.get("source"))
    destination = _location(result.get("destination"))
    latest = logs.get("latest_entry") or {}
    return {
        "traffic": [
            _flow(
                result.get("source"),
                result.get("destination"),
                result.get("destination_port"),
                result.get("protocol"),
                logs.get("verdict"),
            )
        ],
        "log_entries": [_log(latest)] if latest else [],
        "log_summary": {
            "logs_found": bool(
                _count(logs.get("allowed_count")) or _count(logs.get("denied_count")) or latest
            ),
            "verdict": _action(logs.get("verdict")),
            "lookback_hours": _count(logs.get("lookback_hours")),
            "allowed_count": _count(logs.get("allowed_count")),
            "denied_count": _count(logs.get("denied_count")),
            "last_seen_utc": _timestamp(latest.get("time_generated")),
            "denied_flows": [
                {"port": flow.get("port"), "protocol": flow.get("protocol")}
                for flow in logs.get("denied_flows") or ()
            ],
        },
        "source_location": source,
        "destination_location": destination,
        "recommendations": [
            {
                "kind": item.get("kind"),
                "warning_code": (
                    "missing_flow_details"
                    if item.get("code") == "missing_flow_details"
                    else
                    "verify_direction"
                    if item.get("code") == "verify_direction"
                    and source
                    and source["location"] == "internet"
                    and destination
                    and destination["location"] in {"azure", "onprem"}
                    else None
                ),
                "team": item.get("team") if item.get("team") in _TICKET_TEAMS else None,
                "ticket": _ticket(
                    item.get("ticket"), destination["location"] if destination else None
                ),
            }
            for item in result.get("recommendations") or ()
            if isinstance(item, dict) and item.get("kind") in _RECOMMENDATION_KINDS
        ],
    }


def _group_resolution(result):
    projected = {
        key: result[key]
        for key in ("status", "query", "message", "addresses", "candidate_count")
        if key in result
    }

    def group(value):
        return {
            key: value[key]
            for key in ("name", "id", "resource_group", "address_count")
            if key in value
        }

    if "group" in result:
        projected["group"] = group(result["group"])
    if "candidates" in result:
        projected["candidates"] = [group(value) for value in result["candidates"]]
    return projected


_DIRECTIONS = {"inbound", "outbound", "both"}
_LOG_TYPES = {
    "AZFWApplicationRule": "application",
    "AZFWNetworkRule": "network",
    "AZFWNatRule": "DNAT",
    "AZFWThreatIntel": "threat intel",
}


def _activity_row(row, *, peer=True):
    """Counts, times and endpoints only; policy/rule names stay backend-side."""
    flow = _flow(None, row.get("peer"), row.get("port"), row.get("protocol"), row.get("action"))
    projected = {
        "direction": row.get("direction") if row.get("direction") in _DIRECTIONS else None,
        "action": flow["action"],
        "log_type": _LOG_TYPES.get(row.get("log_type"), "other") if row.get("log_type") else None,
        "hits": _count(row.get("hits")),
        "last_seen_utc": _timestamp(row.get("last_seen")),
    }
    if peer:
        projected.update(
            peer=flow["destination"],
            port=flow["port"],
            protocol=flow["protocol"],
            matched_ips=_strings(row.get("matched_ips")),
        )
    return {key: value for key, value in projected.items() if value not in (None, [])}


def _ip_activity(result):
    return {
        "lookback_hours": _count(result.get("lookback_hours")),
        "totals": [_activity_row(row, peer=False) for row in result.get("totals") or ()],
        **{
            key: [
                {
                    k: v
                    for k, v in _activity_row({**row, "action": "Deny"}).items()
                    if k not in {"direction", "log_type"}
                }
                for row in result.get(key) or ()
            ]
            for key in ("denied_urls", "denied_destination_ips", "denied_inbound_sources")
        },
        "groups": [_activity_row(row) for row in result.get("groups") or ()],
        "groups_truncated": bool(result.get("groups_truncated")),
    }


def model_traffic(name: str, result) -> dict:
    """Never forward raw log fields, policy objects, free-text findings or proposals."""
    if isinstance(result, dict) and "endpoint_resolution" in result:
        return {
            "status": result["status"],
            "message": result["message"],
            "endpoint_resolution": {
                field: _group_resolution(resolution)
                for field, resolution in result["endpoint_resolution"].items()
            },
            "checks": [
                {"result": model_traffic(name, check["result"])}
                for check in result.get("checks", [])
            ],
        }
    if name == "check_onprem_runner_rules":
        return {
            key: value
            for key, value in result.items()
            if key in {"status", "runner", "candidates", "message", "scope", "next_step"}
        } | {
            "checks": [
                {
                    key: value
                    for key, value in row.items()
                    if key in {"source", "destination", "port", "protocol", "action"}
                }
                for row in result.get("checks", [])
            ]
        }
    if name == "search_firewall_logs":
        return {"traffic": [_log(row) for row in result]}
    if name == "find_denied_sources":
        return {
            "destination": _destination(result.get("destination")),
            "action": "Deny",
            "lookback_hours": _count(result.get("lookback_hours")),
            "match_scope": result.get("match_scope")
            if result.get("match_scope") in {"exact_fqdn", "destination_ip", "fqdn_or_resolved_ip"}
            else "unknown",
            "dns": {
                "status": (result.get("dns") or {}).get("status")
                if (result.get("dns") or {}).get("status")
                in {"resolved", "failed", "not_applicable"}
                else "unknown",
                "resolved_ips": _strings((result.get("dns") or {}).get("resolved_ips")),
                "cname_chain": _strings((result.get("dns") or {}).get("cname_chain")),
                "checked_at": _timestamp((result.get("dns") or {}).get("checked_at")),
            },
            "logs_found": result.get("logs_found") is True,
            "sources_truncated": result.get("sources_truncated") is True,
            "truncated_match_kinds": [
                kind
                for kind in result.get("truncated_match_kinds", [])
                if kind in {"fqdn", "resolved_ip", "destination_ip"}
            ],
            "sources": [
                {
                    **_flow(
                        row.get("source"),
                        row.get("observed_destination") or result.get("destination"),
                        row.get("port"),
                        row.get("protocol"),
                        "Deny",
                    ),
                    "hits": _count(row.get("hits")),
                    "last_seen_utc": _timestamp(row.get("last_seen_utc")),
                    "destination_ip": _endpoint(row.get("destination_ip")),
                    "match_kind": row.get("match_kind")
                    if row.get("match_kind") in {"fqdn", "resolved_ip", "destination_ip"}
                    else "unknown",
                    "log_type": _LOG_TYPES.get(row.get("log_type"), "other"),
                }
                for row in result.get("sources") or ()
            ],
        }
    if name == "check_connection":
        return _connection(result)
    if name in {"check_github_runner_connection", "check_onprem_runner_logs"}:
        # Discovery/onboarding metadata is separate from traffic evidence.
        data = {
            key: value
            for key, value in result.items()
            if key
            in {
                "status",
                "runner",
                "candidates",
                "message",
                "source_ip",
                "expected_network",
                "workflow_settings_if_team_json_missing",
                "diagnostic_steps",
            }
        }
        data["checks"] = [
            {"device": check.get("device"), "result": _connection(check["result"])}
            for check in result.get("checks", [])
        ]
        return data
    if name == "analyze_firewall_rules":
        return {
            "traffic": [_log(row) for row in result.get("recent_logs", [])],
            "policy_evaluation": [
                _flow(
                    result.get("source"),
                    result.get("destination"),
                    result.get("destination_port"),
                    result.get("protocol"),
                    evaluation.get("verdict") if evaluation.get("conclusive") else None,
                )
                for evaluation in result.get("evaluations", [])
            ],
        }
    if name == "search_ip_activity":
        return _ip_activity(result)
    if name == "search_runner_activity":
        return {
            key: value
            for key, value in result.items()
            if key in {"status", "candidates", "message", "lookback_hours"}
        } | {
            "runners": [
                {
                    "runner": row.get("runner"),
                    "subnet": row.get("subnet"),
                    "address_prefix": _endpoint(row.get("address_prefix")),
                    "devices": [
                        {"device": device.get("device"), "ip": _endpoint(device.get("ip"))}
                        for device in row.get("devices") or ()
                    ],
                    "devices_truncated": row.get("devices_truncated") is True,
                    "activity": _ip_activity(row.get("activity") or {}),
                }
                for row in result.get("runners") or ()
            ]
        }
    if name == "plan_rules_from_denied_logs":
        return {
            "traffic": [
                _flow(
                    source, target.get("target"), target.get("port"), target.get("protocol"), "Deny"
                )
                for target in result.get("denied_targets", [])
                for source in target.get("source_ips", [])
            ]
        }
    raise ValueError(f"No traffic projection for tool {name}")


def user_history(history: list[dict]) -> list[dict]:
    """Do not resend old assistant answers containing pre-minimization Azure evidence."""
    return [message for message in history if message.get("role") == "user"]


class TrafficReferences:
    """Keep log strings backend-side; the model sees opaque, conversation-scoped handles."""

    value_pattern = _HOSTNAME
    restore_outputs = False

    def __init__(self, values: dict[str, str] | None = None, *, public_domain: str | None = None):
        self.values = dict(values or {})
        self.public_domain = public_domain

    def _reference(self, value: str) -> str:
        if value in self.values:
            return value
        for reference, original in self.values.items():
            if original == value:
                return reference
        reference = f"fw_endpoint_{uuid.uuid4().hex}"
        self.values[reference] = value
        return reference

    def protect(self, value):
        if isinstance(value, list):
            return [self.protect(item) for item in value]
        if isinstance(value, dict):
            if "endpoint_resolution" in value:
                value = {
                    **value,
                    "endpoint_resolution": self._group_references(value["endpoint_resolution"]),
                }
            if {"source", "destination", "port", "protocol", "action"} <= set(value):
                value = {
                    key: self._reference(item)
                    if key in {"source", "destination"} and isinstance(item, str)
                    else item
                    for key, item in value.items()
                }
            value = {
                key: (
                    self._reference(item)
                    if isinstance(item, str)
                    else [
                        self._reference(entry) if isinstance(entry, str) else entry
                        for entry in item
                    ]
                    if isinstance(item, list)
                    else item
                )
                if key in _ENDPOINT_FIELDS and item
                else item
                for key, item in value.items()
            }
            return {
                self.protect(key) if isinstance(key, str) else key: self.protect(item)
                for key, item in value.items()
            }
        if isinstance(value, str):
            if value.lstrip().startswith(("{", "[")):
                try:
                    decoded = json.loads(value)
                except ValueError:
                    pass
                else:
                    return json.dumps(self.protect(decoded), ensure_ascii=False)
            originals = sorted(set(self.values.values()) - {""}, key=len, reverse=True)
            known = "|".join(
                rf"(?<![\w.-]){re.escape(original)}(?![\w.-])" for original in originals
            )
            pattern = "|".join(part for part in (_REFERENCE, known, self.value_pattern) if part)

            def replace(match):
                text = match[0]
                if self.public_domain and text.casefold().rstrip(".") == self.public_domain:
                    return text
                return text if re.fullmatch(_REFERENCE, text) else self._reference(text)

            return re.sub(pattern, replace, value)
        return value

    def _group_references(self, resolutions: dict) -> dict:
        """Group labels/IDs and ranges stay local, including names with no DNS dots."""
        protected = deepcopy(resolutions)
        for resolution in protected.values():
            if "query" in resolution:
                resolution["query"] = self._reference(resolution["query"])
            if "addresses" in resolution:
                resolution["addresses"] = [
                    self._reference(address) for address in resolution["addresses"]
                ]
            groups = resolution.get("candidates", [])
            if resolution.get("group"):
                groups = [*groups, resolution["group"]]
            for group in groups:
                for key in ("name", "id", "resource_group"):
                    if key in group:
                        group[key] = self._reference(group[key])
        return protected

    def messages(self, messages: list[dict]) -> list[dict]:
        """Protect content, never API roles, tool names, IDs or other envelope fields."""
        messages = deepcopy(messages)
        for message in messages:
            if "content" in message:
                message["content"] = self.protect(message["content"])
            for call in message.get("tool_calls", []):
                function = call.get("function", {})
                if "arguments" in function:
                    function["arguments"] = self.protect(function["arguments"])
        return messages

    def arguments(self, value):
        if isinstance(value, dict):
            return {self.arguments(key): self.arguments(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.arguments(item) for item in value]
        if isinstance(value, str):

            def resolve(match):
                if match[0] not in self.values:
                    raise ValueError(
                        "Endpoint reference is no longer available. Ask for the endpoint and retry."
                    )
                return self.values[match[0]]

            return re.sub(_REFERENCE, resolve, value)
        return value

    def restore(self, text: str) -> str:
        """Real values for backend-only use such as search; unknown references are dropped."""
        return re.sub(_REFERENCE, lambda match: self.values.get(match[0], ""), text)

    def render(self, answer: str) -> str:
        # One pass avoids interpreting restored values as other references.
        return re.sub(
            _REFERENCE,
            lambda match: (
                self.values.get(match[0], match[0])
                .replace("\n", " ")
                .replace("\r", " ")
                .replace("|", r"\|")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace("`", r"\`")
            ),
            answer,
        )


_OCTET = r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
_INTERNAL_SUFFIXES = ("local", "internal", "corp", "lan")
_INTERNAL_PATTERNS = (
    r"/subscriptions/[^\s\"'`<>|(){}\[\],;]+",
    r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+",
    r"(?<![\w-])[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}(?![\w-])",
    rf"(?<![\w.]){_OCTET}(?:\.{_OCTET}){{3}}(?:/(?:3[0-2]|[12]?\d))?(?!\w|\.\d|/\d)",
    r"(?<![\w.-])(?:[\w-]+\.)*privatelink\.(?:[\w-]+\.)*[\w-]+(?![\w-])",
)


class InternalReferences(TrafficReferences):
    """Non-Firewall assistants: mask internal identifiers, keep ordinary names and files readable.

    Model output is restored right after each call, so search, schema validation and
    answers use the real values while the model only ever sees opaque references.
    """

    restore_outputs = True

    def __init__(self, domains: list[str] | tuple[str, ...] = ()):
        super().__init__()
        suffixes = {
            item.strip().strip(".").casefold()
            for item in (*_INTERNAL_SUFFIXES, *domains)
            if item.strip().strip(".")
        }
        names = "|".join(re.escape(item) for item in sorted(suffixes, key=len, reverse=True))
        self.value_pattern = "|".join(
            (
                *_INTERNAL_PATTERNS,
                rf"(?<![\w.@-])(?:[\w-]+\.)+(?i:{names})\.?(?![\w-]|\.[\w-])",
            )
        )


@contextmanager
def internal_boundary(settings):
    """Mask internal names for every assistant except Firewall, which has its own boundary."""
    if not getattr(settings, "model_mask_internal_names", False) or active_references():
        yield active_references()
        return
    references = InternalReferences(getattr(settings, "model_mask_domains", []) or [])
    token = _ACTIVE_REFERENCES.set(references)
    try:
        yield references
    finally:
        _ACTIVE_REFERENCES.reset(token)


def active_references() -> TrafficReferences | None:
    return _ACTIVE_REFERENCES.get()


def protect_model_data(value):
    references = active_references()
    return references.protect(value) if references is not None else value


@contextmanager
def firewall_boundary(references: TrafficReferences | None = None):
    current = active_references()
    if current is not None:
        yield current
        return
    references = references or TrafficReferences()
    token = _ACTIVE_REFERENCES.set(references)
    try:
        yield references
    finally:
        _ACTIVE_REFERENCES.reset(token)


@contextmanager
def public_purpose_boundary(hostname: str):
    """Only for a DNS-validated public-purpose request with no conversation context."""
    token = _ACTIVE_REFERENCES.set(TrafficReferences(public_domain=hostname))
    try:
        yield
    finally:
        _ACTIVE_REFERENCES.reset(token)
