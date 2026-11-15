from dataclasses import dataclass


@dataclass(frozen=True)
class Capability:
    title: str
    description: str


@dataclass(frozen=True)
class Assistant:
    id: str
    title: str
    description: str
    suggested_questions: tuple[str, ...]
    capabilities: tuple[Capability, ...]


ASSISTANTS = {
    "platform-admin": Assistant(
        id="platform-admin",
        title="CloudOps Platform Assistant",
        description=(
            "Understand platform architecture, bootstrap, identities and disaster recovery "
            "from restricted CloudOps documentation. Guidance only; nothing is executed."
        ),
        suggested_questions=(),
        capabilities=(
            Capability(
                "Platform architecture",
                "Explain documented platform components, repository ownership and dependencies.",
            ),
            Capability(
                "Bootstrap and recovery guidance",
                "Draft setup checklists and runbook steps from the admin documentation.",
            ),
            Capability(
                "Identity and governance",
                "Explain documented identity, permission and governance arrangements. "
                "No credential values, Azure changes or workflow execution.",
            ),
        ),
    ),
    "service-catalog": Assistant(
        id="service-catalog",
        title="Service Catalog Assistant",
        description=(
            "Open forms for supported catalog resources and settings, prepare requests "
            "and trigger existing workflows only in "
            "almbrand/landingzone-service-catalog, using your authorized AB GitHub access."
        ),
        suggested_questions=(),
        capabilities=(
            Capability(
                "Guided catalog forms",
                "Review supported resource settings and create or update team catalog entries.",
            ),
            Capability(
                "Existing GitHub workflows",
                "Choose from verified workflow modes; existing reviews and approvals remain.",
            ),
            Capability(
                "Allowed teams",
                "Only active membership of the folder's CODEOWNERS team authorizes submissions.",
            ),
        ),
    ),
    "firewall": Assistant(
        id="firewall",
        title="Azure Firewall Assistant",
        description=(
            "Ask about Azure Firewall setup and documentation, or troubleshoot live logs, "
            "DNS, policies, IP Groups, routing and GitHub runner connectivity."
        ),
        suggested_questions=(),
        capabilities=(
            Capability(
                "Firewall setup and guidance",
                "Explain documented architecture, policies, network/application/NAT rules, "
                "DNS, routing and diagnostics.",
            ),
            Capability(
                "Read-only connectivity troubleshooting",
                "Check live policies, IP Groups, DNS and Firewall logs for a source, "
                "destination and port; distinguish allowed, denied and missing evidence.",
            ),
            Capability(
                "GitHub runner connectivity",
                "Identify runners and help diagnose their access to a destination. "
                "Checks are read-only; no probes or workflows are executed.",
            ),
            Capability(
                "Rules from denied logs",
                "List URLs denied by application rules, check them with DNS and generate "
                "application-rule JSON for public targets. Network rules (internal RFC 1918 "
                "targets) are managed by CloudOps: you get a CloudOps ticket, never JSON. "
                "Proposals only; nothing is applied.",
            ),
        ),
    ),
    "landing-zone": Assistant(
        id="landing-zone",
        title="Landing Zone Assistant",
        description=(
            "Get source-backed Landing Zone setup, access and governance guidance, "
            "understand supported service settings, and troubleshoot failed catalog workflows. "
            "Guidance only; it does not deploy resources or run workflows."
        ),
        suggested_questions=(),
        capabilities=(
            Capability(
                "Explain supported parameters",
                "Explain required/optional fields and supported settings from current "
                "service-catalog code and approved docs.",
            ),
            Capability(
                "Access and governance guidance",
                "Explain documented access procedures and CloudOps/SecOps requirements. "
                "Answers are guidance, not access grants or deployment approval.",
            ),
            Capability(
                "Troubleshoot catalog pipelines",
                "Provide a service-catalog GitHub Actions run URL to inspect failed-job "
                "logs and suggest fixes. It does not start or rerun workflows or change files.",
            ),
            Capability(
                "Explain Landing Zone procedures",
                "Get source-backed setup, ordering and operational guidance.",
            ),
        ),
    ),
}
