from functools import lru_cache
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = "development"
    session_secret: str = ""
    session_https_only: bool = True
    # Hub session and AZP connection lifetime.
    session_hours: int = 1
    entra_tenant_id: str = ""
    entra_client_id: str = ""
    entra_client_secret: str = ""
    entra_required_app_role: str = "CloudOpsHub.User"
    # AZP accounts can live on a different domain than AB accounts.
    azp_login_domain: str = ""
    github_catalog_client_id: str = ""
    github_catalog_client_secret: str = ""
    # "CloudOps Hub - Read Sources" GitHub App; required for GitHub reads (0 disables them).
    github_read_app_id: int = 0
    # 0 discovers the installation from the catalog repository.
    github_read_app_installation_id: int = 0
    import_job_id: str = ""
    # Separate scheduled job that extracts glossary keywords from logged questions.
    keyword_scan_job_id: str = ""
    catalog_live_vault_url: str = ""
    approved_firewall_schema_id: str = ""
    firewall_analysis_enabled: bool = False
    firewall_subscription_ids: Annotated[list[str], NoDecode] = []
    firewall_log_workspace_id: str = ""
    firewall_runner_vnet_id: str = ""
    firewall_dns_servers: Annotated[list[str], NoDecode] = ["168.63.129.16"]
    firewall_dns_search_suffixes: Annotated[list[str], NoDecode] = []
    entra_redirect_uri: str = ""
    frontend_url: str = ""
    cors_origins: Annotated[list[str], NoDecode] = []
    azure_openai_endpoint: str = ""
    azure_openai_deployment: str = ""
    azure_openai_model: str = ""
    azure_openai_private_ip: str = ""
    azure_openai_api_version: str = "v1"
    azure_openai_api_key: str = ""
    # Empty selects deterministic parameters for non-reasoning chat models.
    azure_openai_reasoning_effort: str = ""
    azure_client_id: str = ""
    azure_search_endpoint: str = ""
    azure_search_index: str = "cloudops-docs"
    azure_search_platform_index: str = "cloudops-platform-v1"
    azure_search_hybrid_enabled: bool = False
    azure_search_semantic_enabled: bool = False
    retrieval_routing_schema_enabled: bool = False
    azure_openai_embedding_deployment: str = ""
    azure_openai_embedding_dimensions: int = 1536
    retrieval_synonyms_blob: str = Field(
        default="retrieval-synonyms.json", min_length=1, pattern=r"\S"
    )
    azure_storage_knowledge_container: str = "knowledge"
    retrieval_candidates: int = Field(default=20, ge=5, le=50)
    retrieval_max_chunks: int = Field(default=6, ge=1, le=12)
    retrieval_retry_enabled: bool = True
    confluence_retrieval_enabled: bool = False
    retrieval_min_score: float = Field(default=0, ge=0)
    retrieval_min_reranker_score: float = Field(default=0, ge=0, le=4)
    retrieval_vector_k: int = Field(default=50, ge=1, le=100)
    retrieval_context_chars: int = Field(default=24_000, ge=6_000, le=60_000)
    azure_storage_account_url: str = ""
    # Optional copy of the audit trail in Log Analytics (Logs Ingestion API).
    audit_ingestion_endpoint: str = ""
    audit_dcr_immutable_id: str = ""
    audit_stream_name: str = "Custom-CloudOpsHubAudit"
    azure_storage_conversation_container: str = "conversations"
    azure_storage_insights_container: str = "insights"
    question_log_enabled: bool = True
    knowledge_graph_enabled: bool = True
    # 0 keeps question insights until an administrator deletes them.
    question_log_retention_days: int = Field(default=0, ge=0, le=3650)
    arm_lookups_enabled: bool = False
    arm_subscription_ids: Annotated[list[str], NoDecode] = []
    graph_lookups_enabled: bool = False
    graph_allowed_types: Annotated[list[str], NoDecode] = [
        "group",
        "application",
        "service_principal",
    ]
    max_request_bytes: int = 65_536
    chat_rate_limit_per_minute: int = 30
    model_request_timeout_seconds: float = Field(default=180, gt=0, le=180)
    answer_output_tokens: int | None = Field(default=None, ge=64, le=8000)
    troubleshooting_output_tokens: int | None = Field(default=None, ge=64, le=8000)
    json_output_tokens: int | None = Field(default=None, ge=128, le=16000)
    full_file_output_tokens: int | None = Field(default=None, ge=128, le=32000)
    classifier_output_tokens: int | None = Field(default=None, ge=64, le=2000)
    vision_output_tokens: int | None = Field(default=None, ge=128, le=16000)
    reasoning_budget_tokens: int = Field(default=1024, ge=0, le=8000)
    history_max_messages: int = Field(default=4, ge=0, le=8)
    history_max_chars: int = Field(default=4000, ge=0, le=24000)
    chat_max_input_chars: int = Field(default=10000, ge=100, le=10000)
    model_input_price_per_million: float | None = Field(default=None, ge=0)
    model_output_price_per_million: float | None = Field(default=None, ge=0)
    embedding_input_price_per_million: float | None = Field(default=None, ge=0)
    model_price_currency: str = "DKK"
    # Internal names are replaced by opaque references before every chat-model call.
    model_mask_internal_names: bool = True
    model_mask_domains: Annotated[list[str], NoDecode] = []

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @field_validator(
        "cors_origins",
        "arm_subscription_ids",
        "graph_allowed_types",
        "firewall_subscription_ids",
        "firewall_dns_servers",
        "firewall_dns_search_suffixes",
        "model_mask_domains",
        mode="before",
    )
    @classmethod
    def split_csv(cls, value: object) -> object:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator("azure_openai_reasoning_effort", mode="before")
    @classmethod
    def validate_reasoning_effort(cls, value: object) -> str:
        if not isinstance(value, str):
            raise ValueError("AZURE_OPENAI_REASONING_EFFORT must be a string")
        effort = value.strip().lower()
        if effort not in {"", "none", "minimal", "low", "medium", "high", "xhigh", "max"}:
            raise ValueError("AZURE_OPENAI_REASONING_EFFORT is not a supported reasoning level")
        return effort

    def for_assistant(self, assistant_id: str) -> "Settings":
        if assistant_id != "platform-admin":
            return self
        index = self.azure_search_platform_index.strip()
        if not index or index.casefold() == self.azure_search_index.strip().casefold():
            raise ValueError("Platform knowledge requires a separate search index")
        return self.model_copy(update={"azure_search_index": index})

    def validate_production(self) -> None:
        if self.firewall_analysis_enabled:
            from ipaddress import ip_address
            from uuid import UUID

            if not self.firewall_subscription_ids or not self.firewall_log_workspace_id:
                raise RuntimeError(
                    "Firewall analysis requires subscription IDs and a log workspace ID"
                )
            for value in (*self.firewall_subscription_ids, self.firewall_log_workspace_id):
                UUID(value)
            if not self.firewall_dns_servers:
                raise RuntimeError("Firewall analysis requires approved DNS servers")
            for server in self.firewall_dns_servers:
                ip_address(server)
        if self.azure_search_semantic_enabled and not self.azure_search_hybrid_enabled:
            raise RuntimeError("Semantic ranking requires the enriched hybrid index")
        if self.azure_search_hybrid_enabled and not self.azure_openai_embedding_deployment:
            raise RuntimeError("Hybrid retrieval requires an embedding deployment")
        if not 1 <= self.azure_openai_embedding_dimensions <= 3072:
            raise RuntimeError("Embedding dimensions must be between 1 and 3072")
        if self.app_env == "production":
            required = {
                "SESSION_SECRET": self.session_secret,
                "ENTRA_TENANT_ID": self.entra_tenant_id,
                "ENTRA_CLIENT_ID": self.entra_client_id,
                "ENTRA_CLIENT_SECRET": self.entra_client_secret,
                "ENTRA_REQUIRED_APP_ROLE": self.entra_required_app_role,
                "ENTRA_REDIRECT_URI": self.entra_redirect_uri,
                "FRONTEND_URL": self.frontend_url,
                "AZURE_STORAGE_ACCOUNT_URL": self.azure_storage_account_url,
            }
            missing = [name for name, value in required.items() if not value]
            if missing:
                raise RuntimeError(
                    f"Missing required production configuration: {', '.join(missing)}"
                )
            if len(self.session_secret) < 32:
                raise RuntimeError("SESSION_SECRET must contain at least 32 characters")
            if not self.session_https_only:
                raise RuntimeError("SESSION_HTTPS_ONLY must be true in production")


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.validate_production()
    return settings
