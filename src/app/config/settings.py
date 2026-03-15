"""Runtime settings."""

from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.domain import FilingType, RuntimeEnvironment


class ApiSettings(BaseModel):
    """HTTP API settings."""

    host: str = "127.0.0.1"
    port: int = 8000


class StorageSettings(BaseModel):
    """Storage settings shared by ETL and API layers."""

    database_url: str = "sqlite+pysqlite:///./data/app.db"
    data_dir: str = "./data"
    raw_store_dir: str = "./data/raw"


class SourceScopeSettings(BaseModel):
    """Hard constraints on the evidence universe."""

    allowed_filing_types: list[FilingType] = Field(
        default_factory=lambda: [
            FilingType.FORM_10K,
            FilingType.FORM_10Q,
            FilingType.FORM_8K,
        ]
    )
    transcript_optional: bool = True
    domestic_filers_only: bool = True
    universe_name: str = "NASDAQ_100"
    forbid_external_sources: bool = True


class GraphRuntimeSettings(BaseModel):
    """Graph runtime limits and verifier behavior."""

    max_reretrievals: int = Field(default=2, ge=0, le=5)
    verifier_mode: str = "confidence_calibrated"
    checkpointer_backend: str = "memory"
    checkpoint_namespace: str = "research"
    enable_node_logging: bool = True
    enable_state_snapshot_logging: bool = True
    trace_dir: str = "./data/graph_traces"


class SecSettings(BaseModel):
    """SEC submissions metadata fetch settings."""

    base_url: str = "https://data.sec.gov/submissions"
    user_agent: str = "EMS Research contact@example.com"
    edgar_identity: str | None = None
    raw_sec_dir: Path = Path("data/raw/sec")
    timeout_seconds: float = Field(default=30.0, gt=0)
    requests_per_second: float = Field(default=5.0, gt=0)
    max_retries: int = Field(default=3, ge=0, le=10)
    backoff_seconds: float = Field(default=0.5, ge=0)
    save_raw_json: bool = False
    skip_existing_downloads: bool = True


class TranscriptSettings(BaseModel):
    """Transcript ingestion settings."""

    default_source_reliability: str = "medium"
    skip_existing_imports: bool = True


class LlmSettings(BaseModel):
    """LLM runtime settings for reasoning adapters."""

    enabled: bool = False
    provider: str = "gemini"
    model: str = "gemini-2.5-flash"
    api_key: str | None = None
    base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    timeout_seconds: float = Field(default=30.0, gt=0)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)


class IndexingSettings(BaseModel):
    """Retrieval indexing settings."""

    chunk_size_chars: int = Field(default=1200, gt=0)
    chunk_overlap_chars: int = Field(default=150, ge=0)
    embedding_dimensions: int = Field(default=12, gt=0)


class Settings(BaseSettings):
    """Application settings loaded from the environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="APP_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    app_name: str = "EMS Research"
    environment: RuntimeEnvironment = RuntimeEnvironment.LOCAL
    api: ApiSettings = Field(default_factory=ApiSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    source_scope: SourceScopeSettings = Field(default_factory=SourceScopeSettings)
    graph: GraphRuntimeSettings = Field(default_factory=GraphRuntimeSettings)
    sec: SecSettings = Field(default_factory=SecSettings)
    transcript: TranscriptSettings = Field(default_factory=TranscriptSettings)
    llm: LlmSettings = Field(default_factory=LlmSettings)
    indexing: IndexingSettings = Field(default_factory=IndexingSettings)

    @property
    def name(self) -> str:
        """Backward-compatible app name accessor."""

        return self.app_name

    @property
    def database_url(self) -> str:
        """Backward-compatible database URL accessor."""

        return self.storage.database_url

    @property
    def api_host(self) -> str:
        """Backward-compatible host accessor."""

        return self.api.host

    @property
    def api_port(self) -> int:
        """Backward-compatible port accessor."""

        return self.api.port


@lru_cache
def get_settings() -> Settings:
    """Return a cached settings object."""

    return Settings()
