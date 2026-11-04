"""
config.py — Centralised application configuration using pydantic-settings.

All settings are loaded from environment variables or .env file.
Type-safe, validated, and dependency-injectable throughout the application.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List, Union

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide configuration.

    Loaded from environment variables and .env file.
    Use :func:`get_settings` to obtain the singleton instance.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM ───────────────────────────────────────────────────────────────────
    gemini_api_key: str = Field(default="", description="Google Gemini API key")
    openai_api_key: str = Field(default="", description="OpenAI API key")
    anthropic_api_key: str = Field(default="", description="Anthropic API key")

    default_llm_provider: str = Field(default="gemini", description="Default LLM provider")
    default_llm_model: str = Field(default="gemini-1.5-pro", description="Default LLM model")

    # ── Embeddings ────────────────────────────────────────────────────────────
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence-transformers model name",
    )

    # ── Database ──────────────────────────────────────────────────────────────
    database_url: str = Field(
        default="sqlite+aiosqlite:///./data/agent.db",
        description="SQLAlchemy async database URL",
    )
    chroma_db_path: str = Field(
        default="./data/embeddings",
        description="ChromaDB persistence directory",
    )

    # ── API ───────────────────────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    api_debug: bool = Field(default=False)
    secret_key: str = Field(default="change-me", description="JWT / session secret key")

    cors_origins: Union[List[str], str] = Field(
        default=["http://localhost:5173", "http://localhost:3000"],
        description="Allowed CORS origins",
    )

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str = Field(default="INFO")
    log_dir: str = Field(default="./logs")

    # ── Agent ─────────────────────────────────────────────────────────────────
    agent_name: str = Field(default="Sentia")
    max_iterations: int = Field(default=15, ge=1, le=50)
    max_memory_items: int = Field(default=100, ge=10)
    experience_replay_batch: int = Field(default=16, ge=1)

    # ── Web Search ────────────────────────────────────────────────────────────
    max_search_results: int = Field(default=5, ge=1, le=20)
    request_timeout: int = Field(default=30, ge=5)

    # ── Derived properties ────────────────────────────────────────────────────
    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: object) -> List[str]:
        """Allow CORS_ORIGINS to be a comma-separated string in .env."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v  # type: ignore[return-value]

    @property
    def log_dir_path(self) -> Path:
        return Path(self.log_dir)

    @property
    def chroma_db_dir(self) -> Path:
        return Path(self.chroma_db_path)

    @property
    def data_dir(self) -> Path:
        return Path("./data")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached application settings singleton."""
    return Settings()
