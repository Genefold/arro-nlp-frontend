"""Settings loaded from environment / .env at import time.

Priority (pydantic-settings default):
  1. Environment variables
  2. .env file in cwd
  3. Field defaults

Usage:
    from arro_nlp_frontend.config import settings
    print(settings.embed_backend)
"""

from __future__ import annotations

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ── Embedding ──────────────────────────────────────────────────────────────
    embed_backend: str = "local"
    embed_model: str = "all-MiniLM-L6-v2"
    embedder_model_path: str = ""
    embed_scale_factor: float = 1.0
    openai_api_key: str = ""

    # ── arro-server ────────────────────────────────────────────────────────────
    arro_server_url: str = "http://localhost:8001"
    arro_server_dataset_id: str = "cve/embeddings"
    arro_server_root_label: str = "main"
    arro_server_upload_path: str = ""
    """Override the upload path returned by /api/upload/init.

    Leave empty (default) to use the path arro-server provides.
    Set this only when arro-server and arro-nlp-frontend share a
    filesystem volume and the arro-server path is accessible locally
    under a different mount point. upload_init is ALWAYS called
    regardless — this only overrides the path used for the local
    Zarr write and the subsequent upload_commit body.
    """
    arro_server_search_tau: float = 0.42

    # ── Document store ─────────────────────────────────────────────────────────
    store_db_path: str = "./data/documents.sqlite"

    # ── Ingest ─────────────────────────────────────────────────────────────────
    ingest_batch_size: int = 100

    # ── Server ─────────────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("embed_backend")
    @classmethod
    def _valid_backend(cls, v: str) -> str:
        if v not in ("local", "openai"):
            raise ValueError(f"EMBED_BACKEND must be 'local' or 'openai', got {v!r}")
        return v

    @field_validator("embed_scale_factor")
    @classmethod
    def _positive_scale(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"EMBED_SCALE_FACTOR must be > 0, got {v}")
        return v

    @field_validator("arro_server_search_tau")
    @classmethod
    def _valid_tau(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(
                f"ARRO_SERVER_SEARCH_TAU must be in [0, 1], got {v}. "
                "Suggested: 0.42 (spectral), 0.70 (hybrid), 1.00 (cosine)"
            )
        return v

    @model_validator(mode="after")
    def _openai_needs_key(self) -> Settings:
        if self.embed_backend == "openai" and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when EMBED_BACKEND=openai")
        return self


settings = Settings()
