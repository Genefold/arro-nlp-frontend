"""FastAPI application entry point.

Lifespan handles startup/shutdown. Embedder, store, arro_client, and
ingest_locks are initialised once and injected via app.state.
"""

from __future__ import annotations

import asyncio  # noqa: F401  -- kept for type annotations and future use
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI

from arro_nlp_frontend.arro_client import ArroClient
from arro_nlp_frontend.config import settings
from arro_nlp_frontend.embedder import Embedder
from arro_nlp_frontend.ingest import router as ingest_router
from arro_nlp_frontend.search import router as search_router
from arro_nlp_frontend.store import DocumentStore

logger = logging.getLogger(__name__)


def _check_single_worker() -> None:
    """Warn loudly if the process appears to be running in multi-worker mode.

    arro-nlp-frontend uses a process-local asyncio.Lock for ingest
    serialisation. Running with multiple uvicorn worker processes (--workers N
    or WEB_CONCURRENCY > 1) will cause concurrent workers to compute the same
    next_row_index(), silently corrupting the SQLite document store and the
    Zarr vector array on arro-server.

    This function checks two signals:
    - The ``WEB_CONCURRENCY`` environment variable (set by platforms such as
      Railway, Fly.io, and Render based on CPU count).
    - The ``UVICORN_WORKERS`` environment variable (used by some deployment
      configurations).

    A CRITICAL log is emitted when either variable is set to a value other
    than "1". The application is NOT hard-crashed because the guard must also
    work correctly in test environments where those env vars may be set for
    unrelated reasons. Platform operators are responsible for ensuring
    WEB_CONCURRENCY=1.
    """
    import os

    for env_var in ("WEB_CONCURRENCY", "UVICORN_WORKERS"):
        value = os.environ.get(env_var, "1")
        if value != "1":
            logger.critical(
                "[startup] %s=%s detected. arro-nlp-frontend does NOT support "
                "multi-worker mode. The per-dataset asyncio.Lock is process-local "
                "and WILL NOT prevent row index corruption across workers. "
                "Set %s=1 to suppress this warning.",
                env_var,
                value,
                env_var,
            )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown.

    ArroClient is instantiated before the try block so its underlying
    httpx.AsyncClient is always closed in the finally block, regardless of
    whether downstream startup steps (Embedder, DocumentStore) raise.

    Startup sequence
    ----------------
    1. ArroClient created (before try — ensures aclose() is always called).
    2. Multi-worker guard (_check_single_worker) — logs CRITICAL if WEB_CONCURRENCY
       or UVICORN_WORKERS > 1 (first statement inside try).
    3. Embedder.from_settings() — loads the sentence-transformer or OpenAI backend.
    4. DocumentStore(...) — opens SQLite, applies schema, detects v1 migration need.
    5. ingest_locks dict initialised.
    6. Startup health probe against arro-server (non-fatal on failure).

    Shutdown sequence (finally)
    ---------------------------
    - arro_client.aclose() — always executed.
    - app.state.store.close() — only if DocumentStore was successfully assigned.
    """
    arro_client = ArroClient(base_url=settings.arro_server_url)
    try:
        _check_single_worker()
        app.state.embedder = Embedder.from_settings()
        app.state.store = DocumentStore(Path(settings.store_db_path))
        app.state.ingest_locks = {}
        app.state.arro_client = arro_client

        logger.info(
            "Loading embedder: backend=%s model=%s path=%r scale=%s",
            settings.embed_backend,
            settings.embed_model,
            settings.embedder_model_path or "(HF Hub)",
            settings.embed_scale_factor,
        )
        logger.info("Embedder ready. dim=%d", app.state.embedder.dim)

        try:
            await arro_client._client.get("/health", timeout=3.0)
            logger.info("[startup] arro-server is reachable.")
        except httpx.RequestError:
            logger.warning(
                "[startup] Could not reach arro-server. Search will fail until it is available."
            )

        yield

    finally:
        await arro_client.aclose()
        if hasattr(app.state, "store"):
            app.state.store.close()


def create_app() -> FastAPI:
    """Application factory. Returns a configured FastAPI instance."""
    app = FastAPI(
        title="arro-nlp-frontend",
        description="OpenAI-compatible NLP frontend for arro-server",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.include_router(ingest_router)
    app.include_router(search_router)

    @app.get("/health", tags=["ops"])
    async def health() -> dict:
        """Liveness probe. Returns 200 when the embedder is loaded."""
        return {
            "status": "ok",
            "embed_backend": settings.embed_backend,
            "embed_model": settings.embed_model,
            "embedder_dim": app.state.embedder.dim,
        }

    return app


def run() -> None:
    """Entry point for `arro-nlp-frontend` CLI command.

    workers is hard-coded to 1. arro-nlp-frontend uses a process-local
    asyncio.Lock for ingest serialisation; multi-worker mode would silently
    corrupt row indices. See issue #27.
    """
    uvicorn.run(
        "arro_nlp_frontend.main:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
        log_level="info",
        reload=False,
        workers=1,
    )


if __name__ == "__main__":
    run()
