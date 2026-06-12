"""FastAPI application entry point.

Lifespan handles startup/shutdown. Embedder, store, arro_client, and
ingest_lock are initialised once and injected via app.state.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI

from arro_nlp_frontend.arro_client import ArroClient, ArroServerError
from arro_nlp_frontend.config import settings
from arro_nlp_frontend.embedder import Embedder
from arro_nlp_frontend.ingest import router as ingest_router
from arro_nlp_frontend.store import DocumentStore

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown.

    On startup: initialises Embedder, DocumentStore, ArroClient, and the
    asyncio.Lock used to serialise ingest operations. Logs embedder config
    and performs a consistency check between the arro-server dataset shape
    and the local document store row count.

    On shutdown: closes the ArroClient httpx session and the SQLite connection.
    """
    app.state.embedder = Embedder.from_settings()
    app.state.store = DocumentStore(Path(settings.store_db_path))
    app.state.ingest_lock = asyncio.Lock()
    app.state.arro_client = ArroClient(
        base_url=settings.arro_server_url,
        dataset_id=settings.arro_server_dataset_id,
        root_label=settings.arro_server_root_label,
    )

    logger.info(
        "Loading embedder: backend=%s model=%s path=%r scale=%s",
        settings.embed_backend,
        settings.embed_model,
        settings.embedder_model_path or "(HF Hub)",
        settings.embed_scale_factor,
    )
    logger.info("Embedder ready. dim=%d", app.state.embedder.dim)

    try:
        meta = await app.state.arro_client.dataset_metadata()
        arro_rows = meta["shape"][0] if meta is not None else 0
        store_rows = app.state.store.count()
        if arro_rows != store_rows:
            logger.warning(
                "[startup] arro-server has %d rows, store has %d documents. "
                "If arro-server was rebuilt, the store must be rebuilt too.",
                arro_rows,
                store_rows,
            )
    except ArroServerError:
        logger.warning("[startup] Could not reach arro-server for metadata check.")

    yield

    await app.state.arro_client.aclose()
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
    """Entry point for `arro-nlp-frontend` CLI command."""
    uvicorn.run(
        "arro_nlp_frontend.main:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
        log_level="info",
        reload=False,
    )


if __name__ == "__main__":
    run()
