"""FastAPI application entry point.

This is the scaffold for the arro-nlp-frontend server.
Endpoints (search, embed, health) will be added in the next phase.
The Embedder is instantiated once at startup and injected via app.state.
"""

from __future__ import annotations

import logging

import uvicorn
from fastapi import FastAPI

from arro_nlp_frontend.config import settings
from arro_nlp_frontend.embedder import Embedder

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Application factory. Returns a configured FastAPI instance."""
    app = FastAPI(
        title="arro-nlp-frontend",
        description="OpenAI-compatible NLP frontend for arro-server",
        version="0.1.0",
    )

    @app.on_event("startup")
    async def _startup() -> None:
        logger.info(
            "Loading embedder: backend=%s model=%s path=%r scale=%s",
            settings.embed_backend,
            settings.embed_model,
            settings.embedder_model_path or "(HF Hub)",
            settings.embed_scale_factor,
        )
        app.state.embedder = Embedder.from_settings()
        logger.info("Embedder ready. dim=%d", app.state.embedder.dim)

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        logger.info("arro-nlp-frontend shutting down.")

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


# Convenience: allows `python -m arro_nlp_frontend.main`
if __name__ == "__main__":
    run()
