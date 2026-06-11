"""Shared fixtures for arro-nlp-frontend tests.

All fixtures are offline — no arro-server, no HF Hub downloads at test time
(sentence-transformers caches the model after first download).
"""

from __future__ import annotations

from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient

from arro_nlp_frontend.embedder import Embedder
from arro_nlp_frontend.main import create_app


@pytest.fixture(scope="session")
def local_embedder() -> Embedder:
    """Real all-MiniLM-L6-v2 embedder. Downloaded once, cached by sentence-transformers."""
    return Embedder(backend="local", model="all-MiniLM-L6-v2", scale_factor=1.0)


@pytest.fixture(scope="session")
def app_client(local_embedder: Embedder) -> Generator[TestClient, None, None]:
    """FastAPI TestClient with embedder pre-loaded in app.state."""
    app = create_app()
    app.state.embedder = local_embedder  # inject before startup fires
    with TestClient(app, raise_server_exceptions=True) as client:
        yield client
