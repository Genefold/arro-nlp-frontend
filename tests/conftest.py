"""Shared fixtures for arro-nlp-frontend tests.

All fixtures are offline — no arro-server, no HF Hub downloads at test time
(sentence-transformers caches the model after first download).
"""

from __future__ import annotations

import asyncio
from collections.abc import Generator
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from arro_nlp_frontend.arro_client import ArroClient
from arro_nlp_frontend.embedder import Embedder
from arro_nlp_frontend.main import create_app
from arro_nlp_frontend.store import DocumentStore


@asynccontextmanager
async def _noop_lifespan(app):
    yield


@pytest.fixture(scope="session")
def local_embedder() -> Embedder:
    """Real all-MiniLM-L6-v2 embedder. Downloaded once, cached by sentence-transformers."""
    return Embedder(backend="local", model="all-MiniLM-L6-v2", scale_factor=1.0)


@pytest.fixture
def store(tmp_path: Path) -> DocumentStore:
    """DocumentStore with SQLite backed by a temp file (deleted after test)."""
    with DocumentStore(tmp_path / "ingest_test.sqlite") as s:
        yield s


@pytest.fixture
def mock_arro_client() -> AsyncMock:
    """Mock arro-client with push_vectors & row_count."""
    client = AsyncMock(spec=ArroClient)
    client.push_vectors = AsyncMock(return_value=None)
    client.row_count = AsyncMock(return_value=0)
    return client


@pytest.fixture
def ingest_client(
    store: DocumentStore,
    mock_arro_client: AsyncMock,
) -> Generator[tuple[TestClient, DocumentStore, AsyncMock], None, None]:
    """TestClient with pre-injected store, mock arro_client, and real ingest_lock.

    Returns a 3-tuple: (client, store, mock_arro_client).
    """
    with patch("arro_nlp_frontend.main.lifespan", _noop_lifespan):
        app = create_app()
        app.state.embedder = Embedder(backend="local", model="all-MiniLM-L6-v2", scale_factor=1.0)
        app.state.store = store
        app.state.arro_client = mock_arro_client
        app.state.ingest_lock = asyncio.Lock()
        with TestClient(app, raise_server_exceptions=True) as client:
            yield client, store, mock_arro_client


@pytest.fixture(scope="session")
def app_client(local_embedder: Embedder) -> Generator[TestClient, None, None]:
    """FastAPI TestClient with embedder pre-loaded in app.state."""
    app = create_app()
    app.state.embedder = local_embedder
    with TestClient(app, raise_server_exceptions=True) as client:
        yield client
