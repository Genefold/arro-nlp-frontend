"""Shared fixtures for arro-nlp-frontend tests.

All fixtures are offline -- no arro-server, no HF Hub downloads at test time
(sentence-transformers caches the model after first download).

Lifespan isolation strategy
----------------------------
Every fixture that builds a TestClient patches `arro_nlp_frontend.main.lifespan`
with a no-op context manager, then injects all required app.state attributes
manually. This prevents:
  - real ArroClient connecting to localhost:8001
  - real DocumentStore being created at ./data/documents.sqlite
  - DeprecationWarnings from on_event (now fully removed)
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from arro_nlp_frontend.arro_client import (
    ArroClient,
    UploadCommitResult,
    VectorAppendResult,
    VectorOverwriteResult,
)
from arro_nlp_frontend.embedder import Embedder
from arro_nlp_frontend.main import create_app
from arro_nlp_frontend.store import DocumentStore

DEFAULT_DS = "test/dataset"


@asynccontextmanager
async def _noop_lifespan(app):
    """Drop-in replacement for the real lifespan: does nothing on startup/shutdown."""
    yield


# ---------------------------------------------------------------------------
# Shared singletons (session-scoped to avoid re-downloading the model)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def local_embedder() -> Embedder:
    """Real all-MiniLM-L6-v2 embedder. Downloaded once, cached by sentence-transformers."""
    return Embedder(backend="local", model="all-MiniLM-L6-v2", scale_factor=1.0)


# ---------------------------------------------------------------------------
# Per-test fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path: Path) -> Generator[DocumentStore, None, None]:
    """DocumentStore backed by a temp SQLite file, closed after each test."""
    with DocumentStore(tmp_path / "ingest_test.sqlite") as s:
        yield s


@pytest.fixture
def mock_arro_client() -> AsyncMock:
    """AsyncMock of ArroClient with all methods pre-configured.

    New methods for incremental ingest (issue #21):
      - append_vectors:    returns VectorAppendResult(start_row=0, appended=1, new_shape=[1, 384])
      - overwrite_vectors: returns VectorOverwriteResult(overwritten=1)
      - get_vector_count:  returns 0 (empty dataset by default)

    Tests that require specific return values should override these defaults
    using mock_arro.append_vectors.return_value = VectorAppendResult(...).
    """
    client = AsyncMock(spec=ArroClient)
    client.dataset_metadata  = AsyncMock(return_value=None)
    client.upload_init       = AsyncMock(return_value="/tmp/arro_test_upload.zarr")
    client.upload_commit     = AsyncMock(
        return_value=UploadCommitResult(index_stale=False, shape=[0, 384])
    )
    client.build_index       = AsyncMock(return_value=None)
    client.search            = AsyncMock(return_value=[])
    client.append_vectors    = AsyncMock(
        return_value=VectorAppendResult(start_row=0, appended=1, new_shape=[1, 384])
    )
    client.overwrite_vectors = AsyncMock(
        return_value=VectorOverwriteResult(overwritten=1)
    )
    client.get_vector_count  = AsyncMock(return_value=0)
    return client


@pytest.fixture
def ingest_client(
    store: DocumentStore,
    mock_arro_client: AsyncMock,
    local_embedder: Embedder,
) -> Generator[tuple[TestClient, DocumentStore, AsyncMock], None, None]:
    """TestClient with lifespan patched out, store and arro_client injected.

    Returns a 3-tuple: (client, store, mock_arro_client).
    """
    with patch("arro_nlp_frontend.main.lifespan", _noop_lifespan):
        app = create_app()
        app.state.embedder = local_embedder
        app.state.store = store
        app.state.arro_client = mock_arro_client
        app.state.ingest_locks = {}
        with TestClient(app, raise_server_exceptions=True) as client:
            yield client, store, mock_arro_client


@pytest.fixture
def search_client(
    store: DocumentStore,
    mock_arro_client: AsyncMock,
    local_embedder: Embedder,
) -> Generator[tuple[TestClient, DocumentStore, AsyncMock], None, None]:
    """TestClient wired for search tests.

    Identical injection pattern to ingest_client. Returns (client, store, mock_arro).
    search is pre-configured in mock_arro_client to return an empty list by default.
    """
    with patch("arro_nlp_frontend.main.lifespan", _noop_lifespan):
        app = create_app()
        app.state.embedder = local_embedder
        app.state.store = store
        app.state.arro_client = mock_arro_client
        app.state.ingest_locks = {}
        with TestClient(app, raise_server_exceptions=True) as client:
            yield client, store, mock_arro_client


@pytest.fixture(scope="session")
def app_client(local_embedder: Embedder) -> Generator[TestClient, None, None]:
    """TestClient for smoke tests (/health, /openapi.json).

    Lifespan is patched out to prevent network calls to arro-server and
    side-effect writes to ./data/documents.sqlite.
    store is a real in-memory DocumentStore (path=:memory: not supported by
    our impl, so we use a tmp dir via tempfile).
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("arro_nlp_frontend.main.lifespan", _noop_lifespan):
            app = create_app()
            app.state.embedder = local_embedder
            # /health only needs embedder; store and arro_client not accessed
            # but set defensively so any future endpoint that reads them
            # gets a real object rather than an AttributeError.
            app.state.store = DocumentStore(Path(tmpdir) / "smoke.sqlite")
            app.state.arro_client = AsyncMock(spec=ArroClient)
            app.state.ingest_locks = {}
            with TestClient(app, raise_server_exceptions=True) as client:
                yield client
