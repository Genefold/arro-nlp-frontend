"""Tests for the POST /ingest endpoint. ALL tests run offline (no arro-server).

Tests
-----
1.  Single document ingest returns 200
2.  Batch yields contiguous row indices
3.  Status reflects "created" vs. "updated"
4.  Documents are persisted in the store
5.  Vectors are stored in SQLite (get_all_vectors)
6.  arro-server sync methods are called (upload_init, upload_commit, etc.)
7.  Duplicate doc_ids in request yield 422
8.  Empty batch yields 422
9.  arro-server failure yields 502 (store is written first)
10. arro-server failure leaves all local docs intact
11. Second batch starts at correct row_index
12. start_row uses MAX(row_index)+1, not COUNT (soft-delete safety)
13. Concurrent batches do not overlap row indices
14. duration_ms is present and non-negative
15. Metadata round-trips correctly
16.  Two datasets have independent row indices
17.  root_label override forwarded to upload_init
18.  root_label defaults to settings.arro_server_root_label
19.  Missing dataset_id yields 422
20.  Incremental metadata-only skips build_index
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import numpy as np
import pytest

from arro_nlp_frontend.arro_client import ArroClient, ArroServerError, UploadCommitResult
from arro_nlp_frontend.embedder import Embedder
from arro_nlp_frontend.main import create_app
from arro_nlp_frontend.store import DocumentStore

DEFAULT_DS = "test/dataset"


@asynccontextmanager
async def _noop_lifespan(app):
    yield


def _post(client, docs, dataset_id: str = DEFAULT_DS, root_label: str = ""):
    body = {"dataset_id": dataset_id, "documents": docs}
    if root_label:
        body["root_label"] = root_label
    return client.post("/ingest", json=body)


def _zarr_mock() -> tuple[MagicMock, list[np.ndarray]]:
    """Returns (mock_open, written_chunks).

    written_chunks is populated via __setitem__ when arr[:] = vectors is called.
    """
    written: list[np.ndarray] = []
    mock_arr = MagicMock()
    mock_arr.__setitem__ = lambda self, key, value: written.append(value)  # type: ignore[method-assign]
    mock_open = MagicMock(return_value=mock_arr)
    return mock_open, written


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_ingest_single_doc_returns_200(ingest_client):
    client, _, _ = ingest_client
    r = _post(client, [{"doc_id": "doc0", "text": "hello"}])
    assert r.status_code == 200
    body = r.json()
    assert body["ingested"] == 1
    assert body["results"][0]["row_index"] == 0
    assert body["results"][0]["status"] == "created"


def test_ingest_batch_row_indices_contiguous(ingest_client):
    client, _, _ = ingest_client
    r = _post(
        client,
        [
            {"doc_id": "doc0", "text": "text0"},
            {"doc_id": "doc1", "text": "text1"},
            {"doc_id": "doc2", "text": "text2"},
        ],
    )
    assert r.status_code == 200
    indices = [res["row_index"] for res in r.json()["results"]]
    assert indices == [0, 1, 2]


def test_ingest_status_created_vs_updated(ingest_client):
    client, _, mock_arro = ingest_client
    r1 = _post(client, [{"doc_id": "doc0", "text": "hello"}])
    assert r1.json()["results"][0]["status"] == "created"
    mock_arro.get_vector_count = AsyncMock(return_value=1)
    r2 = client.post(
        "/ingest",
        json={
            "dataset_id": DEFAULT_DS,
            "documents": [{"doc_id": "doc0", "text": "updated text"}],
            "incremental": True,
        },
    )
    assert r2.status_code == 200, f"r2={r2.text}"
    assert r2.json()["results"][0]["status"] == "updated"


def test_ingest_persists_in_store(ingest_client):
    client, store, _ = ingest_client
    _post(client, [{"doc_id": "doc0", "text": "hello"}])
    doc = store.get_by_id(DEFAULT_DS, "doc0")
    assert doc is not None
    assert doc.text == "hello"


def test_ingest_vectors_stored_in_sqlite(ingest_client):
    """Vectors are persisted in SQLite and readable via get_all_vectors."""
    client, store, _ = ingest_client
    _post(client, [{"doc_id": "doc0", "text": "hello"}])
    all_v = store.get_all_vectors(DEFAULT_DS)
    assert all_v.shape[0] == 1
    assert all_v.dtype == np.float64


def test_ingest_arro_sync_called(ingest_client):
    """All arro-server sync methods are called in correct order."""
    client, _, mock_arro = ingest_client
    mock_arro.dataset_metadata = AsyncMock(return_value={"shape": [0, 384]})
    mock_open, written = _zarr_mock()
    with patch("arro_nlp_frontend.ingest.zarr.open_array", mock_open):
        _post(client, [{"doc_id": "doc0", "text": "hello"}])

    mock_arro.upload_init.assert_called_once_with(dataset_id=DEFAULT_DS, root_label="main")
    mock_arro.upload_commit.assert_called_once()
    mock_arro.build_index.assert_not_called()
    assert len(written) == 1
    assert written[0].shape[0] == 1
    assert written[0].dtype == np.float64


def test_ingest_duration_ms_present(ingest_client):
    client, _, _ = ingest_client
    r = _post(client, [{"doc_id": "doc0", "text": "hello world"}])
    assert r.status_code == 200
    body = r.json()
    assert "duration_ms" in body
    assert isinstance(body["duration_ms"], int)
    assert body["duration_ms"] >= 0


def test_ingest_metadata_stored(ingest_client):
    client, store, _ = ingest_client
    meta = {"source": "web", "priority": 5, "notes": ["never"], "empty": {}}
    r = _post(client, [{"doc_id": "docX", "text": "hello", "metadata": meta}])
    assert r.status_code == 200
    doc = store.get_by_id(DEFAULT_DS, "docX")
    assert doc is not None
    assert doc.metadata == meta


# ---------------------------------------------------------------------------
# Row index correctness
# ---------------------------------------------------------------------------


def test_ingest_second_batch_start_row_correct(ingest_client):
    """Second batch of 2 docs starts at row_index=2, not 0."""
    client, _, _ = ingest_client
    _post(
        client,
        [
            {"doc_id": "doc0", "text": "text0"},
            {"doc_id": "doc1", "text": "text1"},
        ],
    )
    r = _post(
        client,
        [
            {"doc_id": "doc2", "text": "text2"},
            {"doc_id": "doc3", "text": "text3"},
        ],
    )
    assert r.status_code == 200
    results = r.json()["results"]
    assert results[0]["row_index"] == 2
    assert results[1]["row_index"] == 3


def test_ingest_start_row_uses_max_not_count(ingest_client):
    """After soft-deleting row 1, next ingest gets row_index=3, not row_index=2."""
    client, store, _ = ingest_client
    _post(
        client,
        [
            {"doc_id": "doc0", "text": "text0"},
            {"doc_id": "doc1", "text": "text1"},
            {"doc_id": "doc2", "text": "text2"},
        ],
    )
    store.delete_by_id(DEFAULT_DS, "doc1")  # ghost row at index 1
    r = _post(client, [{"doc_id": "doc3", "text": "text3"}])
    assert r.status_code == 200
    assert r.json()["results"][0]["row_index"] == 3


def test_ingest_concurrent_batches_no_row_overlap(tmp_path: Path) -> None:
    """Three concurrent ingest requests never assign overlapping row indices.

    Uses httpx.AsyncClient with ASGITransport to allow real async concurrency.
    Lifespan is patched to prevent network calls to arro-server.
    """
    with patch("arro_nlp_frontend.main.lifespan", _noop_lifespan):
        app = create_app()
        app.state.embedder = Embedder(backend="local", model="all-MiniLM-L6-v2", scale_factor=1.0)
        app.state.store = DocumentStore(tmp_path / "concurrent_test.sqlite")
        mock_arro = AsyncMock(spec=ArroClient)
        mock_arro.dataset_metadata = AsyncMock(return_value=None)
        mock_arro.upload_init = AsyncMock(return_value="/tmp/test_upload.zarr")
        mock_arro.upload_commit = AsyncMock(
            return_value=UploadCommitResult(index_stale=False, shape=[1, 384])
        )
        mock_arro.build_index = AsyncMock(return_value=None)
        app.state.arro_client = mock_arro
        app.state.ingest_locks = {}

    transport = httpx.ASGITransport(app=app)

    mock_open_zarr, _ = _zarr_mock()

    async def _post_ingest(doc_id: str) -> list[int]:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            with patch("arro_nlp_frontend.ingest.zarr.open_array", mock_open_zarr):
                r = await c.post(
                    "/ingest",
                    json={
                        "dataset_id": DEFAULT_DS,
                        "documents": [{"doc_id": doc_id, "text": "text"}],
                    },
                )
            assert r.status_code == 200, f"Unexpected {r.status_code}: {r.text}"
            return [x["row_index"] for x in r.json()["results"]]

    async def _run() -> None:
        results = await asyncio.gather(
            _post_ingest("a"),
            _post_ingest("b"),
            _post_ingest("c"),
        )
        all_indices = sorted(i for batch in results for i in batch)
        assert len(all_indices) == 3
        assert all_indices == [0, 1, 2], f"Row index collision: {all_indices}"

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_ingest_duplicate_doc_ids_in_request_422(ingest_client):
    client, _, _ = ingest_client
    r = _post(
        client,
        [
            {"doc_id": "dup", "text": "hello"},
            {"doc_id": "dup", "text": "world"},
        ],
    )
    assert r.status_code == 422


def test_ingest_empty_list_422(ingest_client):
    client, _, _ = ingest_client
    r = client.post("/ingest", json={"dataset_id": DEFAULT_DS, "documents": []})
    assert r.status_code == 422


def test_ingest_arro_server_502_returns_error(ingest_client):
    """If arro-server sync fails pre-commit, 502 is returned and SQLite is rolled back."""
    client, store, mock_arro = ingest_client
    mock_arro.upload_init.side_effect = ArroServerError("mocked failure")
    mock_open, _ = _zarr_mock()
    with patch("arro_nlp_frontend.ingest.zarr.open_array", mock_open):
        r = _post(client, [{"doc_id": "doc0", "text": "hello"}])
    assert r.status_code == 502
    assert store.count(DEFAULT_DS) == 0


def test_ingest_arro_server_502_leaves_previous_intact(ingest_client):
    """A failed sync after a successful one leaves the FIRST batch intact,
    but rolls back the SECOND (failed) batch."""
    client, store, mock_arro = ingest_client
    mock_open, _ = _zarr_mock()
    with patch("arro_nlp_frontend.ingest.zarr.open_array", mock_open):
        _post(client, [{"doc_id": "doc0", "text": "hello"}])
    assert store.count(DEFAULT_DS) == 1

    mock_arro.upload_init.side_effect = ArroServerError("mocked failure")
    mock_open2, _ = _zarr_mock()
    with patch("arro_nlp_frontend.ingest.zarr.open_array", mock_open2):
        r = _post(client, [{"doc_id": "doc1", "text": "world"}])
    assert r.status_code == 502
    assert store.count(DEFAULT_DS) == 1
    assert store.get_by_id(DEFAULT_DS, "doc0") is not None
    assert store.get_by_id(DEFAULT_DS, "doc1") is None





def test_incremental_ingest_skips_build_index_for_metadata_only(
    ingest_client,
) -> None:
    """Incremental must NOT call build_index for metadata-only batch (issue #26)."""
    client, store, mock_arro = ingest_client

    # First ingest to populate the store
    mock_open1, _ = _zarr_mock()
    with patch("arro_nlp_frontend.ingest.zarr.open_array", mock_open1):
        _post(client, [{"doc_id": "doc0", "text": "existing text"}])
    mock_arro.build_index.reset_mock()

    # Second ingest with incremental=True, same text → metadata-only
    mock_open2, _ = _zarr_mock()
    with patch("arro_nlp_frontend.ingest.zarr.open_array", mock_open2):
        r = client.post(
            "/ingest",
            json={
                "dataset_id": DEFAULT_DS,
                "documents": [{"doc_id": "doc0", "text": "existing text"}],
                "incremental": True,
            },
        )
    assert r.status_code == 200
    mock_arro.build_index.assert_not_called()





# ---------------------------------------------------------------------------
# Zarr array content verification
# ---------------------------------------------------------------------------


def test_ingest_zarr_array_written_with_correct_vectors(ingest_client):
    """Vectors actually written to Zarr array -- not empty."""
    client, store, _ = ingest_client
    mock_open, written = _zarr_mock()
    with patch("arro_nlp_frontend.ingest.zarr.open_array", mock_open):
        _post(client, [{"doc_id": "doc0", "text": "hello"}])
    assert len(written) == 1
    assert isinstance(written[0], np.ndarray)
    assert written[0].shape[0] == 1
    assert written[0].dtype == np.float64


# ---------------------------------------------------------------------------
# Multi-dataset tests
# ---------------------------------------------------------------------------


def test_ingest_two_datasets_independent_row_indices(ingest_client):
    """Each dataset has its own independent row_index counter."""
    client, store, mock_arro = ingest_client
    mock_arro.dataset_metadata = AsyncMock(return_value={"shape": [0, 384]})
    mock_open, _ = _zarr_mock()

    with patch("arro_nlp_frontend.ingest.zarr.open_array", mock_open):
        r_a = _post(client, [{"doc_id": "doc0", "text": "a"}], dataset_id="ds/a")
    assert r_a.status_code == 200
    assert r_a.json()["results"][0]["row_index"] == 0

    with patch("arro_nlp_frontend.ingest.zarr.open_array", mock_open):
        r_b = _post(client, [{"doc_id": "doc1", "text": "b"}], dataset_id="ds/b")
    assert r_b.status_code == 200
    assert r_b.json()["results"][0]["row_index"] == 0

    with patch("arro_nlp_frontend.ingest.zarr.open_array", mock_open):
        r_a2 = _post(client, [{"doc_id": "doc2", "text": "a2"}], dataset_id="ds/a")
    assert r_a2.status_code == 200
    assert r_a2.json()["results"][0]["row_index"] == 1

    assert store.get_by_id("ds/a", "doc0").row_index == 0  # type: ignore[union-attr]
    assert store.get_by_id("ds/b", "doc1").row_index == 0  # type: ignore[union-attr]
    assert store.get_by_id("ds/a", "doc2").row_index == 1  # type: ignore[union-attr]


def test_ingest_root_label_override_forwarded(ingest_client):
    """root_label='staging' is forwarded to upload_init."""
    client, _, mock_arro = ingest_client
    mock_arro.dataset_metadata = AsyncMock(return_value={"shape": [0, 384]})
    mock_open, _ = _zarr_mock()
    with patch("arro_nlp_frontend.ingest.zarr.open_array", mock_open):
        _post(client, [{"doc_id": "doc0", "text": "hello"}], root_label="staging")

    mock_arro.upload_init.assert_called_once_with(dataset_id=DEFAULT_DS, root_label="staging")


def test_ingest_root_label_defaults_to_settings(ingest_client):
    """Omitting root_label uses settings.arro_server_root_label."""
    from arro_nlp_frontend.config import settings

    client, _, mock_arro = ingest_client
    mock_arro.dataset_metadata = AsyncMock(return_value={"shape": [0, 384]})
    mock_open, _ = _zarr_mock()
    with patch("arro_nlp_frontend.ingest.zarr.open_array", mock_open):
        _post(client, [{"doc_id": "doc0", "text": "hello"}])

    mock_arro.upload_init.assert_called_once_with(
        dataset_id=DEFAULT_DS, root_label=settings.arro_server_root_label
    )


def test_ingest_missing_dataset_id_422(ingest_client):
    """Request without dataset_id returns 422."""
    client, _, _ = ingest_client
    r = client.post("/ingest", json={"documents": [{"doc_id": "d", "text": "t"}]})
    assert r.status_code == 422


def test_ingest_concurrent_batches_different_datasets_do_not_block(
    tmp_path: Path,
) -> None:
    """Two concurrent requests targeting DIFFERENT datasets run in parallel.

    With a global lock they would serialise; with per-dataset locks they do not.
    We verify correctness only (no timing assertion -- fragile in CI):
      - ds/a gets row_index=0
      - ds/b gets row_index=0  (independent counter)
      - No collision, no exception.
    """
    with patch("arro_nlp_frontend.main.lifespan", _noop_lifespan):
        app = create_app()
        app.state.embedder = Embedder(backend="local", model="all-MiniLM-L6-v2", scale_factor=1.0)
        app.state.store = DocumentStore(tmp_path / "cross_dataset_concurrent.sqlite")
        mock_arro = AsyncMock(spec=ArroClient)
        mock_arro.dataset_metadata = AsyncMock(return_value=None)
        mock_arro.upload_init = AsyncMock(return_value="/tmp/test_upload.zarr")
        mock_arro.upload_commit = AsyncMock(
            return_value=UploadCommitResult(index_stale=False, shape=[1, 384])
        )
        mock_arro.build_index = AsyncMock(return_value=None)
        app.state.arro_client = mock_arro
        app.state.ingest_locks = {}

    transport = httpx.ASGITransport(app=app)
    mock_open_zarr, _ = _zarr_mock()

    async def _post_ds(dataset_id: str, doc_id: str) -> int:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            with patch("arro_nlp_frontend.ingest.zarr.open_array", mock_open_zarr):
                r = await c.post(
                    "/ingest",
                    json={
                        "dataset_id": dataset_id,
                        "documents": [{"doc_id": doc_id, "text": "text"}],
                    },
                )
        assert r.status_code == 200, f"Unexpected {r.status_code}: {r.text}"
        return int(r.json()["results"][0]["row_index"])

    async def _run() -> None:
        idx_a, idx_b = await asyncio.gather(
            _post_ds("ds/a", "doc-a"),
            _post_ds("ds/b", "doc-b"),
        )
        assert idx_a == 0, f"ds/a got row {idx_a}, expected 0"
        assert idx_b == 0, f"ds/b got row {idx_b}, expected 0"

    asyncio.run(_run())


def test_ingest_concurrent_batches_same_dataset_still_serialised(
    tmp_path: Path,
) -> None:
    """Three concurrent requests targeting the SAME dataset never collide.

    The per-dataset lock must still serialise same-dataset requests.
    Expected row indices: [0, 1, 2] with no duplicates.
    """
    with patch("arro_nlp_frontend.main.lifespan", _noop_lifespan):
        app = create_app()
        app.state.embedder = Embedder(backend="local", model="all-MiniLM-L6-v2", scale_factor=1.0)
        app.state.store = DocumentStore(tmp_path / "same_dataset_concurrent.sqlite")
        mock_arro = AsyncMock(spec=ArroClient)
        mock_arro.dataset_metadata = AsyncMock(return_value=None)
        mock_arro.upload_init = AsyncMock(return_value="/tmp/test_upload.zarr")
        mock_arro.upload_commit = AsyncMock(
            return_value=UploadCommitResult(index_stale=False, shape=[1, 384])
        )
        mock_arro.build_index = AsyncMock(return_value=None)
        app.state.arro_client = mock_arro
        app.state.ingest_locks = {}

    transport = httpx.ASGITransport(app=app)
    mock_open_zarr, _ = _zarr_mock()

    async def _post_same(doc_id: str) -> list[int]:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            with patch("arro_nlp_frontend.ingest.zarr.open_array", mock_open_zarr):
                r = await c.post(
                    "/ingest",
                    json={
                        "dataset_id": DEFAULT_DS,
                        "documents": [{"doc_id": doc_id, "text": "text"}],
                    },
                )
        assert r.status_code == 200, f"Unexpected {r.status_code}: {r.text}"
        return [x["row_index"] for x in r.json()["results"]]

    async def _run() -> None:
        results = await asyncio.gather(
            _post_same("x"),
            _post_same("y"),
            _post_same("z"),
        )
        all_indices = sorted(i for batch in results for i in batch)
        assert all_indices == [0, 1, 2], f"Row index collision: {all_indices}"

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Issue #25 regression tests — partial-write rollback on full-rewrite path
# ---------------------------------------------------------------------------


def test_full_ingest_pre_commit_failure_rolls_back_sqlite(ingest_client) -> None:
    """Issue #25: if upload_commit fails, SQLite must be rolled back.

    After the failure, store.count() must return the pre-ingest value (0).
    next_row_index() must also return 0, so the next call gets start_row=0
    and not start_row=N (which would create a gap in arro-server).
    """
    client, store, mock_arro = ingest_client
    mock_arro.upload_commit.side_effect = ArroServerError("network error")
    mock_open, _ = _zarr_mock()

    with patch("arro_nlp_frontend.ingest.zarr.open_array", mock_open):
        r = _post(client, [{"doc_id": "doc0", "text": "hello"}])

    assert r.status_code == 502
    assert r.headers.get("X-Partial-Write") == "rolled-back"
    assert store.count(DEFAULT_DS) == 0  # rolled back
    assert store.next_row_index(DEFAULT_DS) == 0  # safe to re-ingest


def test_full_ingest_pre_commit_failure_upload_init_rolls_back_sqlite(
    ingest_client,
) -> None:
    """Issue #25: rollback also works when upload_init (not upload_commit) fails."""
    client, store, mock_arro = ingest_client
    mock_arro.upload_init.side_effect = ArroServerError("connection refused")
    mock_open, _ = _zarr_mock()

    with patch("arro_nlp_frontend.ingest.zarr.open_array", mock_open):
        r = _post(client, [{"doc_id": "doc0", "text": "hello"}])

    assert r.status_code == 502
    assert r.headers.get("X-Partial-Write") == "rolled-back"
    assert store.count(DEFAULT_DS) == 0
    assert store.next_row_index(DEFAULT_DS) == 0


def test_full_ingest_pre_commit_failure_batch_of_three_rolls_back_all(
    ingest_client,
) -> None:
    """Issue #25: rollback removes ALL N rows, not just the first."""
    client, store, mock_arro = ingest_client
    mock_arro.upload_commit.side_effect = ArroServerError("timeout")
    mock_open, _ = _zarr_mock()

    with patch("arro_nlp_frontend.ingest.zarr.open_array", mock_open):
        r = _post(
            client,
            [
                {"doc_id": "d0", "text": "a"},
                {"doc_id": "d1", "text": "b"},
                {"doc_id": "d2", "text": "c"},
            ],
        )

    assert r.status_code == 502
    assert r.headers.get("X-Partial-Write") == "rolled-back"
    assert store.count(DEFAULT_DS) == 0
    assert store.next_row_index(DEFAULT_DS) == 0


def test_full_ingest_rollback_does_not_affect_previous_successful_rows(
    ingest_client,
) -> None:
    """Issue #25: rollback of batch N does not touch rows from batch N-1."""
    client, store, mock_arro = ingest_client
    mock_open, _ = _zarr_mock()

    # First batch succeeds
    with patch("arro_nlp_frontend.ingest.zarr.open_array", mock_open):
        r1 = _post(client, [{"doc_id": "doc0", "text": "first"}])
    assert r1.status_code == 200
    assert store.count(DEFAULT_DS) == 1

    # Second batch fails pre-commit
    mock_arro.upload_commit.side_effect = ArroServerError("timeout")
    with patch("arro_nlp_frontend.ingest.zarr.open_array", mock_open):
        r2 = _post(client, [{"doc_id": "doc1", "text": "second"}])
    assert r2.status_code == 502
    assert r2.headers.get("X-Partial-Write") == "rolled-back"

    # doc0 from first batch must be intact
    assert store.count(DEFAULT_DS) == 1
    assert store.get_by_id(DEFAULT_DS, "doc0") is not None
    assert store.get_by_id(DEFAULT_DS, "doc1") is None
    assert store.next_row_index(DEFAULT_DS) == 1  # next will get row 1


def test_full_ingest_502_response_body_contains_recovery_hint(ingest_client) -> None:
    """Issue #25 option 3: 502 detail must contain actionable recovery hint."""
    client, _, mock_arro = ingest_client
    mock_arro.upload_commit.side_effect = ArroServerError("network error")
    mock_open, _ = _zarr_mock()

    with patch("arro_nlp_frontend.ingest.zarr.open_array", mock_open):
        r = _post(client, [{"doc_id": "doc0", "text": "hello"}])

    assert r.status_code == 502
    detail = r.json()["detail"]
    assert "rolled back" in detail.lower() or "rollback" in detail.lower()
    assert "re-ingest" in detail.lower() or "safe" in detail.lower()


def test_full_ingest_rollback_is_idempotent_if_row_already_absent(
    ingest_client,
) -> None:
    """store.rollback_rows must not raise if row_index is already absent.

    This guards against double-call scenarios (e.g., retry logic or signal
    handlers calling rollback twice).
    """
    _, store, _ = ingest_client
    result = store.rollback_rows(DEFAULT_DS, [0, 1, 2])
    assert result == 0


@pytest.mark.asyncio
async def test_ingest_full_rewrite_409_on_existing_doc_id(ingest_client) -> None:
    client, store, _ = ingest_client
    mock_open, _ = _zarr_mock()
    with patch("arro_nlp_frontend.ingest.zarr.open_array", mock_open):
        r1 = _post(client, [{"doc_id": "doc-1", "text": "hello"}])
    assert r1.status_code == 200

    mock_open2, _ = _zarr_mock()
    with patch("arro_nlp_frontend.ingest.zarr.open_array", mock_open2):
        r2 = _post(client, [{"doc_id": "doc-1", "text": "hello again"}])
    assert r2.status_code == 409
    detail = r2.json()["detail"]
    assert "doc-1" in detail
    assert "incremental=True" in detail
    doc = store.get_by_id(DEFAULT_DS, "doc-1")
    assert doc is not None
    assert doc.row_index == 0


@pytest.mark.asyncio
async def test_ingest_incremental_not_blocked_by_409_guard(ingest_client) -> None:
    client, store, mock_arro = ingest_client
    mock_open, _ = _zarr_mock()
    with patch("arro_nlp_frontend.ingest.zarr.open_array", mock_open):
        r1 = _post(client, [{"doc_id": "doc-2", "text": "original"}])
    assert r1.status_code == 200

    mock_arro.get_vector_count = AsyncMock(return_value=1)

    mock_open2, _ = _zarr_mock()
    with patch("arro_nlp_frontend.ingest.zarr.open_array", mock_open2):
        r2 = client.post(
            "/ingest",
            json={
                "dataset_id": DEFAULT_DS,
                "documents": [{"doc_id": "doc-2", "text": "updated"}],
                "incremental": True,
            },
        )
    assert r2.status_code == 200
    assert r2.json()["results"][0]["status"] == "updated"


@pytest.mark.asyncio
async def test_ingest_full_rewrite_first_ingest_not_blocked(ingest_client) -> None:
    client, _, _ = ingest_client
    mock_open, _ = _zarr_mock()
    with patch("arro_nlp_frontend.ingest.zarr.open_array", mock_open):
        r = _post(client, [{"doc_id": "new-doc", "text": "hello"}])
    assert r.status_code == 200
