"""Tests for the POST /ingest endpoint. ALL tests run offline (no arro-server).

Tests
-----
1.  Single document ingest returns 200
2.  Batch yields contiguous row indices
3.  Status reflects "created" vs. "updated"
4.  Documents are persisted in the store
5.  Vectors are pushed to arro-server
6.  Duplicate doc_ids in request yield 422
7.  Empty batch yields 422
8.  arro-server failure yields 502, no store write
9.  arro-server failure leaves store unchanged
10. Second batch starts at correct row_index
11. start_row uses MAX(row_index)+1 (not COUNT) so soft-deletes do not reuse rows
12. Concurrent batches do not overlap row indices
13. duration_ms is present
14. Metadata is round-tripped
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import httpx
import pytest

from arro_nlp_frontend.arro_client import ArroClient, ArroServerError
from arro_nlp_frontend.embedder import Embedder
from arro_nlp_frontend.main import create_app
from arro_nlp_frontend.store import DocumentStore

from unittest.mock import AsyncMock


def _post(client, docs):
    return client.post("/ingest", json={"documents": docs})


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
    r = _post(client, [
        {"doc_id": "doc0", "text": "text0"},
        {"doc_id": "doc1", "text": "text1"},
        {"doc_id": "doc2", "text": "text2"},
    ])
    assert r.status_code == 200
    indices = [res["row_index"] for res in r.json()["results"]]
    assert sorted(indices) == [0, 1, 2]


def test_ingest_status_created_vs_updated(ingest_client):
    client, _, _ = ingest_client
    r1 = _post(client, [{"doc_id": "doc0", "text": "hello"}])
    assert r1.json()["results"][0]["status"] == "created"
    r2 = _post(client, [{"doc_id": "doc0", "text": "different"}])
    assert r2.json()["results"][0]["status"] == "updated"


def test_ingest_persists_in_store(ingest_client):
    client, store, _ = ingest_client
    _post(client, [{"doc_id": "doc0", "text": "hello"}])
    doc = store.get_by_id("doc0")
    assert doc is not None
    assert doc.doc_id == "doc0"
    assert doc.text == "hello"


def test_ingest_vectors_pushed_to_arro(ingest_client):
    client, _, mock_arro_client = ingest_client
    _post(client, [{"doc_id": "doc0", "text": "hello"}])
    mock_arro_client.push_vectors.assert_called_once()
    call_args = mock_arro_client.push_vectors.call_args
    # Check if start_row is 0 (either positional or keyword argument)
    if len(call_args[0]) > 1:
        assert call_args[0][1] == 0  # positional
    else:
        assert call_args[1]["start_row"] == 0  # keyword


def test_ingest_duplicate_doc_ids_in_request_422(ingest_client):
    client, _, _ = ingest_client
    r = _post(client, [
        {"doc_id": "dup", "text": "hello"},
        {"doc_id": "dup", "text": "hello"},
    ])
    assert r.status_code == 422


def test_ingest_empty_list_422(ingest_client):
    client, _, _ = ingest_client
    r = client.post("/ingest", json={"documents": []})
    assert r.status_code == 422


def test_ingest_arro_server_502_no_store_write(ingest_client):
    client, store, mock_arro_client = ingest_client
    mock_arro_client.push_vectors.side_effect = ArroServerError("mocked failure")
    r = _post(client, [{"doc_id": "doc0", "text": "hello"}])
    assert r.status_code == 502
    assert store.count() == 0


def test_ingest_arro_server_502_store_unchanged(ingest_client):
    client, store, mock_arro_client = ingest_client
    _post(client, [{"doc_id": "doc0", "text": "hello"}])
    mock_arro_client.push_vectors.reset_mock(side_effect=True)
    mock_arro_client.push_vectors.side_effect = ArroServerError("mocked failure")
    r = _post(client, [{"doc_id": "doc1", "text": "hello"}])
    assert r.status_code == 502
    assert store.count() == 1
    assert store.get_by_id("doc1") is None


def test_ingest_second_batch_start_row_correct(ingest_client):
    client, _, _ = ingest_client
    _post(client, [
        {"doc_id": "doc0", "text": "text0"},
        {"doc_id": "doc1", "text": "text1"},
    ])
    r = _post(client, [
        {"doc_id": "doc2", "text": "text2"},
        {"doc_id": "doc3", "text": "text3"},
    ])
    assert r.status_code == 200
    results = r.json()["results"]
    assert results[0]["row_index"] == 2
    assert results[1]["row_index"] == 3


def test_ingest_start_row_uses_max_not_count(ingest_client):
    client, store, _ = ingest_client
    _post(client, [
        {"doc_id": "doc0", "text": "text0"},
        {"doc_id": "doc1", "text": "text1"},
        {"doc_id": "doc2", "text": "text2"},
    ])
    store.delete_by_id("doc1")
    r = _post(client, [{"doc_id": "doc3", "text": "text3"}])
    assert r.status_code == 200
    assert r.json()["results"][0]["row_index"] == 3


def test_ingest_concurrent_batches_no_row_overlap(tmp_path: Path) -> None:
    """Concurrent ingest requests never assign overlapping row indices.

    Uses httpx.AsyncClient directly (not TestClient) to allow real async concurrency.
    """
    app = create_app()
    app.state.embedder = Embedder(backend="local", model="all-MiniLM-L6-v2", scale_factor=1.0)
    app.state.store = DocumentStore(tmp_path / "concurrent_test.sqlite")
    app.state.arro_client = AsyncMock(spec=ArroClient)
    app.state.arro_client.push_vectors = AsyncMock(return_value=None)
    app.state.arro_client.row_count = AsyncMock(return_value=0)
    app.state.ingest_lock = asyncio.Lock()

    transport = httpx.ASGITransport(app=app)

    async def _post_ingest(doc_id: str) -> list[int]:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            r = await c.post(
                "/ingest",
                json={"documents": [{"doc_id": doc_id, "text": "text"}]},
            )
            return [x["row_index"] for x in r.json()["results"]]

    async def _run():
        results = await asyncio.gather(
            _post_ingest("a"),
            _post_ingest("b"),
            _post_ingest("c"),
        )
        all_indices = sorted(i for batch in results for i in batch)
        assert len(all_indices) == 3
        assert all_indices == [0, 1, 2]

    asyncio.run(_run())


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
    doc = store.get_by_id("docX")
    assert doc is not None
    assert doc.metadata == meta
