"""Tests for POST /ingest with incremental=True.

ALL tests run offline -- no arro-server, no HF Hub downloads.
The arro_client is an AsyncMock; append_vectors/overwrite_vectors return
values are configured per-test.

Test index
----------
 1. test_incremental_new_docs_appended
        All docs are new -> append_vectors called once, correct row indices.
 2. test_incremental_changed_docs_overwritten
        Pre-existing docs with changed text -> overwrite_vectors called,
        row indices preserved, no append call.
 3. test_incremental_metadata_only_no_embed_no_vector_write
        Pre-existing docs, text unchanged -> neither append nor overwrite
        called.
 4. test_incremental_build_index_called_once_for_mixed_batch
        Mixed batch (new + changed + metadata-only) -> build_index called
        exactly once.
 5. test_incremental_build_index_not_called_for_metadata_only
        All metadata-only -> build_index NOT called.
 6. test_incremental_consistency_guard_raises_409_on_mismatch
        server nrows != local next_row_index -> 409, no append/overwrite.
 7. test_incremental_response_order_matches_request_order
        Response results are in the same order as the request documents,
        regardless of new/changed/metadata classification order.
 8. test_incremental_status_skipped_for_metadata_only
        metadata-only docs have status="skipped" in the response.
 9. test_incremental_text_fingerprint_stable_across_calls
        _text_fingerprint() returns the same value on repeated calls
        (no PYTHONHASHSEED dependency).
10. test_incremental_does_not_call_full_rewrite_methods
        upload_init and upload_commit are NOT called in incremental mode.
11. test_incremental_arro_server_error_returns_502
        append_vectors raises ArroServerError -> 502, SQLite left intact.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import numpy as np
import pytest

from arro_nlp_frontend.arro_client import (
    ArroServerError,
    VectorAppendResult,
    VectorOverwriteResult,
)
from arro_nlp_frontend.ingest import _text_fingerprint
from arro_nlp_frontend.store import Document

DEFAULT_DS = "test/dataset"


def _post_incremental(client, docs, dataset_id: str = DEFAULT_DS):
    """Helper: POST /ingest with incremental=True."""
    return client.post(
        "/ingest",
        json={
            "dataset_id": dataset_id,
            "documents": docs,
            "incremental": True,
        },
    )


def _seed_store(store, dataset_id: str, items: list[dict]) -> None:
    """Insert documents into the store directly, bypassing the HTTP layer.

    Each item must have: doc_id, text, row_index.
    Vectors are zero-filled (content irrelevant for incremental tests).
    """
    dim = 384
    docs = [
        Document(
            row_index=it["row_index"],
            doc_id=it["doc_id"],
            text=it["text"],
            metadata=it.get("metadata", {}),
            ingested_at=None,
        )
        for it in items
    ]
    vectors = np.zeros((len(docs), dim), dtype=np.float64)
    store.upsert_batch_with_indices(dataset_id, docs, vectors)


# ---------------------------------------------------------------------------
# 1. New documents
# ---------------------------------------------------------------------------


def test_incremental_new_docs_appended(ingest_client):
    """All docs are new -> append_vectors called once with correct vectors.

    Verifies:
    - append_vectors called once with shape (2, 384)
    - overwrite_vectors NOT called
    - row indices match VectorAppendResult.start_row + offset
    - status is "created" for both
    """
    client, store, mock_arro = ingest_client
    mock_arro.get_vector_count = AsyncMock(return_value=0)
    mock_arro.append_vectors = AsyncMock(
        return_value=VectorAppendResult(start_row=0, appended=2, new_shape=[2, 384])
    )

    r = _post_incremental(
        client,
        [
            {"doc_id": "doc0", "text": "text zero"},
            {"doc_id": "doc1", "text": "text one"},
        ],
    )
    assert r.status_code == 200, r.json()
    body = r.json()

    assert body["ingested"] == 2
    assert body["results"][0] == {"doc_id": "doc0", "row_index": 0, "status": "created"}
    assert body["results"][1] == {"doc_id": "doc1", "row_index": 1, "status": "created"}

    mock_arro.append_vectors.assert_called_once()
    call_args = mock_arro.append_vectors.call_args
    assert call_args.args[0] == DEFAULT_DS
    sent_vecs = call_args.args[1]
    assert sent_vecs.shape == (2, 384)

    mock_arro.overwrite_vectors.assert_not_called()


# ---------------------------------------------------------------------------
# 2. Changed documents
# ---------------------------------------------------------------------------


def test_incremental_changed_docs_overwritten(ingest_client):
    """Pre-existing docs with changed text -> overwrite called, row indices preserved.

    Verifies:
    - overwrite_vectors called once with 2 updates
    - each update carries the original row_index from the store
    - append_vectors NOT called (no new docs)
    - status is "updated"
    """
    client, store, mock_arro = ingest_client

    _seed_store(store, DEFAULT_DS, [
        {"doc_id": "doc0", "text": "old text zero", "row_index": 0},
        {"doc_id": "doc1", "text": "old text one",  "row_index": 1},
    ])
    mock_arro.get_vector_count = AsyncMock(return_value=2)
    mock_arro.overwrite_vectors = AsyncMock(
        return_value=VectorOverwriteResult(overwritten=2)
    )

    r = _post_incremental(
        client,
        [
            {"doc_id": "doc0", "text": "new text zero"},
            {"doc_id": "doc1", "text": "new text one"},
        ],
    )
    assert r.status_code == 200, r.json()
    body = r.json()

    assert body["results"][0] == {"doc_id": "doc0", "row_index": 0, "status": "updated"}
    assert body["results"][1] == {"doc_id": "doc1", "row_index": 1, "status": "updated"}

    mock_arro.append_vectors.assert_not_called()
    mock_arro.overwrite_vectors.assert_called_once()

    updates = mock_arro.overwrite_vectors.call_args.args[1]
    row_indices = [row_idx for row_idx, _ in updates]
    assert row_indices == [0, 1]


# ---------------------------------------------------------------------------
# 3. Metadata-only -- no embed, no vector write
# ---------------------------------------------------------------------------


def test_incremental_metadata_only_no_embed_no_vector_write(ingest_client):
    """Unchanged text -> no vector writes.

    Verifies:
    - append_vectors NOT called
    - overwrite_vectors NOT called
    - status is "skipped"
    - store metadata is updated (text unchanged but metadata may differ)
    """
    client, store, mock_arro = ingest_client

    _seed_store(store, DEFAULT_DS, [
        {"doc_id": "doc0", "text": "unchanged text", "row_index": 0},
    ])

    r = _post_incremental(
        client,
        [{"doc_id": "doc0", "text": "unchanged text", "metadata": {"updated": True}}],
    )
    assert r.status_code == 200, r.json()

    assert r.json()["results"][0]["status"] == "skipped"
    mock_arro.append_vectors.assert_not_called()
    mock_arro.overwrite_vectors.assert_not_called()

    # Metadata updated in store.
    doc = store.get_by_id(DEFAULT_DS, "doc0")
    assert doc is not None
    assert doc.metadata == {"updated": True}


# ---------------------------------------------------------------------------
# 4. build_index called once for mixed batch
# ---------------------------------------------------------------------------


def test_incremental_build_index_called_once_for_mixed_batch(ingest_client):
    """Mixed batch (new + changed + metadata-only) -> build_index called exactly once."""
    client, store, mock_arro = ingest_client

    _seed_store(store, DEFAULT_DS, [
        {"doc_id": "changed", "text": "old text",       "row_index": 0},
        {"doc_id": "meta",    "text": "unchanged text",  "row_index": 1},
    ])
    mock_arro.get_vector_count = AsyncMock(return_value=2)
    mock_arro.append_vectors = AsyncMock(
        return_value=VectorAppendResult(start_row=2, appended=1, new_shape=[3, 384])
    )
    mock_arro.overwrite_vectors = AsyncMock(
        return_value=VectorOverwriteResult(overwritten=1)
    )

    r = _post_incremental(
        client,
        [
            {"doc_id": "new",     "text": "brand new text"},
            {"doc_id": "changed", "text": "new text"},
            {"doc_id": "meta",    "text": "unchanged text"},
        ],
    )
    assert r.status_code == 200, r.json()
    mock_arro.build_index.assert_called_once_with(dataset_id=DEFAULT_DS)


# ---------------------------------------------------------------------------
# 5. build_index NOT called when all metadata-only
# ---------------------------------------------------------------------------


def test_incremental_build_index_not_called_for_metadata_only(ingest_client):
    """All metadata-only -> build_index NOT called (no vector changes)."""
    client, store, mock_arro = ingest_client

    _seed_store(store, DEFAULT_DS, [
        {"doc_id": "doc0", "text": "same text", "row_index": 0},
    ])

    r = _post_incremental(
        client,
        [{"doc_id": "doc0", "text": "same text"}],
    )
    assert r.status_code == 200, r.json()
    mock_arro.build_index.assert_not_called()


# ---------------------------------------------------------------------------
# 6. Consistency guard -> 409
# ---------------------------------------------------------------------------


def test_incremental_consistency_guard_raises_409_on_mismatch(ingest_client):
    """server nrows != local next_row_index -> 409 before any write."""
    client, store, mock_arro = ingest_client

    # Store has 0 rows; mock server says 5 -> mismatch.
    mock_arro.get_vector_count = AsyncMock(return_value=5)

    r = _post_incremental(
        client,
        [{"doc_id": "doc_new", "text": "some text"}],
    )
    assert r.status_code == 409, r.json()
    assert "consistency error" in r.json()["detail"].lower()

    # No writes should have happened.
    mock_arro.append_vectors.assert_not_called()
    mock_arro.overwrite_vectors.assert_not_called()
    assert store.count(DEFAULT_DS) == 0


# ---------------------------------------------------------------------------
# 7. Response order matches request order
# ---------------------------------------------------------------------------


def test_incremental_response_order_matches_request_order(ingest_client):
    """Response results are in request order regardless of classification."""
    client, store, mock_arro = ingest_client

    _seed_store(store, DEFAULT_DS, [
        {"doc_id": "b", "text": "old b", "row_index": 0},
        {"doc_id": "c", "text": "same c", "row_index": 1},
    ])
    mock_arro.get_vector_count = AsyncMock(return_value=2)
    mock_arro.append_vectors = AsyncMock(
        return_value=VectorAppendResult(start_row=2, appended=1, new_shape=[3, 384])
    )
    mock_arro.overwrite_vectors = AsyncMock(
        return_value=VectorOverwriteResult(overwritten=1)
    )

    # Request order: new(a), changed(b), metadata(c)
    r = _post_incremental(
        client,
        [
            {"doc_id": "a", "text": "new a"},
            {"doc_id": "b", "text": "new b"},
            {"doc_id": "c", "text": "same c"},
        ],
    )
    assert r.status_code == 200, r.json()
    results = r.json()["results"]
    assert results[0]["doc_id"] == "a"
    assert results[1]["doc_id"] == "b"
    assert results[2]["doc_id"] == "c"


# ---------------------------------------------------------------------------
# 8. status="skipped" for metadata-only
# ---------------------------------------------------------------------------


def test_incremental_status_skipped_for_metadata_only(ingest_client):
    """metadata-only docs return status='skipped'."""
    client, store, mock_arro = ingest_client

    _seed_store(store, DEFAULT_DS, [
        {"doc_id": "doc0", "text": "exact same text", "row_index": 0},
    ])

    r = _post_incremental(
        client,
        [{"doc_id": "doc0", "text": "exact same text"}],
    )
    assert r.status_code == 200, r.json()
    assert r.json()["results"][0]["status"] == "skipped"


# ---------------------------------------------------------------------------
# 9. _text_fingerprint stability
# ---------------------------------------------------------------------------


def test_incremental_text_fingerprint_stable_across_calls():
    """_text_fingerprint returns identical bytes on repeated calls (no hash randomisation)."""
    text = "The quick brown fox jumps over the lazy dog"
    fp1 = _text_fingerprint(text)
    fp2 = _text_fingerprint(text)
    assert fp1 == fp2
    assert isinstance(fp1, bytes)
    assert len(fp1) == 8

    # Different texts produce different fingerprints (basic collision check).
    assert _text_fingerprint("text A") != _text_fingerprint("text B")


# ---------------------------------------------------------------------------
# 10. Full-rewrite methods NOT called in incremental mode
# ---------------------------------------------------------------------------


def test_incremental_does_not_call_full_rewrite_methods(ingest_client):
    """upload_init and upload_commit are NOT called when incremental=True."""
    client, store, mock_arro = ingest_client
    mock_arro.get_vector_count = AsyncMock(return_value=0)
    mock_arro.append_vectors = AsyncMock(
        return_value=VectorAppendResult(start_row=0, appended=1, new_shape=[1, 384])
    )

    r = _post_incremental(
        client,
        [{"doc_id": "doc0", "text": "some text"}],
    )
    assert r.status_code == 200, r.json()
    mock_arro.upload_init.assert_not_called()
    mock_arro.upload_commit.assert_not_called()


# ---------------------------------------------------------------------------
# 11. arro-server error -> 502
# ---------------------------------------------------------------------------


def test_incremental_arro_server_error_returns_502(ingest_client):
    """append_vectors raises ArroServerError -> 502, store unaffected."""
    client, store, mock_arro = ingest_client
    mock_arro.get_vector_count = AsyncMock(return_value=0)
    mock_arro.append_vectors = AsyncMock(
        side_effect=ArroServerError("mocked append failure", status_code=500)
    )

    r = _post_incremental(
        client,
        [{"doc_id": "doc0", "text": "new text"}],
    )
    assert r.status_code == 502, r.json()
    assert "arro-server error" in r.json()["detail"].lower()
    # Store must not have been written (error occurred inside the lock,
    # before upsert_batch_with_indices was called).
    assert store.count(DEFAULT_DS) == 0
