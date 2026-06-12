"""Tests for POST /search.

All tests run fully offline -- no arro-server, no HF Hub downloads.

Test inventory:
  1.  test_search_returns_hydrated_results
  2.  test_search_rank_is_sequential_from_one
  3.  test_search_skips_missing_row_silently
  4.  test_search_rank_resequenced_after_ghost_skips
  5.  test_search_empty_query_400
  6.  test_search_whitespace_only_query_400
  7.  test_search_arro_server_down_502
  8.  test_search_tau_override_forwarded
  9.  test_search_default_tau_from_settings
  10. test_search_empty_results_from_arro
  11. test_search_query_time_ms_present
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import numpy as np
import pytest

from arro_nlp_frontend.arro_client import ArroServerError, SearchHit
from arro_nlp_frontend.store import Document

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _post(client, query: str, top_k: int = 10, tau: float | None = None):
    body: dict = {"query": query, "top_k": top_k}
    if tau is not None:
        body["tau"] = tau
    return client.post("/search", json=body)


def _seed_store(store, docs: list[tuple[int, str, str]]) -> None:
    """Insert documents at their exact row_index.

    Each (row_index, doc_id, text) tuple is inserted independently so
    upsert_batch(start_row=row, ...) places the doc at exactly that row.
    Calling upsert_batch once with start_row=docs[0][0] would assign rows
    sequentially from that offset, which is wrong for non-contiguous indices.
    """
    for row, doc_id, text in docs:
        doc = Document(
            row_index=row,
            doc_id=doc_id,
            text=text,
            metadata={},
            ingested_at=datetime.now(UTC),
        )
        store.upsert_batch(row, [doc], np.zeros((1, 384), dtype=np.float64))


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_search_returns_hydrated_results(search_client):
    """arro-server returns 2 hits; both row_indices exist in store; response has 2 results."""
    client, store, mock_arro = search_client
    _seed_store(
        store,
        [
            (0, "CVE-2024-1", "Buffer overflow in OpenSSL"),
            (1, "CVE-2024-2", "Use-after-free in libpng"),
        ],
    )
    mock_arro.search = AsyncMock(
        return_value=[
            SearchHit(index=0, score=0.91),
            SearchHit(index=1, score=0.74),
        ],
    )

    r = _post(client, "buffer overflow")
    assert r.status_code == 200
    results = r.json()["results"]
    assert len(results) == 2
    assert results[0]["doc_id"] == "CVE-2024-1"
    assert results[0]["text"] == "Buffer overflow in OpenSSL"
    assert results[0]["score"] == pytest.approx(0.91)
    assert results[1]["doc_id"] == "CVE-2024-2"


def test_search_rank_is_sequential_from_one(search_client):
    """Rank starts at 1 and increments by 1 regardless of scores."""
    client, store, mock_arro = search_client
    _seed_store(store, [(5, "doc5", "text"), (6, "doc9", "text")])
    mock_arro.search = AsyncMock(
        return_value=[
            SearchHit(index=5, score=0.8),
            SearchHit(index=6, score=0.6),
        ],
    )

    r = _post(client, "query")
    assert r.status_code == 200
    ranks = [res["rank"] for res in r.json()["results"]]
    assert ranks == [1, 2]


def test_search_skips_missing_row_silently(search_client):
    """arro-server returns index 99 which is not in store; result is skipped, no 500."""
    client, store, mock_arro = search_client
    _seed_store(store, [(0, "doc0", "exists")])
    mock_arro.search = AsyncMock(
        return_value=[
            SearchHit(index=0, score=0.9),
            SearchHit(index=99, score=0.5),  # ghost
        ],
    )

    r = _post(client, "exists")
    assert r.status_code == 200
    results = r.json()["results"]
    assert len(results) == 1
    assert results[0]["row_index"] == 0
    assert results[0]["rank"] == 1


def test_search_rank_resequenced_after_ghost_skips(search_client):
    """Ranks are 1..N after ghost skips -- no gaps in the rank sequence."""
    client, store, mock_arro = search_client
    _seed_store(store, [(0, "doc0", "first"), (2, "doc2", "third")])
    mock_arro.search = AsyncMock(
        return_value=[
            SearchHit(index=0, score=0.9),   # found
            SearchHit(index=1, score=0.7),   # ghost -- row 1 not in store
            SearchHit(index=2, score=0.5),   # found
        ],
    )

    r = _post(client, "query")
    assert r.status_code == 200
    results = r.json()["results"]
    assert len(results) == 2
    assert results[0]["rank"] == 1
    assert results[1]["rank"] == 2
    assert results[1]["row_index"] == 2


def test_search_empty_results_from_arro(search_client):
    """arro-server returns [] (dataset empty or no matches); response has empty results."""
    client, _, mock_arro = search_client
    mock_arro.search = AsyncMock(return_value=[])

    r = _post(client, "anything")
    assert r.status_code == 200
    assert r.json()["results"] == []


def test_search_query_time_ms_present(search_client):
    """query_time_ms is present and non-negative in every response."""
    client, _, _ = search_client
    r = _post(client, "ssl vulnerability")
    assert r.status_code == 200
    assert r.json()["query_time_ms"] >= 0


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_search_empty_query_400(search_client):
    """Empty string query returns 400."""
    client, _, _ = search_client
    r = _post(client, "")
    assert r.status_code == 400


def test_search_whitespace_only_query_400(search_client):
    """Whitespace-only query returns 400."""
    client, _, _ = search_client
    r = _post(client, "   ")
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_search_arro_server_down_502(search_client):
    """ArroServerError from search() propagates as 502."""
    client, _, mock_arro = search_client
    mock_arro.search = AsyncMock(side_effect=ArroServerError("connection refused"))

    r = _post(client, "ssl vulnerability")
    assert r.status_code == 502


# ---------------------------------------------------------------------------
# tau forwarding
# ---------------------------------------------------------------------------


def test_search_tau_override_forwarded(search_client):
    """Explicit tau in request is forwarded to arro_client.search."""
    client, _, mock_arro = search_client
    mock_arro.search = AsyncMock(return_value=[])

    _post(client, "query", tau=0.70)

    call_kwargs = mock_arro.search.call_args.kwargs
    assert call_kwargs["tau"] == pytest.approx(0.70)


def test_search_default_tau_from_settings(search_client):
    """When tau is absent from request, settings.arro_server_search_tau is used."""
    from arro_nlp_frontend.config import settings

    client, _, mock_arro = search_client
    mock_arro.search = AsyncMock(return_value=[])

    _post(client, "query")  # no tau in body

    call_kwargs = mock_arro.search.call_args.kwargs
    assert call_kwargs["tau"] == pytest.approx(settings.arro_server_search_tau)
