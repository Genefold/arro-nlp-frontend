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
  12. test_search_dataset_isolation
  13. test_search_missing_dataset_id_422
  14. test_search_dataset_id_forwarded_to_arro_client
  Issue #126 -- embedding offloaded from the event loop:
  15. test_search_embedder_called_once_with_query
  16. test_search_invalid_query_never_reaches_embedder
  17. test_embed_query_runs_off_event_loop_thread
  18. test_health_responsive_while_embedding_blocks
  19. test_search_embedder_error_propagates
  20. test_embed_query_cancellation_propagates
"""

from __future__ import annotations

import asyncio
import threading
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock, patch

import httpx
import numpy as np
import pytest
from fastapi import FastAPI

from arro_nlp_frontend.arro_client import ArroClient, ArroServerError, SearchHit
from arro_nlp_frontend.embedder import Embedder
from arro_nlp_frontend.main import create_app
from arro_nlp_frontend.search import embed_query
from arro_nlp_frontend.store import Document

DEFAULT_DS = "test/dataset"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _post(
    client, query: str, dataset_id: str = DEFAULT_DS, top_k: int = 10, tau: float | None = None
):
    body: dict = {"dataset_id": dataset_id, "query": query, "top_k": top_k}
    if tau is not None:
        body["tau"] = tau
    return client.post("/search", json=body)


def _seed_store(store, docs: list[tuple[int, str, str]], dataset_id: str = DEFAULT_DS) -> None:
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
        store.upsert_batch(dataset_id, row, [doc], np.zeros((1, 384), dtype=np.float64))


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
            SearchHit(index=0, score=0.9),  # found
            SearchHit(index=1, score=0.7),  # ghost -- row 1 not in store
            SearchHit(index=2, score=0.5),  # found
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


# ---------------------------------------------------------------------------
# Multi-dataset tests
# ---------------------------------------------------------------------------


def test_search_dataset_isolation(search_client):
    """Search within one dataset does not return results from another."""
    client, store, mock_arro = search_client
    _seed_store(store, [(0, "doc-a", "buffer overflow")], dataset_id="ds/a")
    _seed_store(store, [(0, "doc-b", "sql injection")], dataset_id="ds/b")

    mock_arro.search = AsyncMock(return_value=[SearchHit(index=0, score=0.9)])

    r_a = _post(client, "overflow", dataset_id="ds/a")
    assert r_a.status_code == 200
    assert r_a.json()["results"][0]["doc_id"] == "doc-a"

    r_b = _post(client, "injection", dataset_id="ds/b")
    assert r_b.status_code == 200
    assert r_b.json()["results"][0]["doc_id"] == "doc-b"


def test_search_missing_dataset_id_422(search_client):
    """Request without dataset_id returns 422."""
    client, _, _ = search_client
    r = client.post("/search", json={"query": "test", "top_k": 10})
    assert r.status_code == 422


def test_search_dataset_id_forwarded_to_arro_client(search_client):
    """dataset_id from request is forwarded to arro_client.search."""
    client, _, mock_arro = search_client
    mock_arro.search = AsyncMock(return_value=[])

    _post(client, "query", dataset_id="nvd/embeddings")

    mock_arro.search.assert_called_once()
    call_args = mock_arro.search.call_args
    assert "dataset_id" in call_args.kwargs
    assert call_args.kwargs["dataset_id"] == "nvd/embeddings"


# ---------------------------------------------------------------------------
# Issue #126 -- embedding inference offloaded from the event loop
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _noop_lifespan(app: FastAPI):  # pragma: no cover -- trivial context manager
    yield


def _fake_embedder(encode_impl) -> Mock:
    """Mock(spec=Embedder) whose encode_batch is driven by `encode_impl`.

    `encode_impl` may be a plain sync callable or an exception instance
    (Mock raises it when called). The mock must stay SYNCHRONOUS: the
    endpoint offloads it via asyncio.to_thread, so an AsyncMock would
    bypass the code under test.
    """
    fake = Mock(spec=Embedder)
    fake.encode_batch = Mock(side_effect=encode_impl)
    fake.dim = 384
    return fake


def _build_app(fake_embedder: Mock) -> FastAPI:
    """Standalone app for async (event-loop level) tests, lifespan patched out."""
    with patch("arro_nlp_frontend.main.lifespan", _noop_lifespan):
        app = create_app()
    app.state.embedder = fake_embedder
    app.state.store = Mock()
    app.state.arro_client = AsyncMock(spec=ArroClient)
    app.state.arro_client.search = AsyncMock(return_value=[])
    app.state.ingest_locks = {}
    return app


async def _wait_for_thread_event(evt: threading.Event, timeout: float = 2.0) -> None:
    """Await a threading.Event from the event loop with a generous CI-safe timeout."""

    async def _poll() -> None:
        while not evt.is_set():
            await asyncio.sleep(0.005)

    await asyncio.wait_for(_poll(), timeout=timeout)


def test_search_embedder_called_once_with_query(search_client):
    """One valid query produces exactly one sync encode_batch([query]) call (#126).

    The embedder mock stays synchronous: inference is offloaded by the endpoint
    via asyncio.to_thread, so the correct assertion is a plain call-count
    check, not an await check.
    """
    client, store, mock_arro = search_client
    fake = _fake_embedder(lambda queries: np.ones((len(queries), 384)))
    client.app.state.embedder = fake
    _seed_store(store, [(0, "doc0", "Buffer overflow in OpenSSL")])
    mock_arro.search = AsyncMock(return_value=[SearchHit(index=0, score=0.9)])

    r = _post(client, "buffer overflow")

    assert r.status_code == 200
    fake.encode_batch.assert_called_once_with(["buffer overflow"])


def test_search_invalid_query_never_reaches_embedder(search_client):
    """400 validation fires before the embedder is touched (#126)."""
    client, _, _ = search_client
    fake = _fake_embedder(lambda queries: np.ones((len(queries), 384)))
    client.app.state.embedder = fake

    assert _post(client, "").status_code == 400
    assert _post(client, "   ").status_code == 400

    fake.encode_batch.assert_not_called()


async def test_embed_query_runs_off_event_loop_thread():
    """encode_batch must execute outside the event loop thread (#126).

    Direct regression guard against someone removing the to_thread offload:
    the fake records the thread it ran on and the test asserts it is not the
    caller's thread.
    """
    caller_thread = threading.get_ident()
    encode_threads: list[int] = []

    def encode_impl(queries: list[str]) -> np.ndarray:
        encode_threads.append(threading.get_ident())
        return np.ones((len(queries), 384))

    fake = _fake_embedder(encode_impl)

    vector = await embed_query(fake, "query")

    assert len(encode_threads) == 1
    assert encode_threads[0] != caller_thread
    assert vector.shape == (384,)


def _blocking_encode(started: threading.Event, release: threading.Event, deadline_s: float = 6.0):
    """Sync encoder that spins in its calling thread until released or deadline.

    The deadline bounds the spin: on the pre-#126 code the encoder runs on the
    event loop and would otherwise block it forever (wait_for timers can never
    fire inside a synchronous spin), hanging the test suite instead of failing
    it. With the deadline the regression surfaces as a TimeoutError failure.
    """

    def _encode(queries: list[str]) -> np.ndarray:
        started.set()
        t_end = time.monotonic() + deadline_s
        while not release.is_set() and time.monotonic() < t_end:
            time.sleep(0.001)
        return np.ones((len(queries), 384))

    return _encode


async def test_health_responsive_while_embedding_blocks():
    """A blocked inference must not stall the event loop or /health (#126).

    The search endpoint runs with an encoder that spins in a worker thread
    until released. While it is blocked, /health must still answer -- this
    is exactly the failure mode the offload fixes (the old synchronous call
    starved the loop and made the Docker healthcheck time out).
    Timing is coordinated with threading events and wait_for guards, not
    wall-clock sleeps, so it is stable on slow CI.
    """
    started = threading.Event()
    release = threading.Event()
    app = _build_app(_fake_embedder(_blocking_encode(started, release)))

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        search_task = asyncio.create_task(
            http.post("/search", json={"dataset_id": "d/s", "query": "q", "top_k": 1})
        )
        try:
            await _wait_for_thread_event(started)

            health_resp = await asyncio.wait_for(http.get("/health"), timeout=2.0)
            assert health_resp.status_code == 200
        finally:
            # Always unblock the worker: on the pre-#126 code the health check
            # times out and fails the test, but the encoder thread and the
            # in-flight request must still be released so the client can close.
            release.set()

        search_resp = await asyncio.wait_for(search_task, timeout=5.0)
        assert search_resp.status_code == 200


def test_search_embedder_error_propagates(search_client):
    """An embedder failure surfaces unchanged -- never an empty success (#126).

    The endpoint deliberately does not catch inference errors: they propagate
    to the server's generic 500 handling. With raise_server_exceptions=True
    the RuntimeError reaches the test, proving it is not swallowed and not
    converted into a 200 with empty results.
    """
    client, _, mock_arro = search_client
    fake = _fake_embedder(RuntimeError("model exploded"))
    client.app.state.embedder = fake

    with pytest.raises(RuntimeError, match="model exploded"):
        _post(client, "buffer overflow")

    mock_arro.search.assert_not_called()


async def test_embed_query_cancellation_propagates():
    """Cancelling the request cancels the coroutine-side wait (#126).

    A sync function already entered in a worker thread cannot be interrupted
    (Python cannot kill threads safely) -- the guarantee here is that the
    awaiting coroutine observes the cancellation and stops processing the
    response. The worker is always released in finally so the default
    executor is not left with a spinning thread for other tests.
    """
    started = threading.Event()
    release = threading.Event()
    fake = _fake_embedder(_blocking_encode(started, release))

    task = asyncio.create_task(embed_query(fake, "query"))
    await _wait_for_thread_event(started)
    task.cancel()
    try:
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        release.set()


# ---------------------------------------------------------------------------
# Issue #131 -- single-request cosine + ArrowSpace comparison search
# ---------------------------------------------------------------------------


def _compare_post(client, comparison_tau: float, top_k: int = 3) -> httpx.Response:
    return client.post(
        "/search",
        json={
            "dataset_id": DEFAULT_DS,
            "query": "buffer overflow",
            "top_k": top_k,
            "compare": True,
            "comparison_tau": comparison_tau,
        },
    )


def _seed_compare_store(store, docs: list[tuple[int, str, str]]) -> None:
    _seed_store(store, docs)


def _install_two_search_mock(mock_arro, cosine_hits, variant_hits, tau_order):
    """Route mock_arro.search by tau: returns hits per configured tau order."""
    calls: list[dict] = []

    async def _search(**kwargs):
        calls.append(kwargs)
        tau = kwargs["tau"]
        expected_tau = tau_order[len(calls) - 1]
        assert tau == pytest.approx(expected_tau)
        return cosine_hits if tau == pytest.approx(1.0) else variant_hits

    mock_arro.search = AsyncMock(side_effect=_search)
    return calls


def test_compare_spectral_one_embedding_two_searches(search_client):
    """One embedding, two searches (tau 1.0 then 0.42), same vector (#131)."""
    client, store, mock_arro = search_client
    _seed_compare_store(store, [(0, "CVE-1", "a"), (1, "CVE-2", "b"), (2, "CVE-3", "c")])
    fake = _fake_embedder(lambda queries: np.ones((len(queries), 384)))
    client.app.state.embedder = fake

    calls = _install_two_search_mock(
        mock_arro,
        [SearchHit(index=0, score=0.9), SearchHit(index=1, score=0.8)],
        [SearchHit(index=1, score=0.85), SearchHit(index=2, score=0.7)],
        tau_order=[1.0, 0.42],
    )

    r = _compare_post(client, 0.42)

    assert r.status_code == 200
    fake.encode_batch.assert_called_once_with(["buffer overflow"])
    assert mock_arro.search.call_count == 2
    assert calls[0]["tau"] == pytest.approx(1.0)
    assert calls[1]["tau"] == pytest.approx(0.42)
    assert calls[0]["vector"] is calls[1]["vector"]
    assert all(c["dataset_id"] == DEFAULT_DS for c in calls)


def test_compare_hybrid_tau_070(search_client):
    client, store, mock_arro = search_client
    _seed_compare_store(store, [(0, "CVE-1", "a"), (1, "CVE-2", "b")])

    calls = _install_two_search_mock(
        mock_arro,
        [SearchHit(index=0, score=0.9)],
        [SearchHit(index=1, score=0.8)],
        tau_order=[1.0, 0.70],
    )

    r = _compare_post(client, 0.70)

    assert r.status_code == 200
    assert calls[0]["tau"] == pytest.approx(1.0)
    assert calls[1]["tau"] == pytest.approx(0.70)


def test_compare_metadata_batch_hydration(search_client):
    """The union of hit rows is fetched with ONE store.get_by_rows query (#131)."""
    client, store, mock_arro = search_client
    _seed_compare_store(store, [(0, "CVE-A", "a"), (1, "CVE-B", "b"), (2, "CVE-C", "c")])

    get_by_rows_calls: list[list[int]] = []
    real_get_by_rows = store.get_by_rows

    def _counting_get_by_rows(dataset_id, row_indices):
        get_by_rows_calls.append(list(row_indices))
        return real_get_by_rows(dataset_id, row_indices)

    store.get_by_rows = _counting_get_by_rows
    # Compare mode must not fall back to per-row point lookups.
    store.get_by_row = Mock(
        side_effect=AssertionError("compare mode must use get_by_rows, not get_by_row")
    )

    _install_two_search_mock(
        mock_arro,
        [SearchHit(index=0, score=0.9), SearchHit(index=1, score=0.8), SearchHit(index=2, score=0.7)],
        [SearchHit(index=2, score=0.85), SearchHit(index=0, score=0.75)],
        tau_order=[1.0, 0.42],
    )

    r = _compare_post(client, 0.42)

    assert r.status_code == 200
    # Exactly one batch lookup over the stable first-seen union [0, 1, 2, ...]
    assert len(get_by_rows_calls) == 1
    assert get_by_rows_calls[0] == [0, 1, 2]


def test_compare_rank_statuses_and_kpis(search_client):
    """up/down/new/unchanged statuses, deltas and overlap KPIs are correct."""
    client, store, mock_arro = search_client
    _seed_compare_store(
        store,
        [
            (0, "CVE-Z", "z"),
            (1, "CVE-X", "x"),
            (2, "CVE-Q", "q"),
            (3, "CVE-V", "v"),
            (4, "CVE-Y", "y"),
            (5, "CVE-W", "w"),
        ],
    )
    # Cosine:    Z(1) X(2) Q(3) V(4) Y(5)
    # Spectral:  Y(1) X(2) Z(3) W(4)
    _install_two_search_mock(
        mock_arro,
        [SearchHit(0, 0.9), SearchHit(1, 0.8), SearchHit(2, 0.7), SearchHit(3, 0.6), SearchHit(4, 0.5)],
        [SearchHit(4, 0.95), SearchHit(1, 0.85), SearchHit(0, 0.8), SearchHit(5, 0.75)],
        tau_order=[1.0, 0.42],
    )

    r = _compare_post(client, 0.42, top_k=5)

    assert r.status_code == 200
    body = r.json()
    assert body["baseline"]["mode"] == "cosine"
    assert body["variant"]["mode"] == "spectral"

    items = body["variant"]["results"]
    by_id = {item["doc_id"]: item for item in items}
    assert by_id["CVE-Y"]["rank_status"] == "up" and by_id["CVE-Y"]["rank_delta"] == 4
    assert by_id["CVE-X"]["rank_status"] == "unchanged" and by_id["CVE-X"]["rank_delta"] == 0
    assert by_id["CVE-Z"]["rank_status"] == "down" and by_id["CVE-Z"]["rank_delta"] == -2
    assert by_id["CVE-W"]["rank_status"] == "new" and by_id["CVE-W"]["baseline_rank"] is None

    cmp = body["comparison"]
    assert cmp["overlap_count"] == 3  # X, Y, Z
    assert cmp["overlap_ratio"] == pytest.approx(3 / 5)  # denominator = requested k
    assert cmp["promoted_count"] == 1
    assert cmp["demoted_count"] == 1
    assert cmp["unchanged_count"] == 1
    assert cmp["new_count"] == 1
    assert cmp["dropped_count"] == 2  # Q, V


def test_compare_dedupes_duplicate_doc_ids(search_client):
    """Duplicate doc_ids within a list keep the first occurrence, ranks stay 1..N.

    Uses a mock store: the real DocumentStore cannot hold the same doc_id at
    two row indices, and arro-server may still surface duplicate identities.
    """
    client, _, mock_arro = search_client
    from types import SimpleNamespace

    docs = {
        0: SimpleNamespace(doc_id="CVE-1", text="first", metadata={}),
        1: SimpleNamespace(doc_id="CVE-1", text="second", metadata={}),
        2: SimpleNamespace(doc_id="CVE-2", text="x", metadata={}),
    }
    store = Mock()
    store.get_by_rows = Mock(return_value=docs)
    client.app.state.store = store

    _install_two_search_mock(
        mock_arro,
        [SearchHit(index=0, score=0.9), SearchHit(index=1, score=0.85), SearchHit(index=2, score=0.8)],
        [SearchHit(index=0, score=0.9)],
        tau_order=[1.0, 0.42],
    )

    r = _compare_post(client, 0.42)

    assert r.status_code == 200
    store.get_by_rows.assert_called_once()
    assert store.get_by_rows.call_args.args[0] == DEFAULT_DS
    assert store.get_by_rows.call_args.args[1] == [0, 1, 2]  # stable union, no repeats
    baseline = r.json()["baseline"]["results"]
    assert [res["rank"] for res in baseline] == [1, 2]
    assert [res["row_index"] for res in baseline] == [0, 2]  # first occurrence kept


def test_compare_variant_failure_is_atomic(search_client):
    """Second search fails -> whole comparison fails 502, no partial payload."""
    client, store, mock_arro = search_client
    _seed_compare_store(store, [(0, "CVE-1", "a")])

    async def _search(**kwargs):
        if kwargs["tau"] == pytest.approx(1.0):
            return [SearchHit(index=0, score=0.9)]
        raise ArroServerError("variant search failed")

    mock_arro.search = AsyncMock(side_effect=_search)

    r = _compare_post(client, 0.42)

    assert r.status_code == 502
    assert mock_arro.search.call_count == 2


def test_compare_ghost_rows_do_not_mutate_ranks(search_client):
    """Ghost rows are skipped; ranks stay contiguous and deltas reference real ids."""
    client, store, mock_arro = search_client
    _seed_compare_store(store, [(0, "CVE-1", "a"), (2, "CVE-2", "b")])

    _install_two_search_mock(
        mock_arro,
        [SearchHit(index=0, score=0.9), SearchHit(index=1, score=0.8), SearchHit(index=2, score=0.7)],
        [SearchHit(index=2, score=0.9), SearchHit(index=3, score=0.8)],
        tau_order=[1.0, 0.42],
    )

    r = _compare_post(client, 0.42)

    assert r.status_code == 200
    body = r.json()
    # cosine list: ghost row 1 skipped -> ranks 1..2
    assert [(res["doc_id"], res["rank"]) for res in body["baseline"]["results"]] == [
        ("CVE-1", 1),
        ("CVE-2", 2),
    ]
    # variant: row 2 present, ghost row 3 skipped -> up by one position
    variant = body["variant"]["results"]
    assert len(variant) == 1
    assert variant[0]["rank_status"] == "up"
    assert variant[0]["baseline_rank"] == 2  # cosine rank of CVE-2


def test_compare_missing_comparison_tau_422(search_client):
    client, _, _ = search_client
    r = client.post(
        "/search",
        json={"dataset_id": DEFAULT_DS, "query": "q", "compare": True},
    )
    assert r.status_code == 422


def test_compare_non_cosine_tau_422(search_client):
    client, _, _ = search_client
    r = client.post(
        "/search",
        json={
            "dataset_id": DEFAULT_DS,
            "query": "q",
            "tau": 0.42,
            "compare": True,
            "comparison_tau": 0.42,
        },
    )
    assert r.status_code == 422


def test_compare_unsupported_comparison_tau_422(search_client):
    client, _, _ = search_client
    r = client.post(
        "/search",
        json={"dataset_id": DEFAULT_DS, "query": "q", "compare": True, "comparison_tau": 0.5},
    )
    assert r.status_code == 422


def test_compare_top_k_above_cap_422(search_client):
    client, _, _ = search_client
    r = client.post(
        "/search",
        json={
            "dataset_id": DEFAULT_DS,
            "query": "q",
            "top_k": 21,
            "compare": True,
            "comparison_tau": 0.42,
        },
    )
    assert r.status_code == 422


def test_comparison_tau_without_compare_422(search_client):
    client, _, _ = search_client
    r = client.post(
        "/search",
        json={"dataset_id": DEFAULT_DS, "query": "q", "comparison_tau": 0.42},
    )
    assert r.status_code == 422


def test_compare_absent_keeps_single_mode_shape(search_client):
    """compare omitted -> existing flat {results, query_time_ms} response."""
    client, store, mock_arro = search_client
    _seed_store(store, [(0, "CVE-1", "a")])
    mock_arro.search = AsyncMock(return_value=[SearchHit(index=0, score=0.9)])

    r = _post(client, "q")

    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) == {"results", "query_time_ms"}
    assert body["results"][0]["doc_id"] == "CVE-1"


# Revised #131/#132 spec -- canonical search_mode contract
# ---------------------------------------------------------------------------


def _mode_post(client, search_mode: str, top_k: int = 3) -> httpx.Response:
    return client.post(
        "/search",
        json={
            "dataset_id": DEFAULT_DS,
            "query": "buffer overflow",
            "top_k": top_k,
            "search_mode": search_mode,
        },
    )


def test_search_mode_spectral_maps_to_tau_042(search_client):
    client, store, mock_arro = search_client
    _seed_compare_store(store, [(0, "CVE-1", "a"), (1, "CVE-2", "b")])
    fake = _fake_embedder(lambda queries: np.ones((len(queries), 384)))
    client.app.state.embedder = fake

    calls = _install_two_search_mock(
        mock_arro,
        [SearchHit(index=0, score=0.9)],
        [SearchHit(index=1, score=0.8)],
        tau_order=[1.0, 0.42],
    )

    r = _mode_post(client, "spectral")

    assert r.status_code == 200
    fake.encode_batch.assert_called_once_with(["buffer overflow"])
    assert mock_arro.search.call_count == 2
    assert calls[0]["tau"] == pytest.approx(1.0)
    assert calls[1]["tau"] == pytest.approx(0.42)
    assert calls[0]["vector"] is calls[1]["vector"]
    body = r.json()
    assert body["variant"]["mode"] == "spectral"
    assert body["variant"]["tau"] == pytest.approx(0.42)
    assert body["baseline"]["mode"] == "cosine"
    assert body["baseline"]["tau"] == pytest.approx(1.0)


def test_search_mode_hybrid_maps_to_tau_070(search_client):
    client, store, mock_arro = search_client
    _seed_compare_store(store, [(0, "CVE-1", "a")])

    calls = _install_two_search_mock(
        mock_arro,
        [SearchHit(index=0, score=0.9)],
        [SearchHit(index=0, score=0.8)],
        tau_order=[1.0, 0.70],
    )

    r = _mode_post(client, "hybrid")

    assert r.status_code == 200
    assert calls[1]["tau"] == pytest.approx(0.70)
    assert r.json()["variant"]["mode"] == "hybrid"


def test_search_mode_invalid_rejected(search_client):
    client, _, _ = search_client
    r = client.post(
        "/search",
        json={"dataset_id": DEFAULT_DS, "query": "q", "search_mode": "cosine"},
    )
    assert r.status_code == 422


def test_search_mode_with_legacy_fields_rejected(search_client):
    client, _, _ = search_client
    r = client.post(
        "/search",
        json={
            "dataset_id": DEFAULT_DS,
            "query": "q",
            "search_mode": "spectral",
            "compare": True,
        },
    )
    assert r.status_code == 422


def test_search_mode_unknown_extra_field_rejected(search_client):
    """#132: lam/alpha stay out of the downstream contract."""
    client, _, _ = search_client
    r = client.post(
        "/search",
        json={"dataset_id": DEFAULT_DS, "query": "q", "lam": 0.7},
    )
    assert r.status_code == 422


def test_search_mode_tau_100_variant_rejected(search_client):
    """tau=1.0 is the baseline, never a user-selected variant."""
    client, _, _ = search_client
    r = client.post(
        "/search",
        json={
            "dataset_id": DEFAULT_DS,
            "query": "q",
            "search_mode": "spectral",
            "tau": 0.42,
        },
    )
    assert r.status_code == 422
