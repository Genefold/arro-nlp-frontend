"""POST /search endpoint: embed query, call arro-server, hydrate results from store.

Pipeline (single mode):
  1. Validate: non-empty query string                   -> 400 on failure
  2. Embed query in a worker thread (never the event loop)
     -> float64 vector shape (dim,)
  3. Call arro-server: POST /api/datasets/{id}/search   -> 502 on failure
  4. For each (index, score) in results:
       doc = store.get_by_row(index)
       if doc is None: log warning + skip (data inconsistency, do not 500)
  5. Return SearchResponse { results, query_time_ms }

Pipeline (compare mode, issue #131):
  1. Embed the query exactly once.
  2. Search the SAME resident dataset twice, sequentially:
     tau=1.0 (cosine baseline) and comparison_tau (0.42 spectral / 0.70 hybrid).
  3. Hydrate metadata once over the stable union of hit row indices.
  4. Compute per-variant rank deltas and overlap KPIs server-side.
  5. Return CompareSearchResponse — atomic: any failed branch fails the
     whole comparison, no partial payload.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import cast

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, SerializeAsAny, model_validator

from arro_nlp_frontend.arro_client import ArroServerError
from arro_nlp_frontend.config import settings
from arro_nlp_frontend.embedder import Embedder
from arro_nlp_frontend.store import Document

logger = logging.getLogger(__name__)

router = APIRouter()

# Search-mode tau constants (issue #131). Cosine is the product default;
# spectral and hybrid exist only as approved comparison variants.
COSINE_TAU = 1.0
SPECTRAL_TAU = 0.42
HYBRID_TAU = 0.70
COMPARE_VARIANT_TAUS = frozenset({SPECTRAL_TAU, HYBRID_TAU})
MAX_COMPARE_K = 20


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class SearchRequest(BaseModel):
    """Request body for POST /search."""

    dataset_id: str = Field(
        ...,
        min_length=1,
        description="arro-server dataset to search against, e.g. 'cve/embeddings'.",
    )
    query: str = Field(..., description="Text query to search for.")
    top_k: int = Field(10, ge=1, le=1000, description="Maximum results to return.")
    tau: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description=(
            "Spectral threshold. None uses settings.arro_server_search_tau. "
            "0.42 = spectral-aware, 0.70 = hybrid, 1.00 = pure cosine."
        ),
    )
    compare: bool = Field(
        False,
        description=(
            "When true, return a cosine baseline (tau=1.0) and one approved "
            "ArrowSpace comparison variant in a single response. The query "
            "is embedded once and both searches run against the same "
            "resident dataset."
        ),
    )
    comparison_tau: float | None = Field(
        None,
        description=(
            "Required only when compare=true. 0.42 selects spectral and 0.70 selects hybrid."
        ),
    )

    @model_validator(mode="after")
    def _validate_compare_mode(self) -> SearchRequest:
        if not self.compare:
            if self.comparison_tau is not None:
                raise ValueError("comparison_tau requires compare=true")
            return self

        effective_tau = COSINE_TAU if self.tau is None else self.tau
        if effective_tau != COSINE_TAU:
            raise ValueError("compare mode requires cosine baseline tau=1.0")
        if self.comparison_tau not in COMPARE_VARIANT_TAUS:
            raise ValueError("comparison_tau must be 0.42 (spectral) or 0.70 (hybrid)")
        if self.top_k > MAX_COMPARE_K:
            raise ValueError(f"compare mode supports top_k up to {MAX_COMPARE_K}")
        return self


class SearchResult(BaseModel):
    """A single hydrated search result."""

    rank: int
    score: float
    row_index: int
    doc_id: str
    text: str
    metadata: dict


class CompareSearchResult(SearchResult):
    baseline_rank: int | None = None
    rank_delta: int | None = None
    rank_status: str | None = None


class ModeResults(BaseModel):
    mode: str
    tau: float
    # SerializeAsAny keeps CompareSearchResult rank-delta fields in the output
    # while allowing plain SearchResult items in the baseline column.
    results: list[SerializeAsAny[SearchResult]]


class ComparisonSummary(BaseModel):
    baseline_mode: str
    variant_mode: str
    k: int
    overlap_count: int
    overlap_ratio: float
    promoted_count: int
    demoted_count: int
    unchanged_count: int
    new_count: int
    dropped_count: int
    baseline_result_count: int
    variant_result_count: int


class SearchResponse(BaseModel):
    """Response body for POST /search."""

    results: list[SearchResult]
    query_time_ms: int


class CompareSearchResponse(BaseModel):
    """Response body for POST /search with compare=true."""

    baseline: ModeResults
    variant: ModeResults
    comparison: ComparisonSummary
    query_time_ms: int


# ---------------------------------------------------------------------------
# Embedding offload
# ---------------------------------------------------------------------------


async def embed_query(embedder: Embedder, query: str) -> np.ndarray:
    """Embed a single query in a worker thread (issue #126).

    encode_batch is CPU-bound (sentence-transformers inference). Calling it
    directly in the request coroutine stalls the event loop: /health stops
    answering and concurrent searches queue behind the inference. asyncio.to_thread
    keeps the loop responsive without changing the vector, its dtype, or the
    error semantics (model failures propagate unchanged; the await stays
    cancellable).

    Thread-safety: the shared Embedder is inference-only after startup --
    backend, model, OpenAI client and scale_factor are set once in lifespan
    and never mutated, so concurrent encode_batch calls on the shared instance
    are safe. No Semaphore: uvicorn runs a single worker and the harness
    already rate-limits upstream; serialising here would add queueing without
    telemetry. Revisit only if concurrent inference shows memory pressure on
    the 2 GB droplet.
    """
    encoded = await asyncio.to_thread(embedder.encode_batch, [query])
    # numpy stubs type __getitem__ as Any -- make the ndarray boundary explicit
    return cast(np.ndarray, encoded[0])


# ---------------------------------------------------------------------------
# Hydration (shared by both modes)
# ---------------------------------------------------------------------------


def _hydrate_hits(
    hits,
    documents_by_row: dict[int, Document | None],
) -> list[SearchResult]:
    """Build ranked results from hits and a batch-fetched document map.

    De-duplicates by doc_id (first occurrence wins) so arro-server's
    ordering cannot surface the same identity twice; ranks are assigned
    AFTER de-duplication so 1..N stays contiguous. Missing rows are
    logged and skipped (data inconsistency, not a hard error); hydration
    never reorders hits.

    Single mode fetches via store.get_by_row per hit (unchanged legacy
    path); compare mode fetches the stable union of row indices with ONE
    store.get_by_rows query (#131).
    """
    hydrated: list[SearchResult] = []
    seen_doc_ids: set[str] = set()
    for hit in hits:
        doc = documents_by_row.get(hit.index)
        if doc is None:
            logger.warning(
                "[search] row_index=%d returned by arro-server not found in store "
                "(data inconsistency -- index may be stale). Skipping.",
                hit.index,
            )
            continue
        if doc.doc_id in seen_doc_ids:
            continue
        seen_doc_ids.add(doc.doc_id)
        hydrated.append(
            SearchResult(
                rank=len(hydrated) + 1,
                score=hit.score,
                row_index=hit.index,
                doc_id=doc.doc_id,
                text=doc.text,
                metadata=doc.metadata,
            )
        )
    return hydrated


def _build_compare_response(
    baseline_results: list[SearchResult],
    variant_results: list[SearchResult],
    variant_mode: str,
    variant_tau: float,
    requested_k: int,
) -> CompareSearchResponse:
    """Compute rank deltas and overlap KPIs over the two top-k lists.

    Delta semantics: rank_delta = baseline_rank - variant_rank.
    Positive delta means the item moved up in the variant list.
    overlap_ratio's denominator is the requested k, not either list length.
    """
    baseline_rank_by_id = {r.doc_id: r.rank for r in baseline_results}
    baseline_ids = set(baseline_rank_by_id)
    variant_ids = {r.doc_id for r in variant_results}

    variant_items: list[CompareSearchResult] = []
    for r in variant_results:
        base_rank = baseline_rank_by_id.get(r.doc_id)
        if base_rank is None:
            status, delta, base = "new", None, None
        else:
            delta = base_rank - r.rank
            status = "up" if delta > 0 else ("down" if delta < 0 else "unchanged")
            base = base_rank
        variant_items.append(
            CompareSearchResult(
                **r.model_dump(),
                baseline_rank=base,
                rank_delta=delta,
                rank_status=status,
            )
        )

    overlap_count = len(baseline_ids & variant_ids)
    summary = ComparisonSummary(
        baseline_mode="cosine",
        variant_mode=variant_mode,
        k=requested_k,
        overlap_count=overlap_count,
        overlap_ratio=overlap_count / requested_k if requested_k else 0.0,
        promoted_count=sum(1 for r in variant_items if r.rank_status == "up"),
        demoted_count=sum(1 for r in variant_items if r.rank_status == "down"),
        unchanged_count=sum(1 for r in variant_items if r.rank_status == "unchanged"),
        new_count=sum(1 for r in variant_items if r.rank_status == "new"),
        dropped_count=len(baseline_ids - variant_ids),
        baseline_result_count=len(baseline_results),
        variant_result_count=len(variant_results),
    )
    return CompareSearchResponse(
        baseline=ModeResults(mode="cosine", tau=COSINE_TAU, results=baseline_results),
        variant=ModeResults(
            mode=variant_mode,
            tau=variant_tau,
            # SerializeAsAny keeps the subclass rank-delta fields in the output;
            # list invariance requires the explicit cast.
            results=cast("list[SerializeAsAny[SearchResult]]", variant_items),
        ),
        comparison=summary,
        query_time_ms=0,
    )


@router.post(
    "/search",
    response_model=SearchResponse | CompareSearchResponse,
    tags=["search"],
)
async def search(
    request: SearchRequest,
    req: Request,
):
    """Embed query text, retrieve ranked results from arro-server, hydrate from store.

    Pipeline (no lock required -- this is a pure read path):
      1. Validate: non-empty query string
      2. Embed query in a worker thread -> float64 vector (dim,)  [exactly once]
      3. POST /api/datasets/{id}/search with vector, top_k, tau
         (compare=true: two sequential searches, tau=1.0 then comparison_tau,
          against the same resident dataset)
      4. Hydrate returned row_indices from DocumentStore
         Missing rows are logged and skipped (data inconsistency, not a hard error)
      5. Return ranked, hydrated results (compare=true: baseline + variant
         columns with rank deltas and overlap KPIs)

    Raises:
      400: query is empty or whitespace-only
      422: invalid compare parameters
      502: arro-server unreachable or returned non-2xx
    """
    t0 = time.perf_counter()

    # Step 1 -- validate query
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")

    embedder = req.app.state.embedder
    store = req.app.state.store
    arro_client = req.app.state.arro_client

    # Step 2 -- embed query off the event loop (issue #126) — exactly once,
    # reused by every search pass in compare mode (#131)
    t_embed = time.perf_counter()
    vector = await embed_query(embedder, request.query)
    embedding_ms = int((time.perf_counter() - t_embed) * 1000)

    # Step 3 -- call arro-server
    try:
        if not request.compare:
            # Resolve tau: per-request override, else settings default (cosine)
            tau = request.tau if request.tau is not None else settings.arro_server_search_tau
            hits = await arro_client.search(
                dataset_id=request.dataset_id,
                vector=vector,
                top_k=request.top_k,
                tau=tau,
            )
        else:
            variant_mode = "spectral" if request.comparison_tau == SPECTRAL_TAU else "hybrid"
            baseline_hits = await arro_client.search(
                dataset_id=request.dataset_id,
                vector=vector,
                top_k=request.top_k,
                tau=COSINE_TAU,
            )
            variant_hits = await arro_client.search(
                dataset_id=request.dataset_id,
                vector=vector,
                top_k=request.top_k,
                tau=request.comparison_tau,
            )
    except ArroServerError as exc:
        logger.error("[search] arro-server search failed: %s", exc)
        raise HTTPException(
            status_code=502,
            detail=f"arro-server error: {exc}",
        ) from exc

    # Step 4 -- hydrate from store
    if not request.compare:
        documents_by_row = {
            hit.index: store.get_by_row(dataset_id=request.dataset_id, row_index=hit.index)
            for hit in hits
        }
        results = _hydrate_hits(hits, documents_by_row)

        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        logger.info(
            "[search] dataset=%s query=%r top_k=%d tau=%.2f embedding_ms=%d hits=%d "
            "hydrated=%d duration_ms=%d",
            request.dataset_id,
            request.query[:60],
            request.top_k,
            tau,
            embedding_ms,
            len(hits),
            len(results),
            elapsed_ms,
        )
        return SearchResponse(results=results, query_time_ms=elapsed_ms)

    # Compare mode: ONE batch query over the stable union of row indices.
    row_indices = list(
        dict.fromkeys([hit.index for hit in baseline_hits] + [hit.index for hit in variant_hits])
    )
    documents_by_row = store.get_by_rows(request.dataset_id, row_indices)
    baseline_results = _hydrate_hits(baseline_hits, documents_by_row)
    variant_results = _hydrate_hits(variant_hits, documents_by_row)
    response = _build_compare_response(
        baseline_results,
        variant_results,
        variant_mode,
        cast(float, request.comparison_tau),
        request.top_k,
    )
    response.query_time_ms = int((time.perf_counter() - t0) * 1000)
    logger.info(
        "[search] dataset=%s query=%r top_k=%d compare=cosine|%.2f embedding_ms=%d "
        "baseline_hits=%d variant_hits=%d overlap=%d duration_ms=%d",
        request.dataset_id,
        request.query[:60],
        request.top_k,
        request.comparison_tau,
        embedding_ms,
        len(baseline_results),
        len(variant_results),
        response.comparison.overlap_count,
        response.query_time_ms,
    )
    return response
