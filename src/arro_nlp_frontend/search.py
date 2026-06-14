"""POST /search endpoint: embed query, call arro-server, hydrate results from store.

Pipeline:
  1. Validate: non-empty query string                   -> 400 on failure
  2. Embed query -> float64 vector shape (dim,)          <- synchronous, no lock needed
  3. Call arro-server: POST /api/datasets/{id}/search   -> 502 on failure
  4. For each (index, score) in results:
       doc = store.get_by_row(index)
       if doc is None: log warning + skip (data inconsistency, do not 500)
  5. Return SearchResponse { results, query_time_ms }
"""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from arro_nlp_frontend.arro_client import ArroServerError
from arro_nlp_frontend.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()


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


class SearchResult(BaseModel):
    """A single hydrated search result."""

    rank: int
    score: float
    row_index: int
    doc_id: str
    text: str
    metadata: dict


class SearchResponse(BaseModel):
    """Response body for POST /search."""

    results: list[SearchResult]
    query_time_ms: int


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post("/search", response_model=SearchResponse, tags=["search"])
async def search(
    request: SearchRequest,
    req: Request,
) -> SearchResponse:
    """Embed query text, retrieve ranked results from arro-server, hydrate from store.

    Pipeline (no lock required -- this is a pure read path):
      1. Validate: non-empty query string
      2. Embed query -> float64 vector (dim,)
      3. POST /api/datasets/{id}/search with vector, top_k, tau
      4. Hydrate each returned row_index from DocumentStore
         Missing rows are logged and skipped (data inconsistency, not a hard error)
      5. Return ranked, hydrated results

    Raises:
      400: query is empty or whitespace-only
      502: arro-server unreachable or returned non-2xx
    """
    t0 = time.perf_counter()

    # Step 1 -- validate query
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")

    embedder = req.app.state.embedder
    store = req.app.state.store
    arro_client = req.app.state.arro_client

    # Step 2 -- embed (single vector, no chunking needed)
    vector = embedder.encode_batch([request.query])[0]

    # Step 3 -- resolve tau: per-request override, else settings default
    tau = request.tau if request.tau is not None else settings.arro_server_search_tau

    # Step 4 -- call arro-server
    try:
        hits = await arro_client.search(
            dataset_id=request.dataset_id,
            vector=vector,
            top_k=request.top_k,
            tau=tau,
        )
    except ArroServerError as exc:
        logger.error("[search] arro-server search failed: %s", exc)
        raise HTTPException(
            status_code=502,
            detail=f"arro-server error: {exc}",
        ) from exc

    # Step 5 -- hydrate from store
    results: list[SearchResult] = []
    for hit in hits:
        doc = store.get_by_row(dataset_id=request.dataset_id, row_index=hit.index)
        if doc is None:
            logger.warning(
                "[search] row_index=%d returned by arro-server not found in store "
                "(data inconsistency -- index may be stale). Skipping.",
                hit.index,
            )
            continue
        results.append(
            SearchResult(
                rank=len(results) + 1,
                score=hit.score,
                row_index=hit.index,
                doc_id=doc.doc_id,
                text=doc.text,
                metadata=doc.metadata,
            )
        )

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    logger.info(
        "[search] dataset=%s query=%r top_k=%d tau=%.2f hits=%d hydrated=%d duration_ms=%d",
        request.dataset_id,
        request.query[:60],
        request.top_k,
        tau,
        len(hits),
        len(results),
        elapsed_ms,
    )

    return SearchResponse(results=results, query_time_ms=elapsed_ms)
