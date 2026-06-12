"""POST /ingest endpoint: embed documents and push vectors to arro-server.

Pipeline: validate -> embed -> check existing -> lock -> next_row_index -> push -> persist.
"""

from __future__ import annotations

import logging
import time
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from arro_nlp_frontend.arro_client import ArroServerError
from arro_nlp_frontend.store import Document

logger = logging.getLogger(__name__)

router = APIRouter()


class IngestItem(BaseModel):
    """A single document to ingest."""

    doc_id: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    metadata: dict = Field(default_factory=dict)


class IngestRequest(BaseModel):
    documents: list[IngestItem] = Field(..., min_length=1)


class IngestResult(BaseModel):
    doc_id: str
    row_index: int
    status: Literal["created", "updated"]


class IngestResponse(BaseModel):
    ingested: int
    results: list[IngestResult]
    duration_ms: int


@router.post("/ingest", response_model=IngestResponse, status_code=200)
async def ingest(
    request: IngestRequest,
    req: Request,
) -> IngestResponse:
    """Ingest documents: embed -> push to arro-server -> persist in store.

    SINGLE-PROCESS GUARANTEE ONLY.
    The asyncio.Lock prevents row index corruption under concurrent async
    requests within a single uvicorn worker. It does NOT protect against
    multiple uvicorn worker processes (--workers N > 1). For multi-worker
    production deployments, a database-level advisory lock or a serialised
    ingest queue is required.

    Pipeline (steps 3-5 inside the lock):
      1. Validate: non-empty list, no duplicate doc_ids within the batch
      2. Embed texts -> float64 array shape (N, dim)
      3. start_row = store.next_row_index()   <- MAX(row_index)+1, not COUNT
      4. Push vectors to arro-server          <- if this fails -> 502, no store write
      5. Persist documents in store           <- upsert_batch(start_row, docs)
      6. Return IngestResponse

    Raises:
      422: duplicate doc_ids within the batch
      502: arro-server unreachable or returned non-2xx
    """
    t0 = time.perf_counter()

    embedder = req.app.state.embedder
    store = req.app.state.store
    arro_client = req.app.state.arro_client

    # Step 1 — validate: no duplicate doc_ids within this batch
    doc_ids = [item.doc_id for item in request.documents]
    if len(doc_ids) != len(set(doc_ids)):
        seen: set[str] = set()
        duplicates: list[str] = []
        for d in doc_ids:
            if d in seen:
                duplicates.append(d)
            seen.add(d)
        raise HTTPException(
            status_code=422,
            detail=f"Duplicate doc_ids in request: {duplicates}",
        )

    # Step 2 — embed (outside lock: pure CPU/IO, does not touch shared state)
    texts = [item.text for item in request.documents]
    vectors = embedder.encode_batch(texts)

    # Snapshot "created" vs "updated" status before acquiring the lock.
    # This read is outside the lock intentionally: it is informational only
    # and does not affect index correctness. A race here would change a
    # status label, not corrupt data.
    statuses: list[Literal["created", "updated"]] = [
        "updated" if store.get_by_id(item.doc_id) is not None else "created"
        for item in request.documents
    ]

    # Steps 3-5 — inside the lock: start_row is stable for the duration
    start_row: int = 0
    async with req.app.state.ingest_lock:
        start_row = store.next_row_index()

        try:
            await arro_client.push_vectors(vectors, start_row)
        except ArroServerError as exc:
            logger.error("[ingest] arro-server push failed at start_row=%d: %s", start_row, exc)
            raise HTTPException(
                status_code=502,
                detail=f"arro-server error: {exc}",
            ) from exc

        documents = [
            Document(
                row_index=start_row + i,
                doc_id=item.doc_id,
                text=item.text,
                metadata=item.metadata,
                ingested_at=None,
            )
            for i, item in enumerate(request.documents)
        ]
        store.upsert_batch(start_row, documents)

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    logger.info(
        "[ingest] ingested %d docs start_row=%d elapsed=%dms",
        len(documents),
        start_row,
        elapsed_ms,
    )

    results = [
        IngestResult(
            doc_id=item.doc_id,
            row_index=start_row + i,
            status=statuses[i],
        )
        for i, item in enumerate(request.documents)
    ]

    return IngestResponse(
        ingested=len(results),
        results=results,
        duration_ms=elapsed_ms,
    )
