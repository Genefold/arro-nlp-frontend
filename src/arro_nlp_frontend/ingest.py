"""POST /ingest endpoint: embed, store in SQLite, and sync to arro-server via Zarr.

Pipeline: validate -> embed (chunked) -> lock -> next_row_index -> persist ->
          read all vectors -> Zarr rewrite -> upload_commit -> build_index.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Literal

import numpy as np
import zarr
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from arro_nlp_frontend.arro_client import ArroServerError
from arro_nlp_frontend.config import settings
from arro_nlp_frontend.store import Document

logger = logging.getLogger(__name__)

router = APIRouter()

EMBED_CHUNK = settings.ingest_batch_size


class IngestItem(BaseModel):
    """A single document to ingest."""

    doc_id: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    metadata: dict = Field(default_factory=dict)


class IngestRequest(BaseModel):
    dataset_id: str = Field(
        default="default", description="Dataset namespace for per-dataset locking"
    )
    documents: list[IngestItem] = Field(..., min_length=1)


class IngestResult(BaseModel):
    doc_id: str
    row_index: int
    status: Literal["created", "updated"]


class IngestResponse(BaseModel):
    ingested: int
    results: list[IngestResult]
    duration_ms: int


def _get_dataset_lock(locks: dict[str, asyncio.Lock], dataset_id: str) -> asyncio.Lock:
    """Return the asyncio.Lock for *dataset_id*, creating it lazily if absent.

    Thread-safety note: this function must only be called from within the
    asyncio event loop (single-threaded). Dict mutation here is safe because
    asyncio is cooperative: no two coroutines can reach this point concurrently
    for the same dataset_id without one of them already holding the lock.
    """
    if dataset_id not in locks:
        locks[dataset_id] = asyncio.Lock()
    return locks[dataset_id]


@router.post("/ingest", response_model=IngestResponse, status_code=200)
async def ingest(
    request: IngestRequest,
    req: Request,
) -> IngestResponse:
    """Ingest documents: embed -> store in SQLite -> Zarr rewrite -> index.

    SINGLE-PROCESS GUARANTEE ONLY.
    The asyncio.Lock prevents row index corruption under concurrent async
    requests within a single uvicorn worker. It does NOT protect against
    multiple uvicorn worker processes (--workers N > 1). For multi-worker
    production deployments, a database-level advisory lock or a serialised
    ingest queue is required.

    Pipeline:
      1. Validate: non-empty list, no duplicate doc_ids within the batch
      2. Embed texts in chunks of EMBED_CHUNK -> float64 array (N, dim)
      2a. (inside lock) — per-dataset lock acquired via `_get_dataset_lock`
      3. (inside lock) start_row = store.next_row_index()
      4. (inside lock) upsert_batch into SQLite with vectors
      5. (inside lock) Sync to arro-server:
         a. Read ALL vectors from SQLite (full matrix)
         b. Check if dataset exists (dataset_metadata -> 404 = new)
         c. upload_init -> upload_path
         d. Write Zarr v3 array to upload_path
         e. upload_commit -> index_stale
         f. If index_stale or new dataset -> build_index

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

    # Step 2 — embed in chunks to avoid OOM on large requests
    texts = [item.text for item in request.documents]
    chunks = [texts[i : i + EMBED_CHUNK] for i in range(0, len(texts), EMBED_CHUNK)]
    parts = [embedder.encode_batch(chunk) for chunk in chunks]
    vectors = np.vstack(parts) if len(parts) > 1 else parts[0]

    # Steps 3-5 — inside the lock: start_row is stable for the duration
    lock = _get_dataset_lock(req.app.state.ingest_locks, request.dataset_id)
    async with lock:
        start_row = store.next_row_index()

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
        statuses = store.upsert_batch(start_row, documents, vectors)

        # Step 5 — Sync to arro-server via Zarr rewrite
        try:
            # 5a. Read ALL current vectors from store
            all_vectors = store.get_all_vectors()

            # 5b. Check if the dataset already exists on arro-server
            meta = await arro_client.dataset_metadata()
            is_new = meta is None

            # 5c. Initialise an upload slot on arro-server.
            # upload_init validates dataset_id + root and registers the slot.
            # If arro_server_upload_path is set, we override the returned path
            # (shared-volume deployments) but still call upload_init to register.
            server_upload_path = await arro_client.upload_init()
            upload_path = settings.arro_server_upload_path or server_upload_path

            # 5d. Write the full embedding matrix as a Zarr v3 array
            arr = zarr.open_array(
                upload_path,
                mode="w",
                shape=all_vectors.shape,
                dtype="float64",
                zarr_version=3,
            )
            arr[:] = all_vectors

            # 5e. Commit the upload
            commit_result = await arro_client.upload_commit(upload_path)

            # 5f. Rebuild index if stale or new dataset
            if commit_result.index_stale or is_new:
                await arro_client.build_index()

        except ArroServerError as exc:
            logger.error("[ingest] arro-server sync failed at start_row=%d: %s", start_row, exc)
            raise HTTPException(
                status_code=502,
                detail=f"arro-server error: {exc}",
            ) from exc

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
