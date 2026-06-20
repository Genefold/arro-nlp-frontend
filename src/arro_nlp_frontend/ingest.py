"""POST /ingest endpoint: embed, store in SQLite, and sync to arro-server via Zarr.

Pipeline: validate -> embed (chunked) -> lock -> next_row_index -> persist ->
          read all vectors -> Zarr rewrite -> upload_commit -> build_index.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from typing import Literal

import numpy as np
import zarr
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from arro_nlp_frontend.arro_client import ArroClient, ArroServerError, VectorAppendResult
from arro_nlp_frontend.config import settings
from arro_nlp_frontend.embedder import Embedder
from arro_nlp_frontend.store import Document, DocumentStore

logger = logging.getLogger(__name__)

router = APIRouter()

EMBED_CHUNK = settings.ingest_batch_size


class IngestItem(BaseModel):
    """A single document to ingest."""

    doc_id: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    metadata: dict = Field(default_factory=dict)


class IngestRequest(BaseModel):
    """Request body for POST /ingest."""

    dataset_id: str = Field(
        ...,
        min_length=1,
        description=(
            "arro-server dataset identifier, e.g. 'cve/embeddings' or 'nvd/embeddings'. "
            "All documents in this batch are associated with this dataset."
        ),
    )
    root_label: str = Field(
        default="",
        description=(
            "arro-server data root label. Defaults to settings.arro_server_root_label when empty."
        ),
    )
    documents: list[IngestItem] = Field(..., min_length=1)
    incremental: bool = Field(
        default=False,
        description=(
            "When True, use the incremental pipeline: classify each document as "
            "new / changed / metadata-only, embed only new+changed, call "
            "append_vectors for new rows and overwrite_vectors for changed rows. "
            "When False (default), the full-rewrite pipeline is used (unchanged behaviour)."
        ),
    )


class IngestResult(BaseModel):
    doc_id: str
    row_index: int
    status: Literal["created", "updated", "skipped"]


class IngestResponse(BaseModel):
    ingested: int
    results: list[IngestResult]
    duration_ms: int


def _text_fingerprint(text: str) -> bytes:
    """Return a stable 8-byte fingerprint of *text* for change detection.

    Uses SHA-256 truncated to 8 bytes. Unlike Python's built-in hash(),
    this is deterministic across processes and interpreter restarts
    (PYTHONHASHSEED does not affect it).

    This is intentionally NOT a cryptographic check -- it is only used to
    decide whether a document's text has changed since last ingest.
    Collision probability for 8 bytes (~1 in 18 quintillion) is acceptable
    for this use case.

    Parameters
    ----------
    text: The document text to fingerprint.

    Returns
    -------
    8-byte bytes object.
    """
    return hashlib.sha256(text.encode()).digest()[:8]


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


# ---------------------------------------------------------------------------
# Incremental pipeline helpers
# ---------------------------------------------------------------------------
#
# Design constraints:
#   - The per-dataset asyncio.Lock wraps ONLY the write section
#     (consistency guard + append + overwrite + SQLite upsert).
#     The embed step runs OUTSIDE the lock to maximise concurrency.
#   - Text change detection uses SHA-256[:8] (deterministic across restarts).
#     Direct text comparison (existing.text != doc.text) is equally valid but
#     requires loading the full text for every document; the fingerprint lets
#     us add a text_fingerprint column to the store in the future without
#     changing this interface.
#   - build_index is called ONCE after all writes, outside the lock.
#     Calling it inside the lock would block concurrent requests to the same
#     dataset for the entire index build duration (~seconds for large datasets).


async def _run_incremental_pipeline(
    request: IngestRequest,
    embedder: Embedder,
    store: DocumentStore,
    arro_client: ArroClient,
    locks: dict[str, asyncio.Lock],
    root_label: str,
) -> IngestResponse:
    """Execute the incremental ingest pipeline.

    Classification
    --------------
    Each document in the batch is classified into one of three buckets:
      new_items:       doc_id not found in the store -> needs embed + append
      changed_items:   doc_id exists but text has changed -> needs embed + overwrite
      metadata_items:  doc_id exists and text is unchanged -> SQLite upsert only

    Embed step (outside lock)
    -------------------------
    Only new_items and changed_items are embedded. metadata_items are skipped
    entirely (no model call, no vector write).

    Write step (inside lock)
    ------------------------
    1. Consistency guard: server nrows must equal store.next_row_index().
       Raises HTTPException 409 if out of sync.
       Called whenever new_items OR changed_items is non-empty — both
       operations depend on row indices being in sync with the server.
       Metadata-only batches (both lists empty) skip the guard because
       they perform no vector writes.
    2. append_vectors for new_items (if any). Row indices are derived from
       VectorAppendResult.start_row + offset.
    3. overwrite_vectors for changed_items (if any).
    4. upsert_batch_with_indices for new_items + changed_items (with vectors).
    5. upsert_batch for metadata_items using their existing row_index (no vector write).

    Index rebuild (outside lock)
    ----------------------------
    build_index is called once at the end if any new or changed documents exist.

    Parameters
    ----------
    request:     The validated IngestRequest (incremental=True).
    embedder:    The Embedder instance from app.state.
    store:       The DocumentStore instance from app.state.
    arro_client: The ArroClient instance from app.state.
    locks:       The app.state.ingest_locks dict.
    root_label:  Resolved root label (request.root_label or settings default).

    Returns
    -------
    IngestResponse with per-document results and duration_ms.

    Raises
    ------
    HTTPException 409: SQLite and Zarr row counts are out of sync.
    HTTPException 502: arro-server returned a non-2xx or is unreachable.
    """
    t0 = time.perf_counter()

    # ------------------------------------------------------------------
    # Step 1 -- Classify documents
    # ------------------------------------------------------------------
    # Three buckets. Each item is (IngestItem, existing_row_index_or_None).
    new_items: list[IngestItem] = []
    changed_items: list[tuple[IngestItem, int]] = []  # (item, existing_row_index)
    metadata_items: list[tuple[IngestItem, int]] = []  # (item, existing_row_index)

    for item in request.documents:
        existing = store.get_by_id(request.dataset_id, item.doc_id)
        if existing is None:
            new_items.append(item)
        elif _text_fingerprint(existing.text) != _text_fingerprint(item.text):
            changed_items.append((item, existing.row_index))
        else:
            metadata_items.append((item, existing.row_index))

    # ------------------------------------------------------------------
    # Step 2 -- Embed new + changed documents (outside lock)
    # ------------------------------------------------------------------
    embed_texts: list[str] = [it.text for it in new_items] + [it.text for it, _ in changed_items]

    if embed_texts:
        chunks = [embed_texts[i : i + EMBED_CHUNK] for i in range(0, len(embed_texts), EMBED_CHUNK)]
        parts = [embedder.encode_batch(chunk) for chunk in chunks]
        all_vecs = np.vstack(parts) if len(parts) > 1 else parts[0]
        new_vecs = all_vecs[: len(new_items)]  # shape (N_new, dim)
        changed_vecs = all_vecs[len(new_items) :]  # shape (N_changed, dim)
    else:
        # All documents are metadata-only -- no embed call needed.
        # new_vecs and changed_vecs are unused in this branch but must be
        # defined to satisfy the variable scope below (no append/overwrite
        # will be called since new_items and changed_items are both empty).
        new_vecs = np.empty((0, 0), dtype=np.float64)
        changed_vecs = np.empty((0, 0), dtype=np.float64)

    # ------------------------------------------------------------------
    # Step 3 -- Write section (inside per-dataset lock)
    # ------------------------------------------------------------------
    lock = _get_dataset_lock(locks, request.dataset_id)
    results: list[IngestResult] = []

    try:
        async with lock:
            # 3a. Consistency guard
            if new_items or changed_items:
                server_count = await arro_client.get_vector_count(request.dataset_id)
                local_count = store.next_row_index(dataset_id=request.dataset_id)
                if server_count != local_count:
                    raise HTTPException(
                        status_code=409,
                        detail=(
                            f"Consistency error for dataset '{request.dataset_id}': "
                            f"SQLite next_row_index={local_count} but "
                            f"arro-server nrows={server_count}. "
                            "Run a full re-ingest to repair the dataset before "
                            "using incremental mode."
                        ),
                    )

            # 3b. Append new vectors
            if new_items:
                append_result: VectorAppendResult = await arro_client.append_vectors(
                    request.dataset_id, new_vecs
                )
                new_docs = [
                    Document(
                        row_index=append_result.start_row + i,
                        doc_id=item.doc_id,
                        text=item.text,
                        metadata=item.metadata,
                        ingested_at=None,
                    )
                    for i, item in enumerate(new_items)
                ]
                store.upsert_batch_with_indices(request.dataset_id, new_docs, new_vecs)
                results.extend(
                    IngestResult(
                        doc_id=doc.doc_id,
                        row_index=doc.row_index,
                        status="created",
                    )
                    for doc in new_docs
                )

            # 3c. Overwrite changed vectors
            if changed_items:
                updates = [
                    (row_idx, changed_vecs[i]) for i, (_, row_idx) in enumerate(changed_items)
                ]
                await arro_client.overwrite_vectors(request.dataset_id, updates)
                changed_docs = [
                    Document(
                        row_index=row_idx,
                        doc_id=item.doc_id,
                        text=item.text,
                        metadata=item.metadata,
                        ingested_at=None,
                    )
                    for (item, row_idx) in changed_items
                ]
                store.upsert_batch_with_indices(request.dataset_id, changed_docs, changed_vecs)
                results.extend(
                    IngestResult(
                        doc_id=doc.doc_id,
                        row_index=doc.row_index,
                        status="updated",
                    )
                    for doc in changed_docs
                )

            # 3d. Metadata-only upsert
            if metadata_items:
                meta_row_indices = [row_idx for (_, row_idx) in metadata_items]
                all_stored_vecs = store.get_all_vectors(request.dataset_id)
                meta_vecs = np.stack([all_stored_vecs[row_idx] for row_idx in meta_row_indices])
                meta_docs = [
                    Document(
                        row_index=row_idx,
                        doc_id=item.doc_id,
                        text=item.text,
                        metadata=item.metadata,
                        ingested_at=None,
                    )
                    for (item, row_idx) in metadata_items
                ]
                store.upsert_batch_with_indices(request.dataset_id, meta_docs, meta_vecs)
                results.extend(
                    IngestResult(
                        doc_id=doc.doc_id,
                        row_index=doc.row_index,
                        status="skipped",
                    )
                    for doc in meta_docs
                )

    except HTTPException:
        raise
    except ArroServerError as exc:
        logger.error(
            "[ingest][incremental] arro-server error dataset=%s: %s",
            request.dataset_id,
            exc,
        )
        raise HTTPException(
            status_code=502,
            detail=f"arro-server error: {exc}",
        ) from exc

    # ------------------------------------------------------------------
    # Step 4 -- Rebuild index (outside lock)
    # ------------------------------------------------------------------
    if new_items or changed_items:
        try:
            await arro_client.build_index(dataset_id=request.dataset_id, timeout=600.0)
        except ArroServerError as exc:
            logger.error(
                "[ingest][incremental] build_index failed dataset=%s: %s",
                request.dataset_id,
                exc,
            )
            raise HTTPException(
                status_code=502,
                detail=f"arro-server build_index error: {exc}",
            ) from exc

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    logger.info(
        "[ingest][incremental] dataset=%s new=%d changed=%d skipped=%d elapsed=%dms",
        request.dataset_id,
        len(new_items),
        len(changed_items),
        len(metadata_items),
        elapsed_ms,
    )

    # Restore original request order in the response.
    doc_id_to_result = {r.doc_id: r for r in results}
    ordered_results = [doc_id_to_result[item.doc_id] for item in request.documents]

    return IngestResponse(
        ingested=len(ordered_results),
        results=ordered_results,
        duration_ms=elapsed_ms,
    )


@router.post("/ingest", response_model=IngestResponse, status_code=200)
async def ingest(
    request: IngestRequest,
    req: Request,
) -> IngestResponse | JSONResponse:
    """Ingest documents: embed -> store in SQLite -> Zarr rewrite -> index.

    SINGLE-PROCESS GUARANTEE ONLY.
    The per-dataset asyncio.Lock prevents row index corruption under
    concurrent async requests within a single uvicorn worker. It does NOT
    protect against multiple uvicorn worker processes (--workers N > 1).
    For multi-worker production deployments, a database-level advisory lock
    or a serialised ingest queue is required.

    LOCK IS PER-DATASET.
    A separate asyncio.Lock is created lazily for each dataset_id via
    _get_dataset_lock(). Concurrent requests to different datasets run in
    parallel. Concurrent requests to the same dataset are serialised to
    protect the row_index counter and the Zarr rewrite.

    Pipeline:
       1. Validate: non-empty list, no duplicate doc_ids within the batch
       2. Embed texts in chunks of EMBED_CHUNK -> float64 array (N, dim)
       2a. (inside lock) -- per-dataset lock acquired via `_get_dataset_lock`
       3. (inside lock) start_row = store.next_row_index()
       4. (inside lock) upsert_batch into SQLite with vectors
       5. (inside lock) Sync to arro-server:
          a. Read ALL vectors from SQLite for this dataset (full matrix)
          b. upload_init -> upload_path
          c. Write Zarr v3 array to upload_path
          d. upload_commit
          e. build_index (always on full re-ingest; see issue #26)
             On incremental: only if new_items or changed_items exist.

    Raises:
      422: duplicate doc_ids within the batch
      502: arro-server unreachable or returned non-2xx
    """
    t0 = time.perf_counter()

    embedder = req.app.state.embedder
    store = req.app.state.store
    arro_client = req.app.state.arro_client

    root_label = request.root_label or settings.arro_server_root_label

    # Step 1 -- validate: no duplicate doc_ids within this batch
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

    # Guard against ghost-row creation on full-rewrite pipeline.
    if not request.incremental:
        existing_ids = store.get_existing_ids(request.dataset_id, doc_ids)
        if existing_ids:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"doc_ids already exist in dataset '{request.dataset_id}': "
                    f"{sorted(existing_ids)}. "
                    "Use incremental=True to update documents in place, or "
                    "DELETE the documents before re-ingesting with the full pipeline."
                ),
            )

    # Step 2 -- Route to incremental pipeline if requested
    if request.incremental:
        return await _run_incremental_pipeline(
            request=request,
            embedder=embedder,
            store=store,
            arro_client=arro_client,
            locks=req.app.state.ingest_locks,
            root_label=root_label,
        )

    # Step 2 (full path) -- embed in chunks to avoid OOM on large requests
    texts = [item.text for item in request.documents]
    chunks = [texts[i : i + EMBED_CHUNK] for i in range(0, len(texts), EMBED_CHUNK)]
    parts = [embedder.encode_batch(chunk) for chunk in chunks]
    vectors = np.vstack(parts) if len(parts) > 1 else parts[0]

    # Steps 3-5 -- inside the lock: start_row is stable for the duration
    lock = _get_dataset_lock(req.app.state.ingest_locks, request.dataset_id)

    # Track whether upload_commit succeeded so we know whether rollback
    # is safe (pre-commit failure) or harmful (post-commit, build_index failed).
    _upload_committed = False
    _rolled_back = False

    async with lock:
        start_row = store.next_row_index(dataset_id=request.dataset_id)

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
        statuses = store.upsert_batch(request.dataset_id, start_row, documents, vectors)

        # Step 5 -- Sync to arro-server via Zarr rewrite
        try:
            # 5a. Read ALL current vectors from store
            all_vectors = store.get_all_vectors(dataset_id=request.dataset_id)

            # 5b. Initialise an upload slot on arro-server
            server_upload_path = await arro_client.upload_init(
                dataset_id=request.dataset_id, root_label=root_label
            )
            upload_path = settings.arro_server_upload_path or server_upload_path

            # 5c. Write the full embedding matrix as a Zarr v3 array
            arr = zarr.open_array(
                upload_path,
                mode="w",
                shape=all_vectors.shape,
                dtype="float64",
            )
            arr[:] = all_vectors

            # 5d. Commit the upload
            await arro_client.upload_commit(dataset_id=request.dataset_id, fs_path=upload_path)
            # Mark commit as successful BEFORE build_index.
            # If build_index fails, the vectors are already on the server --
            # rolling back SQLite here would corrupt the dataset.
            _upload_committed = True

            # 5e. Always rebuild the index on full re-ingest.
            # Index builds on large datasets take several minutes; use a
            # generous per-request timeout to avoid httpx.ReadTimeout.
            # Unconditional: after a volume wipe (down -v), arro-server can
            # report index_stale=False (data identical to bulk) and
            # is_new=False (dataset metadata exists) even though the physical
            # index is gone.  Calling build_index unconditionally is safe:
            # if the index is already valid, arro-server returns quickly.
            await arro_client.build_index(dataset_id=request.dataset_id, timeout=600.0)

        except ArroServerError as exc:
            if not _upload_committed:
                # Pre-commit failure: arro-server never received the vectors.
                # Roll back SQLite to restore consistency.
                rolled_back_count = store.rollback_rows(
                    request.dataset_id,
                    list(range(start_row, start_row + len(documents))),
                )
                _rolled_back = True
                logger.error(
                    "[ingest] arro-server sync failed BEFORE commit "
                    "dataset=%s start_row=%d rolled_back=%d: %s",
                    request.dataset_id,
                    start_row,
                    rolled_back_count,
                    exc,
                )
                return JSONResponse(
                    status_code=502,
                    headers={"X-Partial-Write": "rolled-back"},
                    content={
                        "detail": (
                            f"arro-server sync failed before upload_commit: {exc}. "
                            "SQLite has been rolled back. "
                            "Re-ingest is safe — no manual repair needed."
                        )
                    },
                )
            else:
                # Post-commit failure: vectors exist on arro-server but index
                # is not built. SQLite is correct. Do NOT rollback.
                logger.error(
                    "[ingest] build_index failed AFTER commit "
                    "dataset=%s start_row=%d: %s. "
                    "Data is on arro-server but index may be stale. "
                    "Re-trigger build_index or re-ingest.",
                    request.dataset_id,
                    start_row,
                    exc,
                )
                return JSONResponse(
                    status_code=502,
                    headers={"X-Partial-Write": "committed-index-stale"},
                    content={
                        "detail": (
                            f"arro-server build_index failed after successful "
                            f"upload_commit: {exc}. "
                            "Vectors are on arro-server but the search index "
                            "may be stale. Re-ingest to rebuild the index — "
                            "no data loss."
                        )
                    },
                )

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    logger.info(
        "[ingest] ingested %d docs dataset=%s start_row=%d elapsed=%dms",
        len(documents),
        request.dataset_id,
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
