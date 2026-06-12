# arro-nlp-frontend — Roadmap & Architecture Analysis

## Issue tracker

**On [`arro-nlp-frontend`](https://github.com/Genefold/arro-nlp-frontend)**

| # | Title | Status | Depends on | Blocks |
|---|---|---|---|---|
| [4](https://github.com/Genefold/arro-nlp-frontend/issues/4) | `STORE_DB_PATH` in Settings | ✅ Merged | — | #5, #6, #7 |
| [5](https://github.com/Genefold/arro-nlp-frontend/issues/5) | `store.py` — generic `DocumentStore` | ✅ Merged | #4 | #6, #7, #8 |
| [6](https://github.com/Genefold/arro-nlp-frontend/issues/6) | `POST /ingest` | ✅ Merged (PR #11) | #4, #5 | #7, cve-search#9 |
| [7](https://github.com/Genefold/arro-nlp-frontend/issues/7) | `POST /search` | 🔄 In review (PR #16) | #4, #5, #6 | #8, webapp |
| [8](https://github.com/Genefold/arro-nlp-frontend/issues/8) | `GET/DELETE /documents/{doc_id}` | ⏳ Pending | #5, #6 | leaf |

**On [`arro-cve-search`](https://github.com/Genefold/arro-cve-search)**

| # | Title | Status | Depends on |
|---|---|---|---|
| [9](https://github.com/Genefold/arro-cve-search/issues/9) | `harness/ingest.py` — CVE HTTP ingest | ⏳ Pending | arro-nlp-frontend#6 |

## Recommended implementation order

#4 → #5 → #6 → #7 → #8, then cve-search #9 last.

---

# Codebase Analysis

*As of PR #16. Every section covers one architectural layer: design rationale, the specific risks introduced by that design, and the correct future path.*

---

## Layer 1 — Configuration (`config.py`)

### Design

`pydantic-settings` loads from env vars and `.env` with priority: env > `.env` > field defaults. Validators enforce `embed_backend ∈ {"local","openai"}`, `embed_scale_factor > 0`, `arro_server_search_tau ∈ [0.0, 1.0]`, and `OPENAI_API_KEY` presence when `backend=openai`. All service coordinates (`arro_server_url`, `store_db_path`, `host`, `port`) are env-configurable with sensible defaults.

### Risks

- **Silent mis-configuration at scale.** `arro_server_dataset_id` and `arro_server_root_label` have string defaults (`"cve/embeddings"`, `"main"`) with no validation that they match what arro-server has registered. A wrong `dataset_id` causes silent 404s from `dataset_metadata()` (treated as "new dataset") and a fresh Zarr upload that replaces the existing index. No warning is emitted.
- **`ingest_batch_size` is module-level.** `EMBED_CHUNK = settings.ingest_batch_size` is evaluated at import time in `ingest.py`. Changing the env var at runtime has no effect without a server restart. This is intentional but undocumented.
- **No config schema export.** There is no `/config` or `/settings` endpoint. Operators cannot introspect active config without reading env/logs.

### Future work

- Add a `GET /admin/config` endpoint (read-only, redact secrets) for operator visibility.
- Add a validator that pings arro-server at startup and logs a warning if `dataset_id` does not match any registered dataset.
- Document `EMBED_CHUNK` import-time evaluation in the field docstring.

---

## Layer 2 — Embedder (`embedder.py`)

### Design

Two backends: `local` (SentenceTransformer, default `all-MiniLM-L6-v2`, 384-dim) and `openai`. Vectors are **never L2-normalised** — arro-server expects raw scaled vectors and manages its own spectral distance. `scale_factor` is applied uniformly post-encoding. Empty input returns `(0, dim)` without raising. Fails loud at construction if `model_path` is set but does not exist on disk, preventing silent vector distribution mixing.

### Risks

- **`dim` property lies for OpenAI backend.** `dim` returns `384` hardcoded for OpenAI because `_client` has no `.get_embedding_dimension()`. If the operator sets `EMBED_BACKEND=openai` with `text-embedding-3-large` (3072-dim), `dim` returns 384 and the Zarr array is created with `shape=(N, 384)` — truncating or erroring silently depending on NumPy behaviour.
- **`encode_batch` is synchronous and CPU-bound, called inside the async event loop.** For the local backend, `SentenceTransformer.encode()` blocks the event loop for the full embedding duration. Under concurrent ingest requests this does not cause corruption (the lock is acquired *after* embedding), but it does block all other async work including search requests and health checks. For a single worker handling mixed load, a 1000-doc ingest batch can block the event loop for several seconds.
- **No text length validation.** `IngestItem.text` has `min_length=1` but no `max_length`. Feeding a 100K-token document to `all-MiniLM-L6-v2` does not raise — SentenceTransformer silently truncates at 256 or 512 tokens depending on model config. The stored `text` is the full original, but the embedding represents only the truncated prefix. Search quality degrades silently.
- **`scale_factor` interaction with arro-server tau.** The tau parameter in arro-server assumes a specific vector distribution. Applying an arbitrary `scale_factor != 1.0` shifts norms and can invalidate tau thresholds calibrated on unit-norm vectors. No warning is logged when `scale_factor != 1.0`.

### Future work

- Fix `dim` for OpenAI: call `client.models.retrieve(model_name)` or maintain a `{model: dim}` lookup table.
- Offload `encode_batch` to a thread pool: `await asyncio.get_event_loop().run_in_executor(None, embedder.encode_batch, texts)`.
- Add `max_length: int` field to `IngestItem` (default 2048 chars) with a validator that truncates with a warning log.
- Log a `WARNING` when `scale_factor != 1.0` at embedder construction time.

---

## Layer 3 — Document Store (`store.py`)

### Design

Stdlib `sqlite3` only — no ORM. WAL journal mode for concurrent reads. `upsert_batch` is atomic (single transaction). `row_index = MAX(row_index) + 1`, not `COUNT(*)`, to survive soft deletes. `delete_by_id` is a soft delete: the vector in arro-server's Zarr array is never removed. Vectors are stored as raw `BLOB` (NumPy `.tobytes()` / `np.frombuffer`) alongside the document.

### Risks

- **`asyncio.Lock` does not protect multi-worker or multi-replica deployments.** The lock serialises access within a single event loop. With `uvicorn --workers 4`, four processes share the same SQLite file but each has its own `asyncio.Lock`. Two workers can read the same `MAX(row_index)` simultaneously and assign the same `start_row`. The resulting Zarr upload from worker B will overwrite worker A's upload with a matrix that is missing A's documents. Data loss, no error.
- **Re-ingest with existing `doc_id` creates a ghost vector.** `INSERT OR REPLACE` deletes the old row and inserts a new one. If `doc_id="CVE-1"` was at `row_index=0` and is re-ingested, the new row gets `row_index = next_row_index()` (e.g. 42). The Zarr rewrite includes the new vector at position 42 and the old position 0 is now empty in SQLite — but arro-server's previous index still has the old embedding at index 0. Until the next successful ingest triggers a Zarr rewrite and `build_index`, search can return ghost hits at index 0 with no matching document in the store. `search.py` handles this gracefully (skip + log), but the degradation period is unbounded.
- **`get_all_vectors()` reads the full matrix into RAM on every ingest.** For 100K documents at 384-dim float64: 100K × 384 × 8 bytes = ~300MB per ingest call. This is read, assembled in Python, written to Zarr, then discarded. Peak RSS during ingest ≈ 2× the matrix size (Python list + NumPy stack).
- **`check_same_thread=False` with a future `run_in_executor` is a trap.** If any future code moves SQLite calls into a thread pool executor, the `asyncio.Lock` no longer protects them — it is not a threading lock. The code is safe today because everything runs on the event loop thread, but the invariant is fragile and undocumented.
- **No migration path.** There is no schema versioning. Adding a column requires dropping and recreating the table. The `vector BLOB` column is new in PR #11 — any existing deployment with a pre-#11 SQLite file will fail silently (the column is absent, `get_all_vectors()` returns an empty array).

### Future work

- Add `BEGIN EXCLUSIVE` transaction for `next_row_index` + `upsert_batch` to protect multi-worker deployments on the same host (SQLite-level advisory lock, compatible with WAL).
- Implement `POST /admin/reindex`: truncate store, re-embed all documents, rewrite Zarr from scratch. Required before multi-replica is safe.
- Add schema version table and a startup migration check that fails loud if the schema is stale.
- Cap `get_all_vectors()` with a configurable `max_ingest_rows` setting; above the cap, require explicit `POST /admin/reindex` instead of inline rewrite.
- Document the `asyncio.Lock` + `check_same_thread=False` invariant in the store docstring.

---

## Layer 4 — ArroClient (`arro_client.py`)

### Design

Thin async HTTP wrapper using `httpx.AsyncClient`. Single responsibility: translate HTTP ↔ Python types. All methods raise `ArroServerError` on non-2xx or network failure. Five methods: `dataset_metadata`, `upload_init`, `upload_commit`, `build_index`, `search`. The `search()` method sends `{vector, k, tau, mode="tau"}` to `POST /api/datasets/{id}/search`.

### Risks

- **No retry logic.** A transient 503 from arro-server during `upload_commit` causes a 502 to the caller and leaves the store written but arro-server unsynced. There is no automatic retry and no dead-letter queue. The self-healing mechanism (next ingest rewrites Zarr) only works if another ingest happens — for low-traffic deployments, the index can stay stale indefinitely.
- **`vector.tolist()` in `search()` is a Python-level serialisation bottleneck.** A single 384-dim float64 vector → JSON string ≈ 3KB. This is negligible for search (one vector per request). For ingest, the same pattern applied to the full matrix (see `store.py` risks) is the primary bottleneck.
- **`timeout=30.0` is a single scalar.** `build_index` on a large dataset can take minutes. With `timeout=30.0`, a legitimate long-running index build will raise `httpx.ReadTimeout` → `ArroServerError` → 502, leaving the index rebuild aborted mid-flight with no way to poll status.
- **No connection pooling configuration.** `httpx.AsyncClient` pools connections by default, but there is no explicit configuration of pool size, keepalive, or TLS settings. In a high-ingest scenario with many concurrent `upload_commit` calls (if the lock is ever relaxed), connection exhaustion is possible.
- **`aclose()` is only called from lifespan shutdown.** If the process is killed (SIGKILL), the httpx client is not closed. This is acceptable for stateless HTTP but worth documenting.

### Future work

- Add `tenacity`-based retry with exponential backoff for `upload_commit` and `build_index` (idempotent operations).
- Use a separate, longer timeout for `build_index` (e.g. `timeout=600.0`) or implement an async poll loop against a `GET /api/datasets/{id}/index/status` endpoint if arro-server exposes one.
- Consider Arrow IPC or msgpack for vector payloads once arro-server supports it — reduces search payload from ~3KB JSON to ~1.5KB binary per 384-dim vector.

---

## Layer 5 — Ingest endpoint (`ingest.py`)

### Design

Two-phase pipeline: embed outside the lock (CPU-bound, no shared state), then acquire `asyncio.Lock` for the atomic SQLite write + Zarr rewrite. SQLite is written first — a 502 from arro-server leaves the document in the store but the index stale. The next successful ingest self-heals by rewriting the full Zarr. Index rebuild is triggered only when `index_stale=True` or the dataset is new.

### Risks

- **Full Zarr rewrite on every ingest is O(N) in dataset size.** Every ingest — even a single document — reads all N vectors from SQLite, writes a new Zarr array of shape (N, dim), and calls `upload_commit`. At 100K documents: ~300MB read + ~300MB write per ingest call. Ingest throughput degrades linearly as the dataset grows.
- **The lock holds during the full Zarr write.** The `asyncio.Lock` is held from `next_row_index()` through `build_index()`. For a large dataset, the lock is held for the duration of the Zarr write (potentially tens of seconds). All concurrent ingest requests queue behind it. Search requests are unaffected (no lock), but ingest throughput is effectively serialised.
- **No batch size cap at the endpoint level.** `IngestRequest.documents` has `min_length=1` but no `max_length`. A single request with 10K documents triggers a 10K-document embed + a full Zarr rewrite of N+10K vectors. OOM is possible on small deployments.
- **`EMBED_CHUNK` is module-level at import time.** Changing `INGEST_BATCH_SIZE` env var requires a server restart.

### Future work

- Add `max_length=500` (configurable) to `IngestRequest.documents`.
- Investigate append-only Zarr writes (arro-server permitting) to avoid full rewrites: `upload_init` with an `append` mode, write only the new rows, `upload_commit`. This changes the ingest pipeline from O(N) to O(batch).
- Move `EMBED_CHUNK` to a per-request resolved value from `settings` rather than a module-level constant.
- Add `POST /admin/reindex` for full rebuild after arro-server downtime or schema migration.

---

## Layer 6 — Search endpoint (`search.py`)

### Design

Pure read path — no lock. Embeds the query into a single vector, calls arro-server, hydrates each returned `row_index` from SQLite. Ghost rows (soft-deleted or stale) are silently skipped with a `WARNING` log. Ranks are resequenced `1..N` after any skips. `tau` defaults to `settings.arro_server_search_tau` but can be overridden per-request.

### Risks

- **`encode_batch([query])[0]` is synchronous and blocks the event loop.** Same issue as ingest: a single query embedding call blocks the event loop for ~5–50ms depending on hardware. Under high search concurrency this is a meaningful bottleneck. A 50ms embed × 20 concurrent queries = 1 second of sequential blocking on a single worker.
- **No result deduplication.** If arro-server returns the same `row_index` twice (arro-server bug or index corruption), the store returns the same document twice with different ranks. The response will contain duplicate `doc_id` values with no error.
- **`query_time_ms` includes only endpoint time, not arro-server latency breakdown.** Callers cannot distinguish a slow embed from a slow arro-server call from a slow SQLite hydration. Useful for debugging, but structurally incomplete for SLA monitoring.
- **No caching.** Identical repeated queries embed, call arro-server, and hydrate from SQLite every time. For a CVE search use case where the same queries recur frequently (e.g. `"buffer overflow"`, `"SQL injection"`), a simple LRU cache on the (query, top_k, tau) tuple would eliminate arro-server calls for hot queries.

### Future work

- Offload `encode_batch` to `run_in_executor` for both ingest and search.
- Add deduplication: after hydration, deduplicate by `doc_id`, keep highest-score hit.
- Add structured timing to response: `{"embed_ms", "search_ms", "hydrate_ms"}` inside `SearchResponse` (behind a `debug=true` query param to avoid bloating normal responses).
- Add optional LRU cache for (query_hash, top_k, tau) → results with a configurable TTL.

---

## Scalability profile

| Dimension | Current limit | Root cause | Correct fix |
|---|---|---|---|
| Dataset size | ~50K docs before ingest becomes slow | O(N) Zarr rewrite per ingest | Append-only Zarr writes |
| Ingest throughput | 1 concurrent ingest (lock held for full Zarr write) | `asyncio.Lock` scope too wide | Release lock after SQLite write; move Zarr rewrite outside lock |
| Search concurrency | Degrades under load | Sync embed blocks event loop | `run_in_executor` for embed |
| Multi-worker (`--workers N`) | **Broken** — row index corruption | `asyncio.Lock` is process-local | SQLite `BEGIN EXCLUSIVE` or ingest queue |
| Multi-replica (K8s) | **Broken** — same as multi-worker | Same root cause | Distributed advisory lock or message queue |
| Re-ingest existing doc_id | Ghost vectors accumulate over time | Zarr has no point-delete | Option A: `409 Conflict`; Option B: append + soft-delete at search time |
| Large single document | Silent truncation at model token limit | No `max_length` on `IngestItem.text` | Add `max_length` field + validator |
| Large batch | OOM risk | No `max_length` on `IngestRequest.documents` | Add `max_length=500` |

---

## Known open issues (created during analysis)

| Issue | Title | Priority |
|---|---|---|
| [#12](https://github.com/Genefold/arro-nlp-frontend/issues/12) | Re-ingest creates ghost vectors in arro-server | High |
| [#13](https://github.com/Genefold/arro-nlp-frontend/issues/13) | `asyncio.Lock` does not protect multi-worker or multi-replica | High |
| [#14](https://github.com/Genefold/arro-nlp-frontend/issues/14) | `vectors.tolist()` + JSON does not scale for large batches | Medium |
| [#15](https://github.com/Genefold/arro-nlp-frontend/issues/15) | `app.state` untyped, no mypy verification | Low |

---

## Phased delivery roadmap

### Phase 1 — Complete the current feature set (now)

- Merge PR #16 (`POST /search`)
- Implement issue #8 (`GET/DELETE /documents/{doc_id}`)
- Implement cve-search issue #9 (CVE ingest harness)

### Phase 2 — Correctness & safety (next)

These are not optimisations — they fix silent data corruption:

1. **Fix re-ingest ghost vectors** (issue #12): return `409 Conflict` on re-ingest of existing `doc_id` until append+rebuild is implemented.
2. **Fix multi-worker row index race** (issue #13): wrap `next_row_index` + `upsert_batch` in `BEGIN EXCLUSIVE` SQLite transaction, or enforce `--workers 1` at startup with a hard check.
3. **Add schema migration** (no issue yet): version table + startup check, fail loud if stale.
4. **Fix OpenAI `dim` property** (no issue yet): lookup table or API call, not hardcoded 384.

### Phase 3 — Performance (when dataset > 10K docs)

1. **Offload embed to thread pool** (`run_in_executor` for both ingest and search).
2. **Release lock after SQLite write**: move Zarr rewrite outside the `asyncio.Lock` scope. This requires arro-server to tolerate a brief window where SQLite and the Zarr index are temporarily out of sync.
3. **Cap ingest batch size**: add `max_length=500` to `IngestRequest.documents`.
4. **Add `POST /admin/reindex`**: full rebuild for post-downtime recovery and ghost vector cleanup.

### Phase 4 — Scale-out (when single-process is insufficient)

1. **Append-only Zarr writes**: eliminate O(N) rewrite per ingest; requires arro-server API extension.
2. **Multi-worker safety**: replace `asyncio.Lock` with SQLite `BEGIN EXCLUSIVE` (same-host multi-worker) or an external queue (ARQ + Redis for multi-host).
3. **Search LRU cache**: cache (query_hash, top_k, tau) → results with configurable TTL.
4. **Arrow IPC vector transport**: replace JSON serialisation with Arrow IPC for ingest and search payloads.
