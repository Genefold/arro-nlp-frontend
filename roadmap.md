# arro-nlp-frontend — Roadmap & Architecture Analysis

## Issue tracker

**On [`arro-nlp-frontend`](https://github.com/Genefold/arro-nlp-frontend)**

| # | Title | Status | Depends on | Blocks |
|---|---|---|---|---|
| [4](https://github.com/Genefold/arro-nlp-frontend/issues/4) | `STORE_DB_PATH` in Settings | ✅ Merged (PR #9) | — | #5, #6, #7 |
| [5](https://github.com/Genefold/arro-nlp-frontend/issues/5) | `store.py` — generic `DocumentStore` | ✅ Merged (PR #10) | #4 | #6, #7, #8 |
| [6](https://github.com/Genefold/arro-nlp-frontend/issues/6) | `POST /ingest` | ✅ Merged (PR #11) | #4, #5 | #7, cve-search#9 |
| [7](https://github.com/Genefold/arro-nlp-frontend/issues/7) | `POST /search` | ✅ Merged (PR #16) | #4, #5, #6 | #8, webapp |
| [8](https://github.com/Genefold/arro-nlp-frontend/issues/8) | `GET/DELETE /documents/{doc_id}` | ⏳ Pending | #5, #6 | leaf |
| [17](https://github.com/Genefold/arro-nlp-frontend/issues/17) | Multi-dataset serving — DocumentStore v2 | ✅ Merged (PR #18) | #6, #7 | cve-search#9 |
| [19](https://github.com/Genefold/arro-nlp-frontend/issues/19) | Per-dataset ingest lock (`dict[str, asyncio.Lock]`) | ✅ Merged (PR #20) | PR #18 | — |

**On [`arro-cve-search`](https://github.com/Genefold/arro-cve-search)**

| # | Title | Status | Depends on |
|---|---|---|---|
| [9](https://github.com/Genefold/arro-cve-search/issues/9) | `harness/ingest.py` — CVE HTTP ingest | ⏳ Pending | arro-nlp-frontend PR #20 merged |

---

## Where we are right now

### ✅ Complete and merged

- **Config** (`#4` / PR #9): `STORE_DB_PATH`, `ARRO_SERVER_*` settings, `.env.example`.
- **DocumentStore v1** (`#5` / PR #10): SQLite-backed store, WAL, atomic upsert, soft delete, `MAX`-based row index.
- **`POST /ingest`** (`#6` / PR #11): embed → SQLite → Zarr rewrite → arro-server push. Full offline test suite (19 tests).
- **`POST /search`** (`#7` / PR #16): embed query → arro-server → hydrate from store. Ghost row handling. tau override. 11 tests.
- **Multi-dataset serving** (PR #18, closes #17): DocumentStore v2, per-call `dataset_id`, `migrate.py`, 19 new tests.
- **Per-dataset ingest lock** (PR #20, closes #19): `dict[str, asyncio.Lock]` keyed by `dataset_id`, `_get_dataset_lock()` helper, 2 new concurrent tests (cross-dataset parallelism + same-dataset serialisation). Ingest throughput for N datasets is now N-concurrent.

### ⏳ Pending — this repo

- **Issue #8** — `GET /documents/{doc_id}` and `DELETE /documents/{doc_id}`: read and soft-delete endpoints backed by the store. Low priority for the CVE search use case but required for a complete REST API.

### ⏳ Pending — `arro-cve-search`

- **Issue #9** — `harness/ingest.py`: the CVE ingest harness. Ingest calls must include `dataset_id` in the request body: `POST /ingest` with `{"dataset_id": "cve/embeddings", "documents": [...]}`. See **Gap analysis** section below.

---

## What is missing for arro-cve-search to work end-to-end

This section lists every gap between the current state of `arro-nlp-frontend` and a working CVE semantic search product. Items are ordered by the sequence in which they must be resolved.

### Gap 1 — `harness/ingest.py` does not exist (or is not updated for v2 API)

The ingest harness needs to:
- Parse NVD/CVE JSON feed (or CVE List v5 format).
- Extract `cve_id`, `description`, and structured `metadata` (CVSS score, CWE, affected products, published date) per CVE.
- `POST /ingest` to `arro-nlp-frontend` in batches of ≤ 500 docs with `dataset_id: "cve/embeddings"`.
- Handle `502` responses from the frontend (arro-server sync failure) with retry.
- Track ingested CVE IDs to support incremental re-runs (only ingest new/updated CVEs).
- Report progress: total ingested, created vs updated, elapsed time.

This is `arro-cve-search#9`. It has no implementation yet.

### Gap 2 — No CVE data source is configured

The harness needs a CVE data source. Options:
- **NVD REST API v2** (`https://services.nvd.nist.gov/rest/json/cves/2.0`): paginated, rate-limited (50 req/30s without API key, 2000 req/30s with key). Requires an NVD API key env var.
- **CVE List v5 GitHub mirror** (`https://github.com/CVEProject/cvelistV5`): full dataset as JSON files, no rate limit, requires cloning ~2GB repo or using the GitHub API.
- **OSV.dev API** (`https://api.osv.dev`): broader ecosystem, not CVE-specific.

No data source has been chosen or documented. This must be decided before the harness is written.

### Gap 3 — No search client / webapp

`POST /search` is implemented and tested. Nothing consumes it yet. The CVE search product needs at minimum:
- A CLI query tool (`arro-cve-search/query.py`): `python query.py "heap overflow in network driver"` → ranked CVE list with scores.
- Or a minimal web UI.

This is untracked. Should become `arro-cve-search#10` (or equivalent).

### Gap 4 — `GET/DELETE /documents/{doc_id}` is unimplemented (issue #8)

For a CVE search product this matters when:
- A CVE is rejected/withdrawn from the NVD feed and must be removed from search results.
- A CVE description is updated and the embedding must be refreshed.

Without `DELETE`, the only option is re-ingest (which updates the embedding but leaves a ghost vector in arro-server — see issue #12). This is a known correctness gap for the re-ingest case.

### Gap 5 — No end-to-end integration test

All existing tests run fully offline (mocked arro-server). There is no test that spins up a real arro-server instance, ingests CVE documents, and asserts that search returns the correct results. This gap means a silent protocol mismatch between `arro-nlp-frontend` and `arro-server` would not be caught by CI.

A minimal `tests/integration/test_e2e.py` with a Docker Compose fixture (`arro-server` + `arro-nlp-frontend`) is needed before the system goes to production.

### Gap 6 — No deployment configuration

There is no `Dockerfile`, `docker-compose.yml`, or `k8s/` manifest for `arro-nlp-frontend`. Running the full stack (`arro-server` + `arro-nlp-frontend` + a mounted Zarr volume) requires manual setup. A `docker-compose.yml` that wires the two services together and sets all required env vars is the minimum needed for a reproducible deployment.

---

## Recommended next steps (ordered)

1. **Implement `arro-cve-search#9`** (CVE ingest harness) using the v2 `/ingest` API with `dataset_id: "cve/embeddings"`. Decide on data source (Gap 2) first.
2. **Implement issue #8** (`GET/DELETE /documents/{doc_id}`) — needed for CVE withdrawal handling.
3. **Implement `arro-cve-search#10`** (query CLI or web UI) — makes the product usable.
4. **Write `docker-compose.yml`** for `arro-server` + `arro-nlp-frontend` — prerequisite for integration testing and deployment.
5. **Write `tests/integration/test_e2e.py`** — end-to-end smoke test with real arro-server.

---

# Codebase Analysis

*As of PR #20 (merged). Every section covers one architectural layer: design rationale, the specific risks introduced by that design, and the correct future path.*

---

## Layer 1 — Configuration (`config.py`)

### Design

`pydantic-settings` loads from env vars and `.env` with priority: env > `.env` > field defaults. Validators enforce `embed_backend ∈ {"local","openai"}`, `embed_scale_factor > 0`, `arro_server_search_tau ∈ [0.0, 1.0]`, and `OPENAI_API_KEY` presence when `backend=openai`. All service coordinates (`arro_server_url`, `store_db_path`, `host`, `port`) are env-configurable with sensible defaults.

### Risks

- **Silent mis-configuration at scale.** `arro_server_root_label` has a string default (`"main"`) with no validation that it matches what arro-server has registered. A wrong `dataset_id` in a request causes a silent 404 from `dataset_metadata()` (treated as "new dataset") and a fresh Zarr upload that replaces the existing index.
- **`ingest_batch_size` is module-level.** `EMBED_CHUNK = settings.ingest_batch_size` is evaluated at import time in `ingest.py`. Changing the env var at runtime has no effect without a server restart.
- **No config schema export.** There is no `/config` or `/settings` endpoint. Operators cannot introspect active config without reading env/logs.

### Future work

- Add a `GET /admin/config` endpoint (read-only, redact secrets) for operator visibility.
- Add a startup validator that pings arro-server and logs a warning if the `root_label` does not match any registered root.
- Document `EMBED_CHUNK` import-time evaluation in the field docstring.

---

## Layer 2 — Embedder (`embedder.py`)

### Design

Two backends: `local` (SentenceTransformer, default `all-MiniLM-L6-v2`, 384-dim) and `openai`. Vectors are **never L2-normalised** — arro-server expects raw scaled vectors and manages its own spectral distance. `scale_factor` is applied uniformly post-encoding. Fails loud at construction if `model_path` is set but does not exist on disk.

### Risks

- **`dim` property lies for OpenAI backend.** `dim` returns `384` hardcoded. If the operator uses `text-embedding-3-large` (3072-dim), the Zarr array is created with `shape=(N, 384)` — silent truncation.
- **`encode_batch` is synchronous and CPU-bound, called inside the async event loop.** Blocks the event loop for the full embedding duration. Under concurrent load this blocks search and health check responses.
- **No text length validation.** `IngestItem.text` has `min_length=1` but no `max_length`. SentenceTransformer silently truncates at the model token limit; the stored text is the full original but the embedding represents only the prefix.
- **`scale_factor` interaction with arro-server tau.** Applying `scale_factor != 1.0` shifts vector norms and can invalidate tau thresholds calibrated on unit-norm vectors.

### Future work

- Fix `dim` for OpenAI: maintain a `{model: dim}` lookup table.
- Offload `encode_batch` to a thread pool: `await asyncio.get_event_loop().run_in_executor(None, embedder.encode_batch, texts)`.
- Add `max_length: int` field to `IngestItem` (default 2048 chars) with a validator.
- Log a `WARNING` when `scale_factor != 1.0` at embedder construction time.

---

## Layer 3 — Document Store (`store.py`)

### Design

Stdlib `sqlite3` only — no ORM. WAL journal mode. `upsert_batch` is atomic. `row_index = MAX(row_index) + 1` scoped to `dataset_id`, not `COUNT(*)`. `delete_by_id` is a soft delete. As of PR #18: `PRIMARY KEY (dataset_id, row_index)`, fully multi-dataset. Schema versioned at `_SCHEMA_VERSION = 2`. v1 databases detected at startup with a `RuntimeError` pointing at the migration command.

### Risks

- **`asyncio.Lock` does not protect multi-worker or multi-replica deployments.** With `uvicorn --workers 4`, four processes share the same SQLite file but each has its own lock. Two workers can compute the same `start_row` simultaneously and corrupt the index. Data loss, no error.
- **Re-ingest with existing `doc_id` creates a ghost vector.** `INSERT OR REPLACE` assigns a new `row_index` on re-ingest. The old Zarr slot is orphaned until the next `build_index` run. `search.py` skips ghost rows gracefully but the degradation window is unbounded.
- **`get_all_vectors()` reads the full dataset matrix into RAM on every ingest.** At 100K docs × 384-dim float64: ~300MB per ingest call. Peak RSS ≈ 2× matrix size.
- **`check_same_thread=False` with a future `run_in_executor` is a trap.** The `asyncio.Lock` is not a threading lock. If any future code moves SQLite calls to a thread pool, the lock no longer protects them.

### Future work

- Wrap `next_row_index` + `upsert_batch` in `BEGIN EXCLUSIVE` SQLite transaction for multi-worker safety.
- Cap `get_all_vectors()` with a configurable `max_ingest_rows` setting.
- Add `POST /admin/reindex` for ghost vector cleanup.
- Document the `asyncio.Lock` + `check_same_thread=False` invariant in the store docstring.

---

## Layer 4 — ArroClient (`arro_client.py`)

### Design

Thin async HTTP wrapper using `httpx.AsyncClient`. As of PR #18: `dataset_id` is a per-call argument on all 5 methods, not a constructor singleton. Raises `ArroServerError` on any non-2xx or network failure.

### Risks

- **No retry logic.** A transient 503 during `upload_commit` causes a 502 to the caller and leaves the index stale. The self-healing mechanism (next ingest rewrites Zarr) only works if another ingest happens.
- **`timeout=30.0` for all methods.** `build_index` on a large dataset can take minutes. The scalar timeout will abort a legitimate long-running index build with a `ReadTimeout` → 502.
- **No connection pooling configuration.** Pool size, keepalive, and TLS settings are httpx defaults.

### Future work

- Add `tenacity`-based retry with exponential backoff for `upload_commit` and `build_index`.
- Use a separate, longer timeout for `build_index` (e.g. `timeout=600.0`) or implement an async poll loop.
- Consider Arrow IPC or msgpack for vector payloads once arro-server supports it.

---

## Layer 5 — Ingest endpoint (`ingest.py`)

### Design

Two-phase pipeline: embed outside the lock (CPU-bound, no shared state), then acquire the per-dataset lock for `next_row_index → upsert_batch → Zarr rewrite`. SQLite is written first — 502 from arro-server leaves the document locally but the index stale; the next successful ingest self-heals.

As of PR #20: `dataset_id` is required in the request body. The lock is a per-dataset `asyncio.Lock` from `app.state.ingest_locks`, created lazily via `_get_dataset_lock()`. Concurrent requests to different datasets now run in parallel.

### Risks

- **Full Zarr rewrite on every ingest is O(N) in dataset size.** Every ingest — even a single document — reads all N vectors, writes a new Zarr array, and calls `upload_commit`. At 100K documents: ~300MB read + ~300MB write per call.
- **Lock held for the full Zarr write.** The lock is held from `next_row_index()` through `build_index()`. For a large dataset this serialises all ingest requests for tens of seconds.
- **No batch size cap.** `IngestRequest.documents` has `min_length=1` but no `max_length`. OOM is possible on small deployments.

### Future work

- Add `max_length=500` (configurable) to `IngestRequest.documents`.
- Investigate append-only Zarr writes to eliminate O(N) rewrite.
- Release lock after SQLite write; move Zarr rewrite outside the lock scope.

---

## Layer 6 — Search endpoint (`search.py`)

### Design

Pure read path — no lock. As of PR #18: `dataset_id` is required in the request body, forwarded to `arro_client.search()` and `store.get_by_row()`. Ghost rows are silently skipped. Ranks are resequenced `1..N` after any skips.

### Risks

- **`encode_batch([query])[0]` is synchronous and blocks the event loop.** ~5–50ms per query depending on hardware. Under high concurrency this is a meaningful bottleneck.
- **No result deduplication.** If arro-server returns the same `row_index` twice, the response contains duplicate `doc_id` values.
- **`query_time_ms` is a single scalar.** Callers cannot distinguish slow embed from slow arro-server from slow SQLite hydration.
- **No caching.** Repeated identical queries re-embed and re-call arro-server every time.

### Future work

- Offload `encode_batch` to `run_in_executor`.
- Add deduplication: after hydration, deduplicate by `doc_id`, keep highest-score hit.
- Add structured timing: `{embed_ms, search_ms, hydrate_ms}` behind a `debug=true` query param.
- Add optional LRU cache for `(query_hash, top_k, tau)` → results with configurable TTL.

---

## Scalability profile

| Dimension | Current limit | Root cause | Correct fix |
|---|---|---|---|
| Dataset size | ~50K docs before ingest becomes slow | O(N) Zarr rewrite per ingest | Append-only Zarr writes |
| Ingest throughput (same dataset) | 1 concurrent (lock held for full Zarr write) | `asyncio.Lock` scope too wide | Release lock after SQLite write; move Zarr rewrite outside lock |
| Ingest throughput (different datasets) | N concurrent (per-dataset lock) | ✅ Fixed in PR #20 | — |
| Search concurrency | Degrades under load | Sync embed blocks event loop | `run_in_executor` for embed |
| Multi-worker (`--workers N`) | **Broken** — row index corruption | `asyncio.Lock` is process-local | SQLite `BEGIN EXCLUSIVE` or ingest queue |
| Multi-replica (K8s) | **Broken** — same as multi-worker | Same root cause | Distributed advisory lock or message queue |
| Re-ingest existing `doc_id` | Ghost vectors accumulate | Zarr has no point-delete | `409 Conflict` or append + soft-delete at search time |
| Large single document | Silent truncation at model token limit | No `max_length` on `IngestItem.text` | Add `max_length` field + validator |
| Large batch | OOM risk | No `max_length` on `IngestRequest.documents` | Add `max_length=500` |

---

## Known open issues

| Issue | Title | Priority |
|---|---|---|
| [#8](https://github.com/Genefold/arro-nlp-frontend/issues/8) | `GET/DELETE /documents/{doc_id}` | Medium |
| [#12](https://github.com/Genefold/arro-nlp-frontend/issues/12) | Re-ingest creates ghost vectors in arro-server | High |
| [#13](https://github.com/Genefold/arro-nlp-frontend/issues/13) | `asyncio.Lock` does not protect multi-worker or multi-replica | High |
| [#14](https://github.com/Genefold/arro-nlp-frontend/issues/14) | `vectors.tolist()` + JSON does not scale for large batches | Medium |
| [#15](https://github.com/Genefold/arro-nlp-frontend/issues/15) | `app.state` untyped, no mypy verification | Low |

---

## Phased delivery roadmap

### Phase 1 — Complete the core feature set (now)

1. Implement `arro-cve-search#9`: CVE ingest harness targeting v2 API (`dataset_id: "cve/embeddings"`).
2. Implement issue #8: `GET/DELETE /documents/{doc_id}`.

### Phase 2 — Make it usable as a product

1. Implement `arro-cve-search#10` (or equivalent): query CLI or minimal web UI.
2. Write `docker-compose.yml` wiring `arro-server` + `arro-nlp-frontend` with shared Zarr volume and all required env vars.
3. Write `tests/integration/test_e2e.py`: spin up real arro-server via Docker Compose, ingest 10 CVEs, assert search returns correct results.

### Phase 3 — Correctness & safety

These fix silent data corruption, not features:

1. **Fix re-ingest ghost vectors** (issue #12): return `409 Conflict` on re-ingest of existing `doc_id` until append+rebuild is implemented.
2. **Fix multi-worker row index race** (issue #13): wrap `next_row_index` + `upsert_batch` in `BEGIN EXCLUSIVE` SQLite transaction, or enforce `--workers 1` at startup.
3. **Fix OpenAI `dim` property**: `{model: dim}` lookup table, not hardcoded 384.
4. **Add text length validation**: `max_length` on `IngestItem.text`.

### Phase 4 — Performance (when dataset > 10K docs)

1. Offload embed to thread pool (`run_in_executor` for both ingest and search).
2. Release lock after SQLite write; move Zarr rewrite outside the lock scope.
3. Cap ingest batch size: `max_length=500` on `IngestRequest.documents`.
4. Add `POST /admin/reindex`: full rebuild for post-downtime recovery and ghost vector cleanup.

### Phase 5 — Scale-out (when single-process is insufficient)

1. Append-only Zarr writes: eliminate O(N) rewrite per ingest; requires arro-server API extension.
2. Multi-worker safety: `BEGIN EXCLUSIVE` (same-host) or ARQ + Redis queue (multi-host).
3. Search LRU cache: cache `(query_hash, top_k, tau)` → results with configurable TTL.
4. Arrow IPC vector transport: replace JSON with Arrow IPC for ingest and search payloads..




Ho tutto il quadro. Ecco lo storyline completo basato solo su issue e codice reale.

***

## Storyline di implementazione

### Sprint 0 — Fondamenta lato server
**Parallelo, zero dipendenze**

Le tre issue di arro-server non dipendono da nulla di nuovo e possono partire subito tutte e tre in parallelo su branch dedicati.

**[arro-server #34](https://github.com/Genefold/arro-server/issues/34)** — `POST /vectors/append`
`zarr_fs.py` apre oggi solo in `mode="r"`. La prima modifica è aggiungere `append_vectors()` che apre in `mode="r+"`, fa `resize()` e scrive il slice `[old_n:]`. Poi si aggiunge la route in `routes.py` e si chiama `registry.register_dataset()` per aggiornare la shape in cache. I test verificano che la shape cresca e che `start_row` nel response sia corretto.

**[arro-server #35](https://github.com/Genefold/arro-server/issues/35)** — `POST /vectors/overwrite`
Stessa meccanica di #34 ma con write a indici arbitrari `arr[row_index] = vec`. Valida i bounds prima di qualsiasi write per evitare corruzione parziale. Questo issue, una volta merged, chiude anche la root cause di arro-nlp-frontend #12 (silent stale vector bug). 

**[arro-server #36](https://github.com/Genefold/arro-server/issues/36)** — `GET /vectors/count`
Endpoint lightweight: legge `shape` dal `DatasetSummary` in cache (O(1), nessuna I/O su Zarr). Serve come consistency guard prima che arro-nlp-frontend faccia append. 

***

### Sprint 1 — Search: la feature più urgente
**Dipende da: arro-server già funzionante (esistente)**

**[arro-nlp-frontend #22](https://github.com/Genefold/arro-nlp-frontend/issues/22)** — `POST /search`

Questo si può iniziare **in parallelo con Sprint 0** perché l'endpoint di search su arro-server esiste già — bisogna solo verificarne la firma in `routes.py` prima di implementare `ArroClient.search()`. Il lavoro è:

1. Aggiungere `DocumentStore.get_by_row_indices()` in `store.py` — una singola query `WHERE row_index IN (...)`
2. Aggiungere `ArroClient.search()` in `arro_client.py`
3. Scrivere `router/search.py` con `POST /search`
4. Registrarlo in `main.py`

Il ghost-row handling (doc_id non in SQLite → skip silenzioso + WARNING) è già specificato nella issue. 

La search è la feature più visibile per arro-cve-search e non blocca nulla di Sprint 0 — va in parallelo.

***

### Sprint 2 — Ingest incrementale
**Dipende da: Sprint 0 completato (#34, #35, #36)**

**[arro-nlp-frontend #21](https://github.com/Genefold/arro-nlp-frontend/issues/21)** — `POST /ingest?incremental=true`

Solo ora che i tre endpoint di arro-server esistono si può implementare il branch incrementale in `ingest.py`. Il lavoro si divide in due parti separate:

**Parte A — client methods** (in `arro_client.py`): aggiungere `append_vectors()`, `overwrite_vectors()`, `get_vector_count()`. Sono wrapper HTTP puri, testabili con mock indipendentemente dal resto.

**Parte B — logica di branching** (in `ingest.py`): aggiungere `incremental: bool = False` a `IngestRequest`, implementare il diff (`new` / `changed` / `metadata_only`), chiamare append o overwrite dentro il lock esistente (`_get_dataset_lock`), e chiamare `build_index` **una sola volta** alla fine del batch — non dentro il loop. 

Il consistency guard (confronto `get_vector_count()` vs `store.next_row_index()`) va implementato come prima cosa dentro il lock, prima di qualsiasi write.

***

### Sprint 3 — CRUD documenti e qualità
**Dipende da: Sprint 1 completato**

**[arro-nlp-frontend #8](https://github.com/Genefold/arro-nlp-frontend/issues/8)** — `GET /documents/{doc_id}` e `DELETE /documents/{doc_id}`

Con la search funzionante, il CRUD sui documenti chiude il loop: un utente può cercare una CVE, ottenerla per ID, e rimuoverla. La delete è soft: rimuove solo da SQLite, non da Zarr — il ghost handling nella search già copre questo caso. 

**[arro-nlp-frontend #15](https://github.com/Genefold/arro-nlp-frontend/issues/15)** — Typed `AppState`

Da fare nello stesso sprint perché ogni nuovo endpoint (`/search`, `/documents`) accede a `app.state`. Aggiungere `AppState` dataclass e `get_state(req)` ora, prima che lo stato si allarghi ulteriormente. 

***

### Backlog — Non bloccante per produzione

| Issue | Quando |
|---|---|
| [arro-nlp-frontend #13](https://github.com/Genefold/arro-nlp-frontend/issues/13) — multi-worker lock | Solo se si scala oltre un processo |
| [arro-nlp-frontend #14](https://github.com/Genefold/arro-nlp-frontend/issues/14) — binary transport | Quando i batch superano 500 doc |
| [arro-server #26](https://github.com/Genefold/arro-server/issues/26) — async build_index | Quando ci sono build concorrenti |
| [arro-server #16](https://github.com/Genefold/arro-server/issues/16) — only 2D arrays | Già in progress, non blocca pipeline CVE |

***

### Grafico delle dipendenze

```
arro-server
  #34 append ──┐
  #35 overwrite─┼──► nlp-frontend #21 (ingest incrementale)
  #36 count   ──┘

arro-server (search esistente) ──► nlp-frontend #22 (search)
                                         │
                                         ▼
                                   nlp-frontend #8 (CRUD)
                                   nlp-frontend #15 (typed state)
```

Il **minimum viable path** per portare arro-cve-search in produzione è: `#34 + #35 + #36` → `#21 + #22` in parallelo → deploy. Il CRUD (#8) e il typed state (#15) migliorano la qualità ma non bloccano.


