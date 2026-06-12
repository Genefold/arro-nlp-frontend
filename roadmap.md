All 6 issues are open. Here's a summary of everything created.

## Issues opened

**On [`arro-nlp-frontend`](https://github.com/Genefold/arro-nlp-frontend)**

| # | Title | Depends on | Blocks |
|---|---|---|---|
| [4](https://github.com/Genefold/arro-nlp-frontend/issues/4)  | `STORE_DB_PATH` in Settings | — | #5, #6, #7 |
| [5](https://github.com/Genefold/arro-nlp-frontend/issues/5)  | `store.py` — generic `DocumentStore` | #4 | #6, #7, #8 |
| [6](https://github.com/Genefold/arro-nlp-frontend/issues/6)  | `POST /ingest` | #4, #5 | #7, cve-search#9 |
| [7](https://github.com/Genefold/arro-nlp-frontend/issues/7)  | `POST /search` | #4, #5, #6 | webapp |
| [8](https://github.com/Genefold/arro-nlp-frontend/issues/8)  | `GET/DELETE /documents/{doc_id}` | #5, #6 | leaf |

**On [`arro-cve-search`](https://github.com/Genefold/arro-cve-search)**

| # | Title | Depends on |
|---|---|---|
| [9](https://github.com/Genefold/arro-cve-search/issues/9)  | `harness/ingest.py` — CVE HTTP ingest | arro-nlp-frontend#6 |

## Recommended implementation order

#4 → #5 → #6 → #7 → #8, then cve-search #9 last. Each is independently mergeable once its dependencies land. Start with #4 and #5 — they're pure Python with no HTTP mocking needed and will take under an hour each.


Ho tutto. Analisi completa.

***

## Stato della codebase — PR #11

[Certain] Il quadro generale è solido per un sistema single-process. I contratti tra i layer sono scritti bene e i test coprono i casi critici. Detto questo, ci sono criticità strutturali che emergeranno presto se il sistema cresce.

***

## Cosa funziona bene

**Layer separation è corretta.** [Certain] `ArroClient`  non contiene logica di business — fa solo HTTP e traduce errori. `DocumentStore`  non sa nulla di vettori né di arro-server. `ingest.py`  orchestra i due senza accoppiamento diretto. Questo è il pattern giusto e rende i componenti sostituibili.

**Il contratto del row index è blindato.** [Certain] Il commento al DDL in `store.py`  è eccezionalmente chiaro sui 4 invarianti (immutabilità, soft delete, lock ownership, rebuild rule). `next_row_index()` usa `MAX+1` invece di `COUNT` — la scelta è documentata con un esempio concreto. Il test `test_ingest_start_row_uses_max_not_count` verifica proprio questo caso .

**La pipeline di ingest è atomica nel posto giusto.** [Certain] Embed avviene fuori dal lock (CPU pura, non tocca stato condiviso), il lock acquisisce `start_row`, pusha su arro-server, scrive nel store — in quest'ordine. Se arro-server fallisce, il store non viene mai scritto . Questo è l'unico ordine corretto.

**Test isolation è ben progettata.** [Certain] Patchare `lifespan` con `_noop_lifespan` e iniettare manualmente `app.state` è l'approccio più robusto disponibile senza introdurre dependency injection. La fixture `ingest_client` è session-scoped per l'embedder (evita re-download del modello) e function-scoped per lo store (ogni test parte da DB pulito) .

**WAL mode su SQLite.** [Certain] `PRAGMA journal_mode=WAL`  consente letture concorrenti senza bloccare le scritture. Scelta corretta per un server FastAPI async dove le letture (`get_by_id` per il check created/updated) avvengono fuori dal lock.

***

## Criticità reali

### 🔴 `upsert_batch` non è idempotente in caso di re-ingest con doc_id esistente

[Certain] `INSERT OR REPLACE` in SQLite elimina la riga esistente e ne inserisce una nuova. Questo significa che se re-ingesti `doc_id="CVE-1"` che aveva `row_index=0`, la nuova riga riceve `row_index = next_row_index()` al momento del re-ingest — che potrebbe essere, ad esempio, `row_index=42`. Il vettore in arro-server all'indice 0 punta ora a un documento eliminato (ghost), e il vettore all'indice 42 contiene l'embedding del nuovo testo. **Il vecchio vettore a row_index=0 non viene aggiornato in arro-server.** La ricerca può restituire il documento aggiornato via row_index=42, ma può anche restituire ghost hits via row_index=0 con nessun documento corrispondente nel store. Questo è un bug silenzioso già documentato nel commento al `delete_by_id` ("search endpoint must skip missing rows") — ma non è ancora gestito né testato.

### 🔴 `app_client` in `conftest.py` è session-scoped ma `DocumentStore` usa un `TemporaryDirectory`

[Certain] La fixture `app_client`  crea un `TemporaryDirectory` dentro un context manager session-scoped. Il problema: `app_client` ha `scope="session"` ma `tmp_path` non è disponibile a scope session (è function-scoped). La fixture usa `tempfile.TemporaryDirectory()` direttamente per aggirarlo — tecnicamente corretto, ma il `DocumentStore` creato dentro quella fixture viene chiuso solo quando `app_client` esce dallo scope di sessione. Se il `TemporaryDirectory` viene garbage-collected prima (improbabile ma possibile su Windows con file lock), il test può fallire in modo non deterministico.

### 🟡 `ingest.py` accede a `app.state` via attributi dinamici senza type safety

[Likely] `req.app.state.embedder`, `req.app.state.store`, `req.app.state.arro_client` sono accessi a `starlette.datastructures.State` — un bag dinamico senza type hints . mypy non può verificare che questi attributi esistano né che abbiano il tipo corretto. Se il lifespan non viene eseguito (bug in un test, o startup fallito), si ottiene `AttributeError` a runtime invece di un errore a compile time. Soluzione standard: creare un `TypedDict` o un `dataclass` per `app.state` e castare.

### 🟡 La serializzazione JSON dei vettori in `push_vectors` non scala

[Certain] `vectors.tolist()` converte un array NumPy float64 di shape `(N, 384)` in una lista Python di liste di Python floats, che viene poi serializzata come JSON . Per N=100, dim=384: ~30.720 valori float64 → JSON string di ~250KB. Per N=1000 → ~2.5MB. Il bottleneck non è la rete ma la serializzazione/deserializzazione Python ↔ JSON. Se arro-server supporta un formato binario (Arrow IPC, NumPy raw, msgpack), questo va cambiato prima che diventi un problema.

### 🟡 Il lock è su `app.state` ma `DocumentStore` usa `check_same_thread=False`

[Likely] `check_same_thread=False` è necessario perché FastAPI async esegue su un singolo thread con un event loop, ma sqlite3 di default blocca l'accesso da thread diversi . Il lock asyncio serializza le scritture al livello dell'event loop — ma se in futuro qualcuno esegue operazioni sul `DocumentStore` in un `run_in_executor` (thread pool), il lock asyncio non protegge più nulla perché `asyncio.Lock` non è thread-safe tra thread OS diversi.

***

## Scalabilità e generalizzabilità

[Certain] Il sistema come progettato scala **verticalmente** ma non **orizzontalmente**:

| Dimensione | Stato attuale | Limite |
|---|---|---|
| Batch size | Funziona fino a ~500 docs | JSON serialization diventa collo di bottiglia oltre |
| Concorrenza intra-worker | Protetta dall'asyncio.Lock | Nessun problema |
| Multi-worker (`--workers N`) | **NON protetto** | Due worker possono assegnare lo stesso `start_row` |
| Multi-replica (K8s) | **NON protetto** | Stessa race condition cross-process |
| Rebuild arro-server | Store diventa corrotto silenziosamente | Nessun meccanismo di recovery |
| Testo lungo | Nessun limite su `text` in `IngestItem` | Embed di testi da 100K token può OOM il modello |

[Guessing!] L'architettura attuale suggerisce un sistema pensato per deployment single-instance (una replica, un worker). Se il roadmap prevede scaling orizzontale, il `asyncio.Lock` va sostituito con un advisory lock SQLite (`BEGIN EXCLUSIVE`) o con una coda serializzata (Celery, ARQ, o anche un semplice `asyncio.Queue` con un consumer singolo).

Il layer `ArroClient` è ben astratto e può essere sostituito senza toccare `ingest.py` — questo è il punto di estensione corretto per aggiungere retry logic, circuit breaker, o un trasporto binario.

4 issue create. Ecco il riepilogo.

***

## Issue create

### (https://github.com/Genefold/arro-nlp-frontend/issues/12) — Re-ingest crea ghost vectors in arro-server

Il bug silenzioso più insidioso. Re-ingestare un `doc_id` esistente aggiunge un nuovo vettore ad arro-server senza rimuovere il vecchio (Zarr non supporta point-delete). Il vecchio vettore rimane come ghost — non crash, solo degradazione silenziosa della qualità di ricerca nel tempo. L'issue documenta 4 opzioni, la raccomandazione a breve termine è restituire `409 Conflict` sul re-ingest (Option A) e aggiungere un `POST /admin/reindex` in futuro.

### (https://github.com/Genefold/arro-nlp-frontend/issues/13) — `asyncio.Lock` non protegge multi-worker né multi-replica

Il vincolo single-process è già documentato nel docstring dell'endpoint, ma non c'è nulla che lo **enforci** né che avvisi l'operatore. L'issue documenta il rischio concreto (`--workers 4` corrumpe l'indice silenziosamente) e le opzioni di scaling futuro: `BEGIN EXCLUSIVE` SQLite per multi-worker sullo stesso host, ARQ+Redis per multi-host.

### (https://github.com/Genefold/arro-nlp-frontend/issues/14) — `vectors.tolist()` + JSON non scala per batch grandi

A 1000 documenti il payload JSON è ~15MB contro ~3MB dei dati raw NumPy. L'issue include la tabella dimensioni reali, documenta il path verso Arrow IPC (naturale data l'architettura Arrow-centrica del progetto), e raccomanda a breve termine un `max_batch_size=200` per cappare il problema senza richiedere cambi all'API di arro-server.

### (https://github.com/Genefold/arro-nlp-frontend/issues/15) — `app.state` non tipizzato, nessuna verifica mypy

Tre endpoint accedono ad attributi dinamici su `app.state` senza type hints — mypy non può verificarne l'esistenza né il tipo. L'issue propone due soluzioni concrete: un `AppState` dataclass con helper `get_state(req)`, oppure il pattern idiomatico FastAPI con `Depends`. Diventa più rilevante man mano che crescono gli endpoint che leggono da `app.state`.