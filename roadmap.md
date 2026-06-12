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








