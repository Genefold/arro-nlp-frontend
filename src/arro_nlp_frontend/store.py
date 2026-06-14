"""Generic SQLite document store: maps arro-server row indices to documents.

Design rules (NON NEGOTIABLE):
  - No ORM. stdlib sqlite3 only.
  - WAL journal mode is enabled on every connection for safe concurrent reads.
  - upsert_batch wraps all inserts in a single transaction (atomic).
  - ingested_at is always set server-side (UTC). Never trust caller-supplied value.
  - metadata is persisted as a JSON string and deserialised on every read.
  - db_path parent directories are created automatically (mkdir parents=True).
  - Thread-safe: check_same_thread=False with one connection per instance.
  - Empty upsert_batch raises ValueError -- silent no-ops hide bugs.

Row index invariant (NON NEGOTIABLE):
  row_index is the ONLY join key between the vector index (arro-server/Parquet)
  and this store. The following rules protect it:

  1. start_row must be derived inside the same write lock that protects the
     arro-server push. It must NEVER be computed as store.count() before the
     push -- two concurrent callers would compute the same start_row and corrupt
     the index. The ingest endpoint (issue #6) owns this lock.

  2. row_index is IMMUTABLE after insertion. INSERT OR REPLACE is safe only
     because re-ingesting the same doc_id replaces the same row_index.

  3. delete_by_id is a SOFT DELETE. The vector at that row_index remains in
     arro-server forever (Zarr/Parquet do not support row deletion). The search
     endpoint must skip row_index values not found in this store.

  4. If arro-server is ever rebuilt (re-index), this store MUST be rebuilt from
     scratch. A stale store with mismatched row indices will silently return
     wrong documents with no error.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = 2

_DDL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    dataset_id  TEXT    NOT NULL,
    row_index   INTEGER NOT NULL,
    doc_id      TEXT    NOT NULL,
    text        TEXT    NOT NULL,
    vector      BLOB    NOT NULL,
    metadata    TEXT    NOT NULL DEFAULT '{}',
    ingested_at TEXT    NOT NULL,
    PRIMARY KEY (dataset_id, row_index),
    UNIQUE      (dataset_id, doc_id)
);
CREATE INDEX IF NOT EXISTS idx_dataset_doc_id ON documents(dataset_id, doc_id);
"""


@dataclass
class Document:
    """A single document stored in the document store.

    Attributes
    ----------
    row_index:   0-based Zarr row in arro-server. Immutable after insertion.
    doc_id:      Application-defined unique identifier (e.g. "CVE-2024-1234").
    text:        The original text that was embedded.
    metadata:    Arbitrary application fields. Stored as JSON, round-trips losslessly.
    ingested_at: UTC timestamp set by the store on upsert. None before first write.
    """

    row_index: int
    doc_id: str
    text: str
    metadata: dict
    ingested_at: datetime | None


class DocumentStore:
    """SQLite-backed store mapping row_index to Document.

    Parameters
    ----------
    db_path: Path to the SQLite file. Created (with parent dirs) if absent.

    Usage
    -----
    Prefer the context manager form::

        with DocumentStore(Path("./data/documents.sqlite")) as store:
            store.upsert_batch(0, docs)
            doc = store.get_by_row(0)
    """

    def __init__(self, db_path: Path) -> None:
        """Open (or create) the SQLite database at db_path.

        Creates parent directories. Enables WAL journal mode.
        Raises FileNotFoundError if db_path exists and is a directory.
        """
        self.db_path = db_path
        self._conn: sqlite3.Connection | None = None
        self._open()

    def _open(self) -> None:
        """Open the connection, create parent dirs, apply schema, run migrations."""
        if self._conn is not None:
            return

        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        if self.db_path.is_dir():
            raise FileNotFoundError(f"db_path is a directory, not a file: {self.db_path}")

        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.commit()
        self._apply_schema()

    def _apply_schema(self) -> None:
        """Apply DDL and run any pending migrations."""
        assert self._conn is not None

        has_sv = (
            self._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
            ).fetchone()
            is not None
        )

        if not has_sv:
            has_docs = (
                self._conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='documents'"
                ).fetchone()
                is not None
            )
            if has_docs:
                raise RuntimeError(
                    "Detected a version-1 DocumentStore schema (no dataset_id column). "
                    "Run: python -m arro_nlp_frontend.migrate --db-path <path> "
                    "--dataset-id <id> to migrate your existing data before starting "
                    "the server."
                )

        self._conn.executescript(_DDL)
        self._conn.commit()

        cursor = self._conn.execute("SELECT version FROM schema_version LIMIT 1")
        row = cursor.fetchone()
        if row is None:
            self._conn.execute(
                "INSERT INTO schema_version (version) VALUES (?)", (_SCHEMA_VERSION,)
            )
            self._conn.commit()
        elif int(row[0]) < _SCHEMA_VERSION:
            raise RuntimeError(
                f"DocumentStore schema version {row[0]} is older than required "
                f"{_SCHEMA_VERSION}. Run the migration script."
            )

    def upsert_batch(
        self,
        dataset_id: str,
        start_row: int,
        docs: list[Document],
        vectors: np.ndarray,
    ) -> list[Literal["created", "updated"]]:
        """Atomically insert or replace documents starting at start_row.

        Row indices are assigned as start_row, start_row+1, ..., start_row+N-1.
        Uses INSERT OR REPLACE so re-running ingest is idempotent.
        Sets ingested_at to current UTC time for every row in the batch.

        Parameters
        ----------
        dataset_id: Dataset identifier (e.g. "cve/embeddings").
        start_row:  First row index to assign. Must be >= 0.
        docs:       Non-empty list of Document objects.
        vectors:    Float64 array of shape (N, dim). Must match len(docs).

        Returns
        -------
        List of "created" or "updated" per document.

        Raises
        ------
        ValueError: If docs is empty or vectors shape mismatch.
        """
        if not docs:
            raise ValueError("docs cannot be empty")
        if start_row < 0:
            raise ValueError("start_row must be >= 0")
        if len(docs) != vectors.shape[0]:
            raise ValueError(f"vectors shape {vectors.shape} does not match doc count {len(docs)}")

        now = datetime.now(UTC).isoformat()
        batch_data = [
            (
                dataset_id,
                start_row + i,
                doc.doc_id,
                doc.text,
                vectors[i].tobytes(),
                json.dumps(doc.metadata),
                now,
            )
            for i, doc in enumerate(docs)
        ]

        statuses: list[Literal["created", "updated"]] = []
        for doc in docs:
            statuses.append("updated" if self.get_by_id(dataset_id, doc.doc_id) else "created")

        assert self._conn is not None
        with self._conn:
            self._conn.executemany(
                """
                INSERT OR REPLACE INTO documents
                    (dataset_id, row_index, doc_id, text, vector, metadata, ingested_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                batch_data,
            )

        logger.info(
            "[store] upserted %d documents dataset=%s start_row=%d",
            len(docs),
            dataset_id,
            start_row,
        )

        return statuses

    def get_by_row(self, dataset_id: str, row_index: int) -> Document | None:
        """Return the document at row_index for the given dataset, or None if not found."""
        assert self._conn is not None
        cursor = self._conn.execute(
            "SELECT row_index, doc_id, text, metadata, ingested_at"
            " FROM documents WHERE dataset_id = ? AND row_index = ?",
            (dataset_id, row_index),
        )
        row = cursor.fetchone()
        return self._row_to_document(row) if row else None

    def get_by_id(self, dataset_id: str, doc_id: str) -> Document | None:
        """Return the document with the given doc_id for the dataset, or None if not found."""
        assert self._conn is not None
        cursor = self._conn.execute(
            "SELECT row_index, doc_id, text, metadata, ingested_at"
            " FROM documents WHERE dataset_id = ? AND doc_id = ?",
            (dataset_id, doc_id),
        )
        row = cursor.fetchone()
        return self._row_to_document(row) if row else None

    def delete_by_id(self, dataset_id: str, doc_id: str) -> bool:
        """Delete the document with doc_id from the given dataset.

        Note: This is a SOFT DELETE at the retrieval layer. The corresponding
        vector row in arro-server is NOT removed (Zarr arrays do not support
        row deletion). If a deleted document's row_index is returned by
        arro-server, the search endpoint must handle this gracefully by
        skipping the missing row (see issue #7).

        Returns True if a row was deleted, False if doc_id was not found.
        """
        assert self._conn is not None
        cursor = self._conn.execute(
            "DELETE FROM documents WHERE dataset_id = ? AND doc_id = ?",
            (dataset_id, doc_id),
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def next_row_index(self, dataset_id: str) -> int:
        """Return the next available row index as MAX(row_index) + 1 for the dataset.

        Uses MAX() rather than COUNT() to be safe against soft-deleted rows:
        if rows 0, 1, 2 exist and row 1 is deleted, COUNT()=2 but the next
        valid index is 3, not 2. Using COUNT() would silently overwrite row 2.

        Returns 0 if the dataset has no rows.
        """
        assert self._conn is not None
        cursor = self._conn.execute(
            "SELECT COALESCE(MAX(row_index) + 1, 0) FROM documents WHERE dataset_id = ?",
            (dataset_id,),
        )
        return int(cursor.fetchone()[0])

    def count(self, dataset_id: str | None = None) -> int:
        """Return the total number of documents.

        If dataset_id is None, returns the count across all datasets.
        """
        assert self._conn is not None
        if dataset_id is None:
            cursor = self._conn.execute("SELECT COUNT(*) FROM documents")
        else:
            cursor = self._conn.execute(
                "SELECT COUNT(*) FROM documents WHERE dataset_id = ?", (dataset_id,)
            )
        return int(cursor.fetchone()[0])

    def get_all_vectors(self, dataset_id: str) -> np.ndarray:
        """Return all vectors for a dataset as a (N, dim) float64 array ordered by row_index.

        Used to reconstruct the full Zarr dataset for upload to arro-server.
        Returns empty array of shape (0,) if the dataset has no rows.
        """
        assert self._conn is not None
        cursor = self._conn.execute(
            "SELECT vector FROM documents WHERE dataset_id = ? ORDER BY row_index",
            (dataset_id,),
        )
        rows = cursor.fetchall()
        if not rows:
            return np.array([], dtype=np.float64)
        vectors = [np.frombuffer(row[0], dtype=np.float64) for row in rows]
        return np.stack(vectors)

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> DocumentStore:
        """Support context manager usage."""
        return self

    def __exit__(self, *_: object) -> None:
        """Close the connection on context manager exit."""
        self.close()

    @staticmethod
    def _row_to_document(row: tuple) -> Document:
        """Convert a sqlite3 row tuple to a Document.

        Expected column order: row_index, doc_id, text, metadata, ingested_at.
        metadata is a JSON string; ingested_at is an ISO-8601 string.
        """
        row_index, doc_id, text, metadata_json, ingested_at_iso = row
        return Document(
            row_index=row_index,
            doc_id=doc_id,
            text=text,
            metadata=json.loads(metadata_json),
            ingested_at=(datetime.fromisoformat(ingested_at_iso) if ingested_at_iso else None),
        )
