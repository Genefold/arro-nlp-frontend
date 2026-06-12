import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from arro_nlp_frontend.config import settings


logger = logging.getLogger(__name__)


_DDL = """
CREATE TABLE IF NOT EXISTS documents (
    row_index   INTEGER PRIMARY KEY,
    doc_id      TEXT    NOT NULL UNIQUE,
    text        TEXT    NOT NULL,
    metadata    TEXT    NOT NULL DEFAULT '{}',
    ingested_at TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_doc_id ON documents(doc_id);
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
    """SQLite-backed store mapping row_index → Document.

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
        Raises FileNotFoundError only if db_path is a directory, not a file.
        """
        self.db_path = db_path
        self._conn = None
        self.open()

    def open(self) -> None:
        """Open connection and apply schema.

        Creates parent directories. Enables WAL journal mode.
        """
        if self._conn is not None:  # pragma: no cover
            return

        db_path = self.db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        if not db_path.is_file():
            # Ensure it's not a directory we're trying to open
            if db_path.is_dir():
                raise FileNotFoundError(f"Not a file: {db_path}")
            # Create the file if it doesn't exist
            with open(db_path, "w") as f:
                pass

        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute(_DDL)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.commit()

    def upsert_batch(self, start_row: int, docs: list[Document]) -> None:
        """Atomically insert or replace documents starting at start_row.

        Row indices are assigned as start_row, start_row+1, ..., start_row+N-1.
        Uses INSERT OR REPLACE so re-running ingest is idempotent.
        Sets ingested_at to current UTC time for every row in the batch.

        Parameters
        ----------
        start_row: First row index to assign. Must be >= 0.
        docs:      Non-empty list of Document objects.

        Raises
        ------
        ValueError: If docs is empty.
        """
        if not docs:
            raise ValueError("docs cannot be empty")
        if start_row < 0:  # pragma: no cover
            raise ValueError("start_row cannot be negative")

        now = datetime.now(datetime.UTC).isoformat()
        batch_data = [
            (
                start_row + i,
                doc.doc_id,
                doc.text,
                sqlite3.Binary(json.dumps(doc.metadata)),
                now,
            )
            for i, doc in enumerate(docs)
        ]

        with self._conn as conn: # type: ignore[union-attr]
            cursor = conn.cursor()
            cursor.executemany(
                """
                INSERT OR REPLACE INTO documents (row_index, doc_id, text, metadata, ingested_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                batch_data,
            )
            conn.commit()

    def get_by_row(self, row_index: int) -> Document | None:
        """Return the document at row_index, or None if not found."""
        if row_index < 0:  # pragma: no cover
            return None

        cursor = self._conn.execute("SELECT row_index, doc_id, text, metadata, ingested_at FROM documents WHERE row_index = ?", (row_index,))
        row = cursor.fetchone()
        return self._row_to_document(row) if row else None

    def get_by_id(self, doc_id: str) -> Document | None:
        """Return the document with the given doc_id, or None if not found."""
        cursor = self._conn.execute("SELECT row_index, doc_id, text, metadata, ingested_at FROM documents WHERE doc_id = ?", (doc_id,))
        row = cursor.fetchone()
        return self._row_to_document(row) if row else None

    def delete_by_id(self, doc_id: str) -> bool:
        """Delete the document with doc_id from the store.

        Note: This is a SOFT DELETE at the retrieval layer. The corresponding
        vector row in arro-server is NOT removed (Zarr arrays do not support
        row deletion). If a deleted document's row_index is returned by
        arro-server, the search endpoint must handle this gracefully by
        skipping the missing row (see issue #7).

        Returns True if a row was deleted, False if doc_id was not found.
        """
        cursor = self._conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
        deleted = cursor.rowcount > 0
        self._conn.commit()
        return deleted

    def count(self) -> int:
        """Return the total number of documents currently in the store."""
        cursor = self._conn.execute("SELECT COUNT(*) FROM documents")
        return cursor.fetchone()[0]

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "DocumentStore":
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
            ingested_at=datetime.fromisoformat(ingested_at_iso) if ingested_at_iso else None,
        )

