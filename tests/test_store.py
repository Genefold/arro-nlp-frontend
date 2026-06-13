"""Tests for arro_nlp_frontend.store.DocumentStore."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from arro_nlp_frontend.store import Document, DocumentStore

# ── helpers ──────────────────────────────────────────────────────────────────

_DIM = 384
_DS = "test/dataset"


def _make_doc(
    row: int,
    doc_id: str,
    text: str = "text",
    meta: dict | None = None,
) -> Document:
    return Document(
        row_index=row,
        doc_id=doc_id,
        text=text,
        metadata=meta or {},
        ingested_at=None,
    )


def _vecs(n: int) -> np.ndarray:
    """Return a float64 array of shape (n, _DIM) with deterministic values."""
    return np.arange(n * _DIM, dtype=np.float64).reshape(n, _DIM)


# ── tests ─────────────────────────────────────────────────────────────────────


def test_upsert_and_get_by_row(tmp_path: Path) -> None:
    """Write 2-doc batch, get_by_row(0) returns correct doc_id and text."""
    store = DocumentStore(tmp_path / "docs.sqlite")
    store.upsert_batch(_DS, 0, [_make_doc(0, "CVE-2024-1"), _make_doc(1, "CVE-2024-2")], _vecs(2))

    doc = store.get_by_row(_DS, 0)
    assert doc is not None
    assert doc.doc_id == "CVE-2024-1"
    assert doc.text == "text"

    doc = store.get_by_row(_DS, 1)
    assert doc is not None
    assert doc.doc_id == "CVE-2024-2"


def test_upsert_and_get_by_id(tmp_path: Path) -> None:
    """Write batch, get_by_id returns correct row_index."""
    store = DocumentStore(tmp_path / "docs.sqlite")
    store.upsert_batch(_DS, 0, [_make_doc(0, "CVE-2024-1"), _make_doc(1, "CVE-2024-2")], _vecs(2))

    doc = store.get_by_id(_DS, "CVE-2024-1")
    assert doc is not None
    assert doc.row_index == 0

    doc = store.get_by_id(_DS, "CVE-2024-2")
    assert doc is not None
    assert doc.row_index == 1


def test_upsert_sets_ingested_at(tmp_path: Path) -> None:
    """ingested_at is a UTC-aware datetime after upsert."""
    store = DocumentStore(tmp_path / "docs.sqlite")
    store.upsert_batch(_DS, 0, [_make_doc(0, "CVE-2024-1")], _vecs(1))

    doc = store.get_by_row(_DS, 0)
    assert doc is not None
    assert doc.ingested_at is not None
    assert isinstance(doc.ingested_at, datetime)
    assert doc.ingested_at.tzinfo == UTC


def test_upsert_idempotent(tmp_path: Path) -> None:
    """Re-inserting the same batch does not create duplicate rows."""
    store = DocumentStore(tmp_path / "docs.sqlite")
    docs = [_make_doc(0, "CVE-2024-1"), _make_doc(1, "CVE-2024-2")]
    store.upsert_batch(_DS, 0, docs, _vecs(2))
    store.upsert_batch(_DS, 0, docs, _vecs(2))
    assert store.count(_DS) == 2


def test_upsert_empty_raises(tmp_path: Path) -> None:
    """upsert_batch with an empty list raises ValueError."""
    store = DocumentStore(tmp_path / "docs.sqlite")
    with pytest.raises(ValueError, match="docs cannot be empty"):
        store.upsert_batch(_DS, 0, [], _vecs(0))


def test_delete_by_id_returns_true(tmp_path: Path) -> None:
    """Deleting an existing document returns True and removes it from the store."""
    store = DocumentStore(tmp_path / "docs.sqlite")
    store.upsert_batch(_DS, 0, [_make_doc(0, "CVE-2024-1")], _vecs(1))
    assert store.count(_DS) == 1

    assert store.delete_by_id(_DS, "CVE-2024-1") is True
    assert store.count(_DS) == 0


def test_delete_missing_returns_false(tmp_path: Path) -> None:
    """Deleting a non-existent doc_id returns False."""
    store = DocumentStore(tmp_path / "docs.sqlite")
    assert store.delete_by_id(_DS, "NONEXISTENT") is False


def test_get_missing_row_returns_none(tmp_path: Path) -> None:
    """get_by_row on a row that does not exist returns None."""
    store = DocumentStore(tmp_path / "docs.sqlite")
    assert store.get_by_row(_DS, 999) is None


def test_get_missing_id_returns_none(tmp_path: Path) -> None:
    """get_by_id on a doc_id that does not exist returns None."""
    store = DocumentStore(tmp_path / "docs.sqlite")
    assert store.get_by_id(_DS, "NOPE") is None


def test_metadata_roundtrip(tmp_path: Path) -> None:
    """Nested dict survives JSON serialisation through write and read."""
    store = DocumentStore(tmp_path / "docs.sqlite")
    meta = {"a": {"b": [1, 2]}}
    store.upsert_batch(_DS, 0, [_make_doc(0, "CVE-2024-1", meta=meta)], _vecs(1))

    doc = store.get_by_id(_DS, "CVE-2024-1")
    assert doc is not None
    assert doc.metadata == meta


def test_count_reflects_deletes(tmp_path: Path) -> None:
    """count() returns the number of live documents after a delete."""
    store = DocumentStore(tmp_path / "docs.sqlite")
    store.upsert_batch(
        _DS,
        0,
        [_make_doc(0, "CVE-2024-1"), _make_doc(1, "CVE-2024-2"), _make_doc(2, "CVE-2024-3")],
        _vecs(3),
    )
    assert store.count(_DS) == 3

    assert store.delete_by_id(_DS, "CVE-2024-2") is True
    assert store.count(_DS) == 2
    assert store.get_by_id(_DS, "CVE-2024-2") is None
    assert store.get_by_row(_DS, 1) is None


def test_context_manager_closes(tmp_path: Path) -> None:
    """Context manager closes the connection on __exit__."""
    db_path = tmp_path / "docs.sqlite"
    with DocumentStore(db_path) as store:
        assert store._conn is not None
        assert store.count(_DS) == 0
    assert store._conn is None


def test_parent_dirs_created(tmp_path: Path) -> None:
    """DocumentStore creates intermediate parent directories automatically."""
    db_path = tmp_path / "a" / "b" / "c.sqlite"
    with DocumentStore(db_path) as store:
        assert db_path.exists()
        assert db_path.is_file()
        assert store.count(_DS) == 0


def test_consecutive_batches_row_index(tmp_path: Path) -> None:
    """Second batch appended at start_row=2 maps correctly to get_by_row(3)."""
    store = DocumentStore(tmp_path / "docs.sqlite")
    store.upsert_batch(_DS, 0, [_make_doc(0, "CVE-2024-1"), _make_doc(1, "CVE-2024-2")], _vecs(2))
    store.upsert_batch(_DS, 2, [_make_doc(2, "CVE-2024-3"), _make_doc(3, "CVE-2024-4")], _vecs(2))

    assert store.count(_DS) == 4
    assert store.get_by_row(_DS, 0).doc_id == "CVE-2024-1"  # type: ignore[union-attr]
    assert store.get_by_row(_DS, 3).doc_id == "CVE-2024-4"  # type: ignore[union-attr]


def test_next_row_index_empty_store(tmp_path: Path) -> None:
    """next_row_index returns 0 when the store is empty."""
    store = DocumentStore(tmp_path / "docs.sqlite")
    assert store.next_row_index(_DS) == 0


def test_next_row_index_after_insert(tmp_path: Path) -> None:
    """next_row_index returns MAX(row_index)+1 after inserts."""
    store = DocumentStore(tmp_path / "docs.sqlite")
    store.upsert_batch(_DS, 0, [_make_doc(0, "CVE-2024-1"), _make_doc(1, "CVE-2024-2")], _vecs(2))
    assert store.next_row_index(_DS) == 2


def test_next_row_index_uses_max_not_count(tmp_path: Path) -> None:
    """next_row_index returns MAX+1, not COUNT, so soft-deletes do not cause row reuse."""
    store = DocumentStore(tmp_path / "docs.sqlite")
    store.upsert_batch(
        _DS,
        0,
        [_make_doc(0, "CVE-2024-1"), _make_doc(1, "CVE-2024-2"), _make_doc(2, "CVE-2024-3")],
        _vecs(3),
    )
    store.delete_by_id(_DS, "CVE-2024-2")  # row 1 is now a ghost
    assert store.count(_DS) == 2
    assert store.next_row_index(_DS) == 3


def test_update_document_text(tmp_path: Path) -> None:
    """Upserting the same row_index with new text updates the stored text."""
    store = DocumentStore(tmp_path / "docs.sqlite")
    store.upsert_batch(_DS, 0, [_make_doc(0, "CVE-2024-1", text="original")], _vecs(1))
    store.upsert_batch(_DS, 0, [_make_doc(0, "CVE-2024-1", text="updated")], _vecs(1))

    doc = store.get_by_id(_DS, "CVE-2024-1")
    assert doc is not None
    assert doc.text == "updated"


def test_update_document_metadata(tmp_path: Path) -> None:
    """Upserting the same row_index with new metadata updates the stored metadata."""
    store = DocumentStore(tmp_path / "docs.sqlite")
    store.upsert_batch(_DS, 0, [_make_doc(0, "CVE-2024-1", meta={"k": "v1"})], _vecs(1))
    store.upsert_batch(_DS, 0, [_make_doc(0, "CVE-2024-1", meta={"k": "v2", "extra": 1})], _vecs(1))

    doc = store.get_by_id(_DS, "CVE-2024-1")
    assert doc is not None
    assert doc.metadata == {"k": "v2", "extra": 1}


# ── Multi-dataset tests ───────────────────────────────────────────────────────


def test_store_isolation_between_datasets(tmp_path: Path) -> None:
    """Documents in different datasets do not bleed through."""
    store = DocumentStore(tmp_path / "docs.sqlite")
    store.upsert_batch("ds/a", 0, [_make_doc(0, "doc-a")], _vecs(1))
    store.upsert_batch("ds/b", 0, [_make_doc(0, "doc-b")], _vecs(1))

    assert store.get_by_row("ds/a", 0).doc_id == "doc-a"  # type: ignore[union-attr]
    assert store.get_by_row("ds/b", 0).doc_id == "doc-b"  # type: ignore[union-attr]
    assert store.get_by_row("ds/a", 1) is None

    assert store.get_all_vectors("ds/a").shape == (1, _DIM)
    assert store.get_all_vectors("ds/b").shape == (1, _DIM)


def test_store_next_row_index_per_dataset(tmp_path: Path) -> None:
    """Each dataset has an independent next_row_index counter."""
    store = DocumentStore(tmp_path / "docs.sqlite")
    store.upsert_batch("ds/a", 0, [_make_doc(0, "a0"), _make_doc(1, "a1")], _vecs(2))

    assert store.next_row_index("ds/b") == 0
    assert store.next_row_index("ds/a") == 2


def test_store_count_total_vs_per_dataset(tmp_path: Path) -> None:
    """count() scoped to dataset vs global."""
    store = DocumentStore(tmp_path / "docs.sqlite")
    store.upsert_batch("ds/a", 0, [_make_doc(0, "a0"), _make_doc(1, "a1")], _vecs(2))
    store.upsert_batch(
        "ds/b", 0, [_make_doc(0, "b0"), _make_doc(1, "b1"), _make_doc(2, "b2")], _vecs(3)
    )

    assert store.count("ds/a") == 2
    assert store.count("ds/b") == 3
    assert store.count(None) == 5


def test_store_delete_scoped_to_dataset(tmp_path: Path) -> None:
    """Deleting from one dataset does not affect another."""
    store = DocumentStore(tmp_path / "docs.sqlite")
    store.upsert_batch("ds/a", 0, [_make_doc(0, "doc0")], _vecs(1))
    store.upsert_batch("ds/b", 0, [_make_doc(0, "doc0")], _vecs(1))

    assert store.delete_by_id("ds/a", "doc0") is True
    assert store.get_by_id("ds/a", "doc0") is None
    assert store.get_by_id("ds/b", "doc0") is not None


def test_store_migration_v1_raises_runtime_error(tmp_path: Path) -> None:
    """Opening a v1 schema file (no dataset_id) raises RuntimeError."""
    db_path = tmp_path / "v1.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE documents (
            row_index INTEGER PRIMARY KEY,
            doc_id    TEXT NOT NULL UNIQUE,
            text      TEXT NOT NULL,
            vector    BLOB NOT NULL,
            metadata  TEXT NOT NULL DEFAULT '{}',
            ingested_at TEXT NOT NULL
        );
        CREATE INDEX idx_doc_id ON documents(doc_id);
        INSERT INTO documents VALUES (0, 'old', 'text', x'00', '{}', '2024-01-01');
    """)
    conn.commit()
    conn.close()

    with pytest.raises(RuntimeError, match="version-1"):
        DocumentStore(db_path)
