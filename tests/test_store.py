"""Tests for arro_nlp_frontend.store.DocumentStore."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from arro_nlp_frontend.store import Document, DocumentStore

# ── helpers ──────────────────────────────────────────────────────────────────

_DIM = 384


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
    store.upsert_batch(0, [_make_doc(0, "CVE-2024-1"), _make_doc(1, "CVE-2024-2")], _vecs(2))

    doc = store.get_by_row(0)
    assert doc is not None
    assert doc.doc_id == "CVE-2024-1"
    assert doc.text == "text"

    doc = store.get_by_row(1)
    assert doc is not None
    assert doc.doc_id == "CVE-2024-2"


def test_upsert_and_get_by_id(tmp_path: Path) -> None:
    """Write batch, get_by_id returns correct row_index."""
    store = DocumentStore(tmp_path / "docs.sqlite")
    store.upsert_batch(0, [_make_doc(0, "CVE-2024-1"), _make_doc(1, "CVE-2024-2")], _vecs(2))

    doc = store.get_by_id("CVE-2024-1")
    assert doc is not None
    assert doc.row_index == 0

    doc = store.get_by_id("CVE-2024-2")
    assert doc is not None
    assert doc.row_index == 1


def test_upsert_sets_ingested_at(tmp_path: Path) -> None:
    """ingested_at is a UTC-aware datetime after upsert."""
    store = DocumentStore(tmp_path / "docs.sqlite")
    store.upsert_batch(0, [_make_doc(0, "CVE-2024-1")], _vecs(1))

    doc = store.get_by_row(0)
    assert doc is not None
    assert doc.ingested_at is not None
    assert isinstance(doc.ingested_at, datetime)
    assert doc.ingested_at.tzinfo == UTC


def test_upsert_idempotent(tmp_path: Path) -> None:
    """Re-inserting the same batch does not create duplicate rows."""
    store = DocumentStore(tmp_path / "docs.sqlite")
    docs = [_make_doc(0, "CVE-2024-1"), _make_doc(1, "CVE-2024-2")]
    store.upsert_batch(0, docs, _vecs(2))
    store.upsert_batch(0, docs, _vecs(2))
    assert store.count() == 2


def test_upsert_empty_raises(tmp_path: Path) -> None:
    """upsert_batch with an empty list raises ValueError."""
    store = DocumentStore(tmp_path / "docs.sqlite")
    with pytest.raises(ValueError, match="docs cannot be empty"):
        store.upsert_batch(0, [], _vecs(0))


def test_delete_by_id_returns_true(tmp_path: Path) -> None:
    """Deleting an existing document returns True and removes it from the store."""
    store = DocumentStore(tmp_path / "docs.sqlite")
    store.upsert_batch(0, [_make_doc(0, "CVE-2024-1")], _vecs(1))
    assert store.count() == 1

    assert store.delete_by_id("CVE-2024-1") is True
    assert store.count() == 0


def test_delete_missing_returns_false(tmp_path: Path) -> None:
    """Deleting a non-existent doc_id returns False."""
    store = DocumentStore(tmp_path / "docs.sqlite")
    assert store.delete_by_id("NONEXISTENT") is False


def test_get_missing_row_returns_none(tmp_path: Path) -> None:
    """get_by_row on a row that does not exist returns None."""
    store = DocumentStore(tmp_path / "docs.sqlite")
    assert store.get_by_row(999) is None


def test_get_missing_id_returns_none(tmp_path: Path) -> None:
    """get_by_id on a doc_id that does not exist returns None."""
    store = DocumentStore(tmp_path / "docs.sqlite")
    assert store.get_by_id("NOPE") is None


def test_metadata_roundtrip(tmp_path: Path) -> None:
    """Nested dict survives JSON serialisation through write and read."""
    store = DocumentStore(tmp_path / "docs.sqlite")
    meta = {"a": {"b": [1, 2]}}
    store.upsert_batch(0, [_make_doc(0, "CVE-2024-1", meta=meta)], _vecs(1))

    doc = store.get_by_id("CVE-2024-1")
    assert doc is not None
    assert doc.metadata == meta


def test_count_reflects_deletes(tmp_path: Path) -> None:
    """count() returns the number of live documents after a delete."""
    store = DocumentStore(tmp_path / "docs.sqlite")
    store.upsert_batch(
        0,
        [_make_doc(0, "CVE-2024-1"), _make_doc(1, "CVE-2024-2"), _make_doc(2, "CVE-2024-3")],
        _vecs(3),
    )
    assert store.count() == 3

    assert store.delete_by_id("CVE-2024-2") is True
    assert store.count() == 2
    assert store.get_by_id("CVE-2024-2") is None
    assert store.get_by_row(1) is None


def test_context_manager_closes(tmp_path: Path) -> None:
    """Context manager closes the connection on __exit__."""
    db_path = tmp_path / "docs.sqlite"
    with DocumentStore(db_path) as store:
        assert store._conn is not None
        assert store.count() == 0
    assert store._conn is None


def test_parent_dirs_created(tmp_path: Path) -> None:
    """DocumentStore creates intermediate parent directories automatically."""
    db_path = tmp_path / "a" / "b" / "c.sqlite"
    with DocumentStore(db_path) as store:
        assert db_path.exists()
        assert db_path.is_file()
        assert store.count() == 0


def test_consecutive_batches_row_index(tmp_path: Path) -> None:
    """Second batch appended at start_row=2 maps correctly to get_by_row(3)."""
    store = DocumentStore(tmp_path / "docs.sqlite")
    store.upsert_batch(0, [_make_doc(0, "CVE-2024-1"), _make_doc(1, "CVE-2024-2")], _vecs(2))
    store.upsert_batch(2, [_make_doc(2, "CVE-2024-3"), _make_doc(3, "CVE-2024-4")], _vecs(2))

    assert store.count() == 4
    assert store.get_by_row(0).doc_id == "CVE-2024-1"  # type: ignore[union-attr]
    assert store.get_by_row(3).doc_id == "CVE-2024-4"  # type: ignore[union-attr]


def test_next_row_index_empty_store(tmp_path: Path) -> None:
    """next_row_index returns 0 when the store is empty."""
    store = DocumentStore(tmp_path / "docs.sqlite")
    assert store.next_row_index() == 0


def test_next_row_index_after_insert(tmp_path: Path) -> None:
    """next_row_index returns MAX(row_index)+1 after inserts."""
    store = DocumentStore(tmp_path / "docs.sqlite")
    store.upsert_batch(0, [_make_doc(0, "CVE-2024-1"), _make_doc(1, "CVE-2024-2")], _vecs(2))
    assert store.next_row_index() == 2


def test_next_row_index_uses_max_not_count(tmp_path: Path) -> None:
    """next_row_index returns MAX+1, not COUNT, so soft-deletes do not cause row reuse."""
    store = DocumentStore(tmp_path / "docs.sqlite")
    store.upsert_batch(
        0,
        [_make_doc(0, "CVE-2024-1"), _make_doc(1, "CVE-2024-2"), _make_doc(2, "CVE-2024-3")],
        _vecs(3),
    )
    store.delete_by_id("CVE-2024-2")  # row 1 is now a ghost
    # count() == 2, but next safe index is 3, not 2
    assert store.count() == 2
    assert store.next_row_index() == 3


def test_update_document_text(tmp_path: Path) -> None:
    """Upserting the same row_index with new text updates the stored text."""
    store = DocumentStore(tmp_path / "docs.sqlite")
    store.upsert_batch(0, [_make_doc(0, "CVE-2024-1", text="original")], _vecs(1))
    store.upsert_batch(0, [_make_doc(0, "CVE-2024-1", text="updated")], _vecs(1))

    doc = store.get_by_id("CVE-2024-1")
    assert doc is not None
    assert doc.text == "updated"


def test_update_document_metadata(tmp_path: Path) -> None:
    """Upserting the same row_index with new metadata updates the stored metadata."""
    store = DocumentStore(tmp_path / "docs.sqlite")
    store.upsert_batch(0, [_make_doc(0, "CVE-2024-1", meta={"k": "v1"})], _vecs(1))
    store.upsert_batch(0, [_make_doc(0, "CVE-2024-1", meta={"k": "v2", "extra": 1})], _vecs(1))

    doc = store.get_by_id("CVE-2024-1")
    assert doc is not None
    assert doc.metadata == {"k": "v2", "extra": 1}
