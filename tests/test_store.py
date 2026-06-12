"""Tests for arro_nlp_frontend.store.DocumentStore."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from arro_nlp_frontend.store import Document, DocumentStore


# ── helpers ─────────────────────────────────────────────────────────────────

def _make_doc(
    row: int,
    doc_id: str,
    text: str = "text",
    meta: dict | None = None,
    ingested_at: datetime | None = None,
) -> Document:
    return Document(
        row_index=row,
        doc_id=doc_id,
        text=text,
        metadata=meta or {},
        ingested_at=ingested_at,
    )


# ── tests ───────────────────────────────────────────────────────────────────

def test_upsert_and_get_by_row(tmp_path: Path) -> None:
    """Write 2-doc batch, get_by_row(0) returns correct doc_id and text."""
    store = DocumentStore(tmp_path / "docs.sqlite")
    docs = [_make_doc(0, "CVE-2024-1"), _make_doc(1, "CVE-2024-2")]
    store.upsert_batch(0, docs)

    retrieved = store.get_by_row(0)
    assert retrieved is not None
    assert retrieved.doc_id == "CVE-2024-1"
    assert retrieved.text == "text"

    retrieved = store.get_by_row(1)
    assert retrieved is not None
    assert retrieved.doc_id == "CVE-2024-2"
    assert retrieved.text == "text"


def test_upsert_and_get_by_id(tmp_path: Path) -> None:
    """Write batch, get_by_id("CVE-2024-1") returns correct row_index."""
    store = DocumentStore(tmp_path / "docs.sqlite")
    docs = [_make_doc(0, "CVE-2024-1"), _make_doc(1, "CVE-2024-2")]
    store.upsert_batch(0, docs)

    retrieved = store.get_by_id("CVE-2024-1")
    assert retrieved is not None
    assert retrieved.row_index == 0

    retrieved = store.get_by_id("CVE-2024-2")
    assert retrieved is not None
    assert retrieved.row_index == 1


def test_upsert_sets_ingested_at(tmp_path: Path) -> None:
    """`ingested_at` is not None after write; is a `datetime` object."""
    store = DocumentStore(tmp_path / "docs.sqlite")
    docs = [_make_doc(0, "CVE-2024-1")]
    store.upsert_batch(0, docs)

    retrieved = store.get_by_row(0)
    assert retrieved is not None
    assert retrieved.ingested_at is not None
    assert isinstance(retrieved.ingested_at, datetime)
    assert retrieved.ingested_at.tzinfo == timezone.utc


def test_upsert_idempotent(tmp_path: Path) -> None:
    """Re-inserting same batch twice → `count() == 2` (no duplicates)."""
    store = DocumentStore(tmp_path / "docs.sqlite")
    docs = [_make_doc(0, "CVE-2024-1"), _make_doc(1, "CVE-2024-2")]
    store.upsert_batch(0, docs)
    assert store.count() == 2

    # Re-insert the same batch
    store.upsert_batch(0, docs)
    assert store.count() == 2


def test_upsert_empty_raises(tmp_path: Path) -> None:
    """`upsert_batch(0, [])` raises `ValueError`."""
    store = DocumentStore(tmp_path / "docs.sqlite")
    with pytest.raises(ValueError, match="docs cannot be empty"):
        store.upsert_batch(0, [])


def test_delete_by_id_returns_true(tmp_path: Path) -> None:
    """Insert 1 doc, delete it → returns `True`, `count() == 0`."""
    store = DocumentStore(tmp_path / "docs.sqlite")
    docs = [_make_doc(0, "CVE-2024-1")]
    store.upsert_batch(0, docs)
    assert store.count() == 1

    deleted = store.delete_by_id("CVE-2024-1")
    assert deleted is True
    assert store.count() == 0


def test_delete_missing_returns_false(tmp_path: Path) -> None:
    """Delete non-existent `doc_id` → returns `False`."""
    store = DocumentStore(tmp_path / "docs.sqlite")
    deleted = store.delete_by_id("NONEXISTENT-ID")
    assert deleted is False


def test_get_missing_row_returns_none(tmp_path: Path) -> None:
    """`get_by_row(999)` on empty store → `None`."""
    store = DocumentStore(tmp_path / "docs.sqlite")
    assert store.get_by_row(999) is None


def test_get_missing_id_returns_none(tmp_path: Path) -> None:
    """`get_by_id("NOPE")` on empty store → `None`."""
    store = DocumentStore(tmp_path / "docs.sqlite")
    assert store.get_by_id("NOPE") is None


def test_metadata_roundtrip(tmp_path: Path) -> None:
    """Nested dict `{"a": {"b": [1, 2]}}` survives write/read cycle."""
    store = DocumentStore(tmp_path / "docs.sqlite")
    doc_id = "CVE-2024-1"
    original_meta = {"a": {"b": [1, 2]}}
    doc = _make_doc(0, doc_id, meta=original_meta)
    store.upsert_batch(0, [doc])

    retrieved = store.get_by_id(doc_id)
    assert retrieved is not None
    assert retrieved.metadata == original_meta


def test_count_reflects_deletes(tmp_path: Path) -> None:
    """Insert 3, delete 1 → `count() == 2`."""
    store = DocumentStore(tmp_path / "docs.sqlite")
    docs = [
        _make_doc(0, "CVE-2024-1"),
        _make_doc(1, "CVE-2024-2"),
        _make_doc(2, "CVE-2024-3"),
    ]
    store.upsert_batch(0, docs)
    assert store.count() == 3

    deleted = store.delete_by_id("CVE-2024-2")
    assert deleted is True
    assert store.count() == 2

    # Ensure deleted doc is no longer retrievable
    assert store.get_by_id("CVE-2024-2") is None
    assert store.get_by_row(1) is None


def test_context_manager_closes(tmp_path: Path) -> None:
    """`with DocumentStore(...) as s:` executes without error; `s._conn` is closed after block."""
    db_path = tmp_path / "docs.sqlite"
    with DocumentStore(db_path) as store:
        assert store._conn is not None
        assert store.count() == 0
    # Connection should be closed after exiting the with block
    assert store._conn is None


def test_parent_dirs_created(tmp_path: Path) -> None:
    """`db_path = tmp_path / "a" / "b" / "c.sqlite"` → no FileNotFoundError."""
    db_path = tmp_path / "a" / "b" / "c.sqlite"
    # Expecting the directory to be created automatically
    with DocumentStore(db_path) as store:
        assert store.db_path == db_path
        assert db_path.exists()
        assert db_path.is_file()
        assert store.count() == 0


def test_consecutive_batches_row_index(tmp_path: Path) -> None:
    """Two batches of 2 docs each; second batch starts at `start_row=2`; `get_by_row(3)` returns correct doc."""
    store = DocumentStore(tmp_path / "docs.sqlite")

    # First batch
    docs1 = [_make_doc(0, "CVE-2024-1"), _make_doc(1, "CVE-2024-2")]
    store.upsert_batch(0, docs1)
    assert store.count() == 2

    # Second batch
    docs2 = [_make_doc(2, "CVE-2024-3"), _make_doc(3, "CVE-2024-4")]
    store.upsert_batch(2, docs2)
    assert store.count() == 4

    # Verify documents from both batches
    assert store.get_by_row(0).doc_id == "CVE-2024-1"
    assert store.get_by_row(1).doc_id == "CVE-2024-2"
    assert store.get_by_row(2).doc_id == "CVE-2024-3"
    assert store.get_by_row(3).doc_id == "CVE-2024-4"


def test_update_document_text(tmp_path: Path) -> None:
    """Ensure document text can be updated."""
    store = DocumentStore(tmp_path / "docs.sqlite")
    doc_id = "CVE-2024-1"
    original_text = "Initial text"
    updated_text = "Updated text"

    doc = _make_doc(0, doc_id, text=original_text)
    store.upsert_batch(0, [doc])

    retrieved = store.get_by_id(doc_id)
    assert retrieved is not None
    assert retrieved.text == original_text

    # Update the document with new text
    updated_doc = _make_doc(0, doc_id, text=updated_text)
    store.upsert_batch(0, [updated_doc])

    retrieved_updated = store.get_by_id(doc_id)
    assert retrieved_updated is not None
    assert retrieved_updated.text == updated_text


def test_update_document_metadata(tmp_path: Path) -> None:
    """Ensure document metadata can be updated."""
    store = DocumentStore(tmp_path / "docs.sqlite")
    doc_id = "CVE-2024-1"
    original_meta = {"key": "value"}
    updated_meta = {"key": "new_value", "another": 123}

    doc = _make_doc(0, doc_id, meta=original_meta)
    store.upsert_batch(0, [doc])

    retrieved = store.get_by_id(doc_id)
    assert retrieved is not None
    assert retrieved.metadata == original_meta

    # Update the document with new metadata
    updated_doc = _make_doc(0, doc_id, meta=updated_meta)
    store.upsert_batch(0, [updated_doc])

    retrieved_updated = store.get_by_id(doc_id)
    assert retrieved_updated is not None
    assert retrieved_updated.metadata == updated_meta
