"""Tests for arro_nlp_frontend.migrate._migrate().

All tests operate on raw SQLite files created in tmp_path.
No DocumentStore is instantiated here -- these tests target the
migration script in isolation.

Test inventory:
  1. test_migrate_rewrites_all_rows_with_dataset_id
  2. test_migrate_creates_bak_file
  3. test_migrate_new_db_has_schema_version_2
  4. test_migrate_raises_if_db_not_found
  5. test_migrate_raises_if_bak_already_exists
  6. test_migrate_empty_table_produces_empty_v2_db
  7. test_migrate_then_documentstore_opens_cleanly
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from arro_nlp_frontend.migrate import _migrate

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_v1_db(db_path: Path, rows: list[tuple]) -> None:
    """Create a minimal v1 DocumentStore SQLite file at db_path.

    rows: list of (row_index, doc_id, text, vector_bytes, metadata_json, ingested_at_iso)
    """
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE documents (
            row_index   INTEGER PRIMARY KEY,
            doc_id      TEXT    NOT NULL UNIQUE,
            text        TEXT    NOT NULL,
            vector      BLOB    NOT NULL,
            metadata    TEXT    NOT NULL DEFAULT '{}',
            ingested_at TEXT    NOT NULL
        );
        CREATE INDEX idx_doc_id ON documents(doc_id);
    """)
    conn.executemany(
        "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()


_SAMPLE_ROWS = [
    (
        0,
        "CVE-2024-1",
        "buffer overflow",
        b"\x00" * 8,
        '{"severity": "high"}',
        "2024-01-01T00:00:00+00:00",
    ),
    (1, "CVE-2024-2", "sql injection", b"\x01" * 8, "{}", "2024-01-02T00:00:00+00:00"),
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_migrate_rewrites_all_rows_with_dataset_id(tmp_path: Path) -> None:
    """All v1 rows are present in v2 with dataset_id assigned correctly."""
    db_path = tmp_path / "docs.sqlite"
    _make_v1_db(db_path, _SAMPLE_ROWS)

    _migrate(db_path, "cve/embeddings")

    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        "SELECT dataset_id, row_index, doc_id, text, metadata FROM documents ORDER BY row_index"
    ).fetchall()
    conn.close()

    assert len(rows) == 2

    assert rows[0][0] == "cve/embeddings"
    assert rows[0][1] == 0
    assert rows[0][2] == "CVE-2024-1"
    assert rows[0][3] == "buffer overflow"
    assert rows[0][4] == '{"severity": "high"}'

    assert rows[1][0] == "cve/embeddings"
    assert rows[1][1] == 1
    assert rows[1][2] == "CVE-2024-2"


def test_migrate_creates_bak_file(tmp_path: Path) -> None:
    """The original v1 file is renamed to <name>.v1.bak."""
    db_path = tmp_path / "docs.sqlite"
    _make_v1_db(db_path, _SAMPLE_ROWS)

    _migrate(db_path, "cve/embeddings")

    bak_path = db_path.with_name(db_path.name + ".v1.bak")
    assert bak_path.exists()
    assert bak_path.is_file()
    assert db_path.exists()


def test_migrate_new_db_has_schema_version_2(tmp_path: Path) -> None:
    """schema_version table exists and contains version=2 after migration."""
    db_path = tmp_path / "docs.sqlite"
    _make_v1_db(db_path, _SAMPLE_ROWS)

    _migrate(db_path, "cve/embeddings")

    conn = sqlite3.connect(str(db_path))
    version = conn.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
    conn.close()

    assert version is not None
    assert version[0] == 2


def test_migrate_raises_if_db_not_found(tmp_path: Path) -> None:
    """SystemExit(1) is raised when db_path does not exist."""
    with pytest.raises(SystemExit) as exc_info:
        _migrate(tmp_path / "nonexistent.sqlite", "cve/embeddings")
    assert exc_info.value.code == 1


def test_migrate_raises_if_bak_already_exists(tmp_path: Path) -> None:
    """SystemExit(1) is raised when the .v1.bak file already exists."""
    db_path = tmp_path / "docs.sqlite"
    _make_v1_db(db_path, _SAMPLE_ROWS)

    bak_path = db_path.with_name(db_path.name + ".v1.bak")
    bak_path.write_text("stale backup")

    with pytest.raises(SystemExit) as exc_info:
        _migrate(db_path, "cve/embeddings")
    assert exc_info.value.code == 1

    assert db_path.exists()


def test_migrate_empty_table_produces_empty_v2_db(tmp_path: Path) -> None:
    """Migration of a v1 db with zero rows produces a valid empty v2 db."""
    db_path = tmp_path / "docs.sqlite"
    _make_v1_db(db_path, [])

    _migrate(db_path, "cve/embeddings")

    conn = sqlite3.connect(str(db_path))
    count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    version = conn.execute("SELECT version FROM schema_version LIMIT 1").fetchone()[0]
    conn.close()

    assert count == 0
    assert version == 2


def test_migrate_then_documentstore_opens_cleanly(tmp_path: Path) -> None:
    """After migration, DocumentStore opens without raising RuntimeError."""
    from arro_nlp_frontend.store import DocumentStore

    db_path = tmp_path / "docs.sqlite"
    _make_v1_db(db_path, _SAMPLE_ROWS)

    _migrate(db_path, "cve/embeddings")

    with DocumentStore(db_path) as store:
        assert store.count("cve/embeddings") == 2
        doc = store.get_by_id("cve/embeddings", "CVE-2024-1")
        assert doc is not None
        assert doc.row_index == 0
        assert doc.text == "buffer overflow"
