"""One-shot migration: DocumentStore v1 (no dataset_id) to v2 (dataset_id column).

Usage:
    python -m arro_nlp_frontend.migrate \\
        --db-path ./data/documents.sqlite \\
        --dataset-id cve/embeddings

The original file is renamed to <name>.v1.bak before migration.
Run this once before starting the server after upgrading to #17.
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
from pathlib import Path


def _migrate(db_path: Path, dataset_id: str) -> None:
    if not db_path.exists():
        print(f"Error: database not found: {db_path}")
        raise SystemExit(1)

    bak_path = db_path.with_name(db_path.name + ".v1.bak")
    if bak_path.exists():
        print(f"Error: backup already exists: {bak_path}")
        print("Delete the backup manually if you want to re-run the migration.")
        raise SystemExit(1)

    # Open old DB read-only and read all rows
    old_conn = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)
    try:
        cursor = old_conn.execute(
            "SELECT row_index, doc_id, text, vector, metadata, ingested_at FROM documents"
        )
        rows = cursor.fetchall()
    except sqlite3.OperationalError as exc:
        print(f"Error: could not read documents from old schema: {exc}")
        raise SystemExit(1) from exc
    finally:
        old_conn.close()

    # Rename old file to .v1.bak
    shutil.move(str(db_path), str(bak_path))

    # Create new DB with v2 schema
    new_conn = sqlite3.connect(str(db_path))
    try:
        new_conn.executescript("""
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
        """)

        new_conn.execute("INSERT INTO schema_version (version) VALUES (2)")

        # Insert all rows with the provided dataset_id
        new_conn.executemany(
            """
            INSERT INTO documents
                (dataset_id, row_index, doc_id, text, vector, metadata, ingested_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (dataset_id, row_idx, doc_id, text, vector, metadata, ingested_at)
                for row_idx, doc_id, text, vector, metadata, ingested_at in rows
            ],
        )
        new_conn.commit()
    finally:
        new_conn.close()

    print(f"Migrated {len(rows)} documents to dataset_id={dataset_id!r}")
    print(f"Backup saved as: {bak_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate DocumentStore v1 (no dataset_id) to v2.")
    parser.add_argument(
        "--db-path",
        required=True,
        type=Path,
        help="Path to the existing documents.sqlite file.",
    )
    parser.add_argument(
        "--dataset-id",
        required=True,
        type=str,
        help="Dataset identifier to assign to all existing rows (e.g. cve/embeddings).",
    )
    args = parser.parse_args()

    if not args.dataset_id.strip():
        print("Error: --dataset-id must not be empty.")
        raise SystemExit(1)

    _migrate(db_path=args.db_path, dataset_id=args.dataset_id)


if __name__ == "__main__":
    main()
