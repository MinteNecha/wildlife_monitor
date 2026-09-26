"""
Database connections and schema management.

One module owns how the SQLite file is opened and how the schema is applied,
so no other part of the system constructs a connection by hand. Connections
are configured consistently: foreign keys enforced (SQLite leaves them off by
default, which would silently allow orphaned detections), row access by column
name, and write-ahead logging so the dashboard can read while a pipeline run
is still writing.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from wildlife_monitor.config import DATA_DIR

DB_PATH = DATA_DIR / "wildlife.db"
SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def connect(path: str | Path | None = None) -> sqlite3.Connection:
    """Open a configured connection, creating the file if it is absent."""
    target = Path(path) if path else DB_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(target, timeout=30.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    return connection


@contextmanager
def session(path: str | Path | None = None) -> Iterator[sqlite3.Connection]:
    """Transactional connection: commits on success, rolls back on error."""
    connection = connect(path)
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def init_db(path: str | Path | None = None) -> Path:
    """Apply the schema. Safe to run repeatedly — every statement is IF NOT EXISTS."""
    target = Path(path) if path else DB_PATH
    with session(target) as connection:
        connection.executescript(SCHEMA_PATH.read_text())
    return target


def database_exists(path: str | Path | None = None) -> bool:
    """True when the database file exists and carries the schema."""
    target = Path(path) if path else DB_PATH
    if not target.exists():
        return False
    try:
        with session(target) as connection:
            row = connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name='Detection'").fetchone()
        return row is not None
    except sqlite3.Error:
        return False


def table_counts(path: str | Path | None = None) -> dict[str, int]:
    """Row count per table — used by the setup script and the dashboard."""
    if not database_exists(path):
        return {}
    counts: dict[str, int] = {}
    with session(path) as connection:
        tables = [row["name"] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name")]
        for table in tables:
            counts[table] = connection.execute(
                f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]
    return counts
