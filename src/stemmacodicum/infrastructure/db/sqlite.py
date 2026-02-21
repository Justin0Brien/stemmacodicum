from __future__ import annotations

import os
import sqlite3
from pathlib import Path

DEFAULT_SQLITE_CONNECT_TIMEOUT_SECONDS = 30.0
DEFAULT_SQLITE_BUSY_TIMEOUT_MS = 30_000


def _read_float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _read_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _sqlite_connect_timeout_seconds() -> float:
    return _read_float_env("STEMMA_SQLITE_CONNECT_TIMEOUT_SECONDS", DEFAULT_SQLITE_CONNECT_TIMEOUT_SECONDS)


def _sqlite_busy_timeout_ms() -> int:
    return _read_int_env("STEMMA_SQLITE_BUSY_TIMEOUT_MS", DEFAULT_SQLITE_BUSY_TIMEOUT_MS)


def _configure_connection(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.execute(f"PRAGMA busy_timeout = {_sqlite_busy_timeout_ms()};")


def get_connection(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=_sqlite_connect_timeout_seconds())
    conn.row_factory = sqlite3.Row
    _configure_connection(conn)
    return conn


def initialize_schema(db_path: Path, schema_path: Path) -> None:
    with get_connection(db_path) as conn:
        conn.executescript(schema_path.read_text(encoding="utf-8"))
        _apply_lightweight_migrations(conn)
        conn.commit()


def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(row["name"] == column for row in rows)


def _apply_lightweight_migrations(conn: sqlite3.Connection) -> None:
    if not _column_exists(conn, "resources", "download_url"):
        conn.execute("ALTER TABLE resources ADD COLUMN download_url TEXT")
    if not _column_exists(conn, "resources", "download_urls_json"):
        conn.execute("ALTER TABLE resources ADD COLUMN download_urls_json TEXT")
    if not _column_exists(conn, "resources", "display_title"):
        conn.execute("ALTER TABLE resources ADD COLUMN display_title TEXT")
    if not _column_exists(conn, "resources", "title_candidates_json"):
        conn.execute("ALTER TABLE resources ADD COLUMN title_candidates_json TEXT")
