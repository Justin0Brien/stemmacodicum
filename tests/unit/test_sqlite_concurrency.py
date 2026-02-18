from __future__ import annotations

import threading
import time
from pathlib import Path

from stemmacodicum.infrastructure.db.sqlite import get_connection, initialize_schema


def _schema_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "src"
        / "stemmacodicum"
        / "infrastructure"
        / "db"
        / "schema.sql"
    )


def test_connection_enables_wal_and_busy_timeout(tmp_path: Path) -> None:
    db_path = tmp_path / "stemma.db"
    initialize_schema(db_path=db_path, schema_path=_schema_path())

    with get_connection(db_path) as conn:
        journal_mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
        busy_timeout = conn.execute("PRAGMA busy_timeout;").fetchone()[0]
        foreign_keys = conn.execute("PRAGMA foreign_keys;").fetchone()[0]

    assert str(journal_mode).lower() == "wal"
    assert int(busy_timeout) >= 30_000
    assert int(foreign_keys) == 1


def test_write_waits_for_lock_instead_of_failing_immediately(tmp_path: Path) -> None:
    db_path = tmp_path / "stemma.db"
    initialize_schema(db_path=db_path, schema_path=_schema_path())

    with get_connection(db_path) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS lock_test (id INTEGER PRIMARY KEY, value TEXT NOT NULL);")
        conn.commit()

    writer_1 = get_connection(db_path)
    writer_1.execute("BEGIN IMMEDIATE;")
    writer_1.execute("INSERT INTO lock_test (value) VALUES (?)", ("first",))

    out: dict[str, object] = {}

    def _writer_2() -> None:
        started = time.perf_counter()
        try:
            with get_connection(db_path) as conn_2:
                conn_2.execute("INSERT INTO lock_test (value) VALUES (?)", ("second",))
                conn_2.commit()
            out["ok"] = True
        except Exception as exc:  # pragma: no cover - defensive
            out["ok"] = False
            out["error"] = str(exc)
        finally:
            out["elapsed"] = time.perf_counter() - started

    t = threading.Thread(target=_writer_2)
    t.start()
    time.sleep(0.25)
    writer_1.commit()
    writer_1.close()
    t.join(timeout=5)

    assert out.get("ok") is True, str(out.get("error"))
    assert float(out.get("elapsed", 0.0)) >= 0.2

    with get_connection(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM lock_test").fetchone()[0]
    assert int(count) == 2
