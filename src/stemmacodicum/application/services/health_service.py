from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from stemmacodicum.infrastructure.archive.store import ArchiveStore
from stemmacodicum.infrastructure.db.repos.resource_repo import ResourceRepo
from stemmacodicum.infrastructure.db.sqlite import get_connection


@dataclass(slots=True)
class DoctorIssue:
    check: str
    level: str
    message: str


@dataclass(slots=True)
class DoctorReport:
    ok: bool
    checks_run: int
    issues: list[DoctorIssue]
    db_runtime: dict[str, object]


class HealthService:
    def __init__(self, db_path: Path, archive_dir: Path) -> None:
        self.db_path = db_path
        self.archive_dir = archive_dir

    def run_doctor(self) -> DoctorReport:
        issues: list[DoctorIssue] = []
        checks_run = 0
        db_runtime: dict[str, object] = {}

        resource_repo = ResourceRepo(self.db_path)
        archive = ArchiveStore(self.archive_dir)

        # Check 1: database runtime pragmas support concurrent access.
        checks_run += 1
        with get_connection(self.db_path) as conn:
            journal_mode_raw = conn.execute("PRAGMA journal_mode;").fetchone()[0]
            busy_timeout_raw = conn.execute("PRAGMA busy_timeout;").fetchone()[0]
            foreign_keys_raw = conn.execute("PRAGMA foreign_keys;").fetchone()[0]
            synchronous_raw = conn.execute("PRAGMA synchronous;").fetchone()[0]
            wal_autocheckpoint_raw = conn.execute("PRAGMA wal_autocheckpoint;").fetchone()[0]

        journal_mode = str(journal_mode_raw).lower()
        busy_timeout_ms = int(busy_timeout_raw)
        foreign_keys = int(foreign_keys_raw)
        synchronous = int(synchronous_raw)
        wal_autocheckpoint = int(wal_autocheckpoint_raw)

        db_runtime = {
            "journal_mode": journal_mode,
            "busy_timeout_ms": busy_timeout_ms,
            "foreign_keys": bool(foreign_keys),
            "synchronous": synchronous,
            "wal_autocheckpoint_pages": wal_autocheckpoint,
        }

        if journal_mode != "wal":
            issues.append(
                DoctorIssue(
                    check="db_runtime",
                    level="error",
                    message=f"SQLite journal_mode is '{journal_mode}', expected 'wal' for concurrent access.",
                )
            )
        if foreign_keys != 1:
            issues.append(
                DoctorIssue(
                    check="db_runtime",
                    level="error",
                    message="SQLite foreign_keys pragma is disabled.",
                )
            )
        if busy_timeout_ms <= 0:
            issues.append(
                DoctorIssue(
                    check="db_runtime",
                    level="error",
                    message="SQLite busy_timeout is disabled; concurrent writes may fail immediately.",
                )
            )
        elif busy_timeout_ms < 1_000:
            issues.append(
                DoctorIssue(
                    check="db_runtime",
                    level="warning",
                    message=f"SQLite busy_timeout is low ({busy_timeout_ms}ms); consider >= 1000ms.",
                )
            )

        # Check 2: every resource has an archive object and matching digest.
        checks_run += 1
        for res in resource_repo.list(limit=1_000_000):
            path = self.archive_dir / res.archived_relpath
            if not path.exists():
                issues.append(
                    DoctorIssue(
                        check="archive_integrity",
                        level="error",
                        message=f"Missing archive file for resource {res.id}: {path}",
                    )
                )
                continue
            if not archive.verify_archived_integrity(path, res.digest_sha256):
                issues.append(
                    DoctorIssue(
                        check="archive_integrity",
                        level="error",
                        message=f"Digest mismatch for resource {res.id}: {path}",
                    )
                )

        # Check 3: evidence selector JSON is parseable.
        checks_run += 1
        with get_connection(self.db_path) as conn:
            rows = conn.execute("SELECT id, selector_json FROM evidence_selectors").fetchall()
            for row in rows:
                try:
                    json.loads(row["selector_json"])
                except Exception:
                    issues.append(
                        DoctorIssue(
                            check="selector_json",
                            level="error",
                            message=f"Invalid selector JSON for selector {row['id']}",
                        )
                    )

        # Check 4: quantitative claims should have at least one evidence binding.
        checks_run += 1
        with get_connection(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT c.id
                FROM claims c
                LEFT JOIN claim_evidence_bindings b ON b.claim_id = c.id
                WHERE c.claim_type = 'quantitative'
                GROUP BY c.id
                HAVING COUNT(b.evidence_id) = 0
                """
            ).fetchall()
            for row in rows:
                issues.append(
                    DoctorIssue(
                        check="quantitative_bindings",
                        level="warning",
                        message=f"Quantitative claim without evidence bindings: {row['id']}",
                    )
                )

        return DoctorReport(
            ok=not any(i.level == "error" for i in issues),
            checks_run=checks_run,
            issues=issues,
            db_runtime=db_runtime,
        )
