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


class HealthService:
    def __init__(self, db_path: Path, archive_dir: Path) -> None:
        self.db_path = db_path
        self.archive_dir = archive_dir

    def run_doctor(self) -> DoctorReport:
        issues: list[DoctorIssue] = []
        checks_run = 0

        resource_repo = ResourceRepo(self.db_path)
        archive = ArchiveStore(self.archive_dir)

        # Check 1: every resource has an archive object and matching digest.
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

        # Check 2: evidence selector JSON is parseable.
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

        # Check 3: quantitative claims should have at least one evidence binding.
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

        return DoctorReport(ok=not any(i.level == "error" for i in issues), checks_run=checks_run, issues=issues)
