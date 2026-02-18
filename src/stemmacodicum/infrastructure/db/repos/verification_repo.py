from __future__ import annotations

from pathlib import Path

from stemmacodicum.domain.models.verification import VerificationResult, VerificationRun
from stemmacodicum.infrastructure.db.sqlite import get_connection


class VerificationRepo:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def insert_run(self, run: VerificationRun) -> None:
        with get_connection(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO verification_runs (id, claim_set_id, policy_profile, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (run.id, run.claim_set_id, run.policy_profile, run.created_at),
            )
            conn.commit()

    def insert_result(self, result: VerificationResult) -> None:
        with get_connection(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO verification_results (id, run_id, claim_id, status, diagnostics_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    result.id,
                    result.run_id,
                    result.claim_id,
                    result.status,
                    result.diagnostics_json,
                    result.created_at,
                ),
            )
            conn.commit()

    def list_results(self, run_id: str) -> list[VerificationResult]:
        with get_connection(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT * FROM verification_results
                WHERE run_id = ?
                ORDER BY created_at ASC
                """,
                (run_id,),
            ).fetchall()
        return [self._to_result(r) for r in rows]

    def get_run(self, run_id: str) -> VerificationRun | None:
        with get_connection(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM verification_runs WHERE id = ?",
                (run_id,),
            ).fetchone()
        return self._to_run(row) if row else None

    @staticmethod
    def _to_result(row) -> VerificationResult:
        return VerificationResult(
            id=row["id"],
            run_id=row["run_id"],
            claim_id=row["claim_id"],
            status=row["status"],
            diagnostics_json=row["diagnostics_json"],
            created_at=row["created_at"],
        )

    @staticmethod
    def _to_run(row) -> VerificationRun:
        return VerificationRun(
            id=row["id"],
            claim_set_id=row["claim_set_id"],
            policy_profile=row["policy_profile"],
            created_at=row["created_at"],
        )
