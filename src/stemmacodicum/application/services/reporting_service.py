from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from stemmacodicum.core.errors import ReportingError
from stemmacodicum.infrastructure.db.repos.verification_repo import VerificationRepo


@dataclass(slots=True)
class RunSummary:
    run_id: str
    policy_profile: str
    created_at: str
    total: int
    passed: int
    failed: int
    results: list[dict[str, object]]


class ReportingService:
    def __init__(self, verification_repo: VerificationRepo) -> None:
        self.verification_repo = verification_repo

    def build_run_summary(self, run_id: str) -> RunSummary:
        run = self.verification_repo.get_run(run_id)
        if run is None:
            raise ReportingError(f"Verification run not found: {run_id}")

        results = self.verification_repo.list_results(run_id)
        rows: list[dict[str, object]] = []
        passed = 0
        failed = 0

        for r in results:
            diagnostics = json.loads(r.diagnostics_json)
            if r.status == "pass":
                passed += 1
            else:
                failed += 1
            rows.append(
                {
                    "claim_id": r.claim_id,
                    "status": r.status,
                    "diagnostics": diagnostics,
                }
            )

        return RunSummary(
            run_id=run.id,
            policy_profile=run.policy_profile,
            created_at=run.created_at,
            total=len(results),
            passed=passed,
            failed=failed,
            results=rows,
        )

    def export_json_report(self, run_id: str, out_path: Path) -> Path:
        summary = self.build_run_summary(run_id)
        payload = {
            "run_id": summary.run_id,
            "policy_profile": summary.policy_profile,
            "created_at": summary.created_at,
            "total": summary.total,
            "passed": summary.passed,
            "failed": summary.failed,
            "results": summary.results,
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        return out_path

    def export_markdown_report(self, run_id: str, out_path: Path) -> Path:
        summary = self.build_run_summary(run_id)

        lines = [
            f"# Verification Report: {summary.run_id}",
            "",
            f"- Policy: `{summary.policy_profile}`",
            f"- Created: `{summary.created_at}`",
            f"- Total: `{summary.total}`",
            f"- Passed: `{summary.passed}`",
            f"- Failed: `{summary.failed}`",
            "",
            "## Results",
            "",
            "| Claim ID | Status | Reason |",
            "|---|---|---|",
        ]

        for row in summary.results:
            diagnostics = row["diagnostics"]
            reason = diagnostics.get("reason", "") if isinstance(diagnostics, dict) else ""
            lines.append(f"| {row['claim_id']} | {row['status']} | {reason} |")

        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return out_path
