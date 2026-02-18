import json
from pathlib import Path

from stemmacodicum.application.services.reporting_service import ReportingService
from stemmacodicum.core.ids import new_uuid
from stemmacodicum.core.time import now_utc_iso
from stemmacodicum.domain.models.claim import Claim, ClaimSet
from stemmacodicum.domain.models.verification import VerificationResult, VerificationRun
from stemmacodicum.infrastructure.db.repos.claim_repo import ClaimRepo
from stemmacodicum.infrastructure.db.repos.verification_repo import VerificationRepo
from stemmacodicum.infrastructure.db.sqlite import initialize_schema


def test_reporting_summary_and_exports(tmp_path: Path) -> None:
    db_path = tmp_path / "stemma.db"
    schema_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "stemmacodicum"
        / "infrastructure"
        / "db"
        / "schema.sql"
    )
    initialize_schema(db_path, schema_path)

    # Seed claims referenced by verification results.
    claim_repo = ClaimRepo(db_path)
    claim_set_id = new_uuid()
    now = now_utc_iso()
    claim_repo.insert_claim_set(
        ClaimSet(id=claim_set_id, name="reporting-test", description=None, created_at=now)
    )
    claim_repo.insert_claim(
        Claim(
            id="c1",
            claim_set_id=claim_set_id,
            claim_type="narrative",
            subject=None,
            predicate=None,
            object_text=None,
            narrative_text="a",
            value_raw=None,
            value_parsed=None,
            currency=None,
            scale_factor=None,
            period_label=None,
            source_cite_id=None,
            status="active",
            created_at=now,
            updated_at=now,
        )
    )
    claim_repo.insert_claim(
        Claim(
            id="c2",
            claim_set_id=claim_set_id,
            claim_type="narrative",
            subject=None,
            predicate=None,
            object_text=None,
            narrative_text="b",
            value_raw=None,
            value_parsed=None,
            currency=None,
            scale_factor=None,
            period_label=None,
            source_cite_id=None,
            status="active",
            created_at=now,
            updated_at=now,
        )
    )

    repo = VerificationRepo(db_path)
    run = VerificationRun(
        id=new_uuid(),
        claim_set_id=None,
        policy_profile="strict",
        created_at=now_utc_iso(),
    )
    repo.insert_run(run)

    repo.insert_result(
        VerificationResult(
            id=new_uuid(),
            run_id=run.id,
            claim_id="c1",
            status="pass",
            diagnostics_json=json.dumps({"reason": "ok"}),
            created_at=now_utc_iso(),
        )
    )
    repo.insert_result(
        VerificationResult(
            id=new_uuid(),
            run_id=run.id,
            claim_id="c2",
            status="fail",
            diagnostics_json=json.dumps({"reason": "value_mismatch"}),
            created_at=now_utc_iso(),
        )
    )

    service = ReportingService(repo)
    summary = service.build_run_summary(run.id)
    assert summary.total == 2
    assert summary.passed == 1
    assert summary.failed == 1

    json_out = tmp_path / "report.json"
    md_out = tmp_path / "report.md"
    service.export_json_report(run.id, json_out)
    service.export_markdown_report(run.id, md_out)

    assert json.loads(json_out.read_text(encoding="utf-8"))["failed"] == 1
    assert "value_mismatch" in md_out.read_text(encoding="utf-8")
