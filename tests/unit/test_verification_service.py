import json
from pathlib import Path

from stemmacodicum.application.services.claim_service import ClaimService
from stemmacodicum.application.services.evidence_binding_service import EvidenceBindingService
from stemmacodicum.application.services.extraction_service import ExtractionService
from stemmacodicum.application.services.ingestion_service import IngestionService
from stemmacodicum.application.services.verification_service import VerificationService
from stemmacodicum.infrastructure.archive.store import ArchiveStore
from stemmacodicum.infrastructure.db.repos.claim_repo import ClaimRepo
from stemmacodicum.infrastructure.db.repos.evidence_repo import EvidenceRepo
from stemmacodicum.infrastructure.db.repos.extraction_repo import ExtractionRepo
from stemmacodicum.infrastructure.db.repos.resource_repo import ResourceRepo
from stemmacodicum.infrastructure.db.repos.verification_repo import VerificationRepo
from stemmacodicum.infrastructure.db.sqlite import initialize_schema


def _bootstrap(tmp_path: Path):
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

    claim_repo = ClaimRepo(db_path)
    resource_repo = ResourceRepo(db_path)
    evidence_repo = EvidenceRepo(db_path)
    extraction_repo = ExtractionRepo(db_path)
    verification_repo = VerificationRepo(db_path)

    bind_service = EvidenceBindingService(claim_repo, resource_repo, evidence_repo)
    verify_service = VerificationService(
        claim_repo=claim_repo,
        evidence_repo=evidence_repo,
        extraction_repo=extraction_repo,
        verification_repo=verification_repo,
        binding_service=bind_service,
    )

    return claim_repo, resource_repo, extraction_repo, bind_service, verify_service, tmp_path / "archive"


def test_verify_quantitative_claim_pass(tmp_path: Path) -> None:
    claim_repo, resource_repo, extraction_repo, bind_service, verify_service, archive_dir = _bootstrap(tmp_path)

    claims = tmp_path / "claims.json"
    claims.write_text(
        json.dumps(
            {
                "claims": [
                    {
                        "claim_type": "quantitative",
                        "subject": "Institution",
                        "predicate": "cash_at_bank",
                        "value_raw": "5631",
                        "currency": "GBP",
                        "period_label": "FY2024/25",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    claim_service = ClaimService(claim_repo)
    claim_service.import_claims(claims, "json", "verify-set")
    claim = claim_service.list_claims("verify-set")[0]

    source = tmp_path / "table.md"
    source.write_text(
        """
Table 7: Liquidity
| Metric | FY2024/25 |
|---|---:|
| Cash at bank | 5631 |
""",
        encoding="utf-8",
    )

    ingest = IngestionService(resource_repo=resource_repo, archive_store=ArchiveStore(archive_dir))
    resource = ingest.ingest_file(source).resource

    extraction = ExtractionService(resource_repo=resource_repo, extraction_repo=extraction_repo, archive_dir=archive_dir)
    extraction.extract_resource(resource.id)
    table = extraction_repo.list_tables_for_resource(resource.id, limit=1)[0]

    selectors_value = [
        {"type": "PageGeometrySelector", "pageIndex": 0, "boxes": []},
        {
            "type": "TableAddressSelector",
            "table_id": table.table_id,
            "cell_ref": {"row_index": 0, "col_index": 1},
            "units": {"currency": "GBP"},
            "period": {"label": "FY2024/25"},
        },
    ]

    for role in ["value-cell", "row-header", "column-header", "caption"]:
        selectors = selectors_value if role == "value-cell" else [
            {"type": "PageGeometrySelector", "pageIndex": 0, "boxes": []},
            {"type": "TextQuoteSelector", "exact": role},
        ]
        bind_service.bind_evidence(
            claim_id=claim.id,
            resource_id=resource.id,
            role=role,
            selectors=selectors,
        )

    outcome = verify_service.verify_claim(claim.id)
    assert outcome.status == "pass"


def test_verify_quantitative_claim_fails_on_value_mismatch(tmp_path: Path) -> None:
    claim_repo, resource_repo, extraction_repo, bind_service, verify_service, archive_dir = _bootstrap(tmp_path)

    claims = tmp_path / "claims.json"
    claims.write_text(
        json.dumps(
            {
                "claims": [
                    {
                        "claim_type": "quantitative",
                        "subject": "Institution",
                        "predicate": "cash_at_bank",
                        "value_raw": "9999",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    claim_service = ClaimService(claim_repo)
    claim_service.import_claims(claims, "json", "verify-set-b")
    claim = claim_service.list_claims("verify-set-b")[0]

    source = tmp_path / "table.md"
    source.write_text(
        "|Metric|FY2024/25|\n|---|---:|\n|Cash at bank|5631|\n",
        encoding="utf-8",
    )

    ingest = IngestionService(resource_repo=resource_repo, archive_store=ArchiveStore(archive_dir))
    resource = ingest.ingest_file(source).resource

    extraction = ExtractionService(resource_repo=resource_repo, extraction_repo=extraction_repo, archive_dir=archive_dir)
    extraction.extract_resource(resource.id)
    table = extraction_repo.list_tables_for_resource(resource.id, limit=1)[0]

    for role in ["value-cell", "row-header", "column-header", "caption"]:
        selectors = [
            {"type": "PageGeometrySelector", "pageIndex": 0, "boxes": []},
            {"type": "TableAddressSelector", "table_id": table.table_id, "cell_ref": {"row_index": 0, "col_index": 1}}
            if role == "value-cell"
            else {"type": "TextQuoteSelector", "exact": role},
        ]
        bind_service.bind_evidence(claim.id, resource.id, role, selectors)

    outcome = verify_service.verify_claim(claim.id)
    assert outcome.status == "fail"
    assert outcome.diagnostics.get("reason") == "value_mismatch"
