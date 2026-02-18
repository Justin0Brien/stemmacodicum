import json
from pathlib import Path

from stemmacodicum.application.services.claim_service import ClaimService
from stemmacodicum.application.services.evidence_binding_service import EvidenceBindingService
from stemmacodicum.application.services.ingestion_service import IngestionService
from stemmacodicum.infrastructure.archive.store import ArchiveStore
from stemmacodicum.infrastructure.db.repos.claim_repo import ClaimRepo
from stemmacodicum.infrastructure.db.repos.evidence_repo import EvidenceRepo
from stemmacodicum.infrastructure.db.repos.resource_repo import ResourceRepo
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

    return claim_repo, resource_repo, evidence_repo, tmp_path / "archive"


def test_binding_validation_requires_roles_for_quantitative_claim(tmp_path: Path) -> None:
    claim_repo, resource_repo, evidence_repo, archive_dir = _bootstrap(tmp_path)

    claims_file = tmp_path / "claims.json"
    claims_file.write_text(
        json.dumps(
            {
                "claims": [
                    {
                        "claim_type": "quantitative",
                        "subject": "Institution",
                        "predicate": "cash_at_bank",
                        "value_raw": "5631",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    claim_service = ClaimService(claim_repo)
    claim_service.import_claims(claims_file, "json", "set-a")
    claim = claim_service.list_claims("set-a")[0]

    source = tmp_path / "source.md"
    source.write_text("|A|B|\n|---|---|\n|x|y|\n", encoding="utf-8")
    ingest = IngestionService(resource_repo=resource_repo, archive_store=ArchiveStore(archive_dir))
    resource = ingest.ingest_file(source).resource

    bind = EvidenceBindingService(
        claim_repo=claim_repo,
        resource_repo=resource_repo,
        evidence_repo=evidence_repo,
    )

    bind.bind_evidence(
        claim_id=claim.id,
        resource_id=resource.id,
        role="value-cell",
        selectors=[
            {"type": "PageGeometrySelector", "pageIndex": 0, "boxes": []},
            {"type": "TextQuoteSelector", "exact": "5631"},
        ],
    )

    result = bind.validate_binding(claim.id)
    assert result.ok is False
    assert set(result.missing_roles) == {"row-header", "column-header", "caption"}


def test_binding_validation_passes_with_required_roles_and_selector_diversity(tmp_path: Path) -> None:
    claim_repo, resource_repo, evidence_repo, archive_dir = _bootstrap(tmp_path)

    claims_file = tmp_path / "claims.json"
    claims_file.write_text(
        json.dumps(
            {
                "claims": [
                    {
                        "claim_type": "quantitative",
                        "subject": "Institution",
                        "predicate": "cash_at_bank",
                        "value_raw": "5631",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    claim_service = ClaimService(claim_repo)
    claim_service.import_claims(claims_file, "json", "set-b")
    claim = claim_service.list_claims("set-b")[0]

    source = tmp_path / "source.md"
    source.write_text("|A|B|\n|---|---|\n|x|y|\n", encoding="utf-8")
    ingest = IngestionService(resource_repo=resource_repo, archive_store=ArchiveStore(archive_dir))
    resource = ingest.ingest_file(source).resource

    bind = EvidenceBindingService(
        claim_repo=claim_repo,
        resource_repo=resource_repo,
        evidence_repo=evidence_repo,
    )

    for role in ["value-cell", "row-header", "column-header", "caption"]:
        bind.bind_evidence(
            claim_id=claim.id,
            resource_id=resource.id,
            role=role,
            selectors=[
                {"type": "PageGeometrySelector", "pageIndex": 0, "boxes": []},
                {"type": "TextQuoteSelector", "exact": role},
            ],
        )

    result = bind.validate_binding(claim.id)
    assert result.ok is True
    assert result.missing_roles == []
    assert result.evidence_with_too_few_selectors == []
