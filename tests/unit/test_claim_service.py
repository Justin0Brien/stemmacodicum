import json
from pathlib import Path

from stemmacodicum.application.services.claim_service import ClaimService
from stemmacodicum.infrastructure.db.repos.claim_repo import ClaimRepo
from stemmacodicum.infrastructure.db.sqlite import initialize_schema


def _service(tmp_path: Path) -> ClaimService:
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
    return ClaimService(ClaimRepo(db_path))


def test_import_csv_quantitative_claims(tmp_path: Path) -> None:
    service = _service(tmp_path)

    source = tmp_path / "claims.csv"
    source.write_text(
        "subject,predicate,value_raw,currency,period_label,claim_type\n"
        "City St George's University,cash_at_bank,5631,GBP,FY2024/25,quantitative\n",
        encoding="utf-8",
    )

    summary = service.import_claims(source, "csv", "financial-report")
    assert summary.imported == 1

    claims = service.list_claims("financial-report")
    assert len(claims) == 1
    assert claims[0].claim_type == "quantitative"
    assert claims[0].value_raw == "5631"


def test_import_json_narrative_claims(tmp_path: Path) -> None:
    service = _service(tmp_path)

    source = tmp_path / "claims.json"
    source.write_text(
        json.dumps(
            {
                "claims": [
                    {
                        "claim_type": "narrative",
                        "narrative_text": "The report asserts sustained liquidity improvement.",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    summary = service.import_claims(source, "json", "narratives")
    assert summary.imported == 1

    claims = service.list_claims("narratives")
    assert len(claims) == 1
    assert claims[0].narrative_text is not None


def test_import_markdown_bullets_as_narrative_claims(tmp_path: Path) -> None:
    service = _service(tmp_path)

    source = tmp_path / "claims.md"
    source.write_text(
        """
- This is a narrative claim about governance quality. @Ab12
- This is another claim without citation.
""",
        encoding="utf-8",
    )

    summary = service.import_claims(source, "markdown", "bullets")
    assert summary.imported == 2

    claims = service.list_claims("bullets")
    assert len(claims) == 2
    assert any(c.source_cite_id == "Ab12" for c in claims)
