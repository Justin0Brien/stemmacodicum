import json
from pathlib import Path

from stemmacodicum.application.services.claim_service import ClaimService
from stemmacodicum.application.services.evidence_binding_service import EvidenceBindingService
from stemmacodicum.application.services.ingestion_service import IngestionService
from stemmacodicum.application.services.reference_service import ReferenceService
from stemmacodicum.application.services.trace_service import TraceService
from stemmacodicum.infrastructure.archive.store import ArchiveStore
from stemmacodicum.infrastructure.db.repos.citation_repo import CitationRepo
from stemmacodicum.infrastructure.db.repos.claim_repo import ClaimRepo
from stemmacodicum.infrastructure.db.repos.evidence_repo import EvidenceRepo
from stemmacodicum.infrastructure.db.repos.reference_repo import ReferenceRepo
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

    return {
        "db_path": db_path,
        "claim_repo": ClaimRepo(db_path),
        "resource_repo": ResourceRepo(db_path),
        "evidence_repo": EvidenceRepo(db_path),
        "citation_repo": CitationRepo(db_path),
        "reference_repo": ReferenceRepo(db_path),
        "archive_dir": tmp_path / "archive",
    }


def test_trace_claim_and_resource_and_citation(tmp_path: Path) -> None:
    ctx = _bootstrap(tmp_path)

    # Claim + resource + binding
    claim_file = tmp_path / "claims.json"
    claim_file.write_text(
        json.dumps({"claims": [{"claim_type": "narrative", "narrative_text": "Liquidity improved"}]}),
        encoding="utf-8",
    )

    claim_service = ClaimService(ctx["claim_repo"])
    claim_service.import_claims(claim_file, "json", "trace-set")
    claim = claim_service.list_claims("trace-set")[0]

    source = tmp_path / "source.md"
    source.write_text("Evidence text", encoding="utf-8")
    ingest = IngestionService(ctx["resource_repo"], ArchiveStore(ctx["archive_dir"]))
    resource = ingest.ingest_file(source).resource

    bind = EvidenceBindingService(ctx["claim_repo"], ctx["resource_repo"], ctx["evidence_repo"])
    bind.bind_evidence(
        claim_id=claim.id,
        resource_id=resource.id,
        role="quote",
        selectors=[
            {"type": "PageGeometrySelector", "pageIndex": 0, "boxes": []},
            {"type": "TextQuoteSelector", "exact": "Liquidity"},
        ],
    )

    # Citation + reference + link to resource
    bib = tmp_path / "refs.bib"
    bib.write_text(
        """
@article{TraceKey,
  title={Trace Test},
  year={2024},
  url={https://example.org/trace}
}
""",
        encoding="utf-8",
    )
    ref_service = ReferenceService(ctx["citation_repo"], ctx["reference_repo"], ctx["resource_repo"])
    ref_service.import_bibtex(bib)
    cite = ctx["citation_repo"].get_by_normalized_key("tracekey")
    assert cite is not None
    ref_service.link_reference_to_resource(cite.cite_id, resource.digest_sha256)

    trace = TraceService(ctx["db_path"])

    claim_trace = trace.trace_claim(claim.id)
    assert claim_trace.claim_id == claim.id
    assert len(claim_trace.evidence) == 1

    resource_trace = trace.trace_resource(resource_id=resource.id)
    assert resource_trace.resource_id == resource.id
    assert len(resource_trace.references) == 1
    assert len(resource_trace.claims) == 1

    citation_trace = trace.trace_citation(cite.cite_id)
    assert citation_trace.cite_id == cite.cite_id
    assert citation_trace.reference is not None
    assert len(citation_trace.resources) == 1
