from pathlib import Path

from stemmacodicum.application.services.ingestion_service import IngestionService
from stemmacodicum.application.services.reference_service import ReferenceService
from stemmacodicum.infrastructure.archive.store import ArchiveStore
from stemmacodicum.infrastructure.db.repos.citation_repo import CitationRepo
from stemmacodicum.infrastructure.db.repos.reference_repo import ReferenceRepo
from stemmacodicum.infrastructure.db.repos.resource_repo import ResourceRepo
from stemmacodicum.infrastructure.db.sqlite import get_connection, initialize_schema


def _bootstrap(tmp_path: Path) -> tuple[ReferenceService, ResourceRepo, Path, Path]:
    db_path = tmp_path / "stemma.db"
    archive_dir = tmp_path / "archive"
    schema_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "stemmacodicum"
        / "infrastructure"
        / "db"
        / "schema.sql"
    )
    initialize_schema(db_path, schema_path)

    citation_repo = CitationRepo(db_path)
    reference_repo = ReferenceRepo(db_path)
    resource_repo = ResourceRepo(db_path)

    service = ReferenceService(
        citation_repo=citation_repo,
        reference_repo=reference_repo,
        resource_repo=resource_repo,
    )
    return service, resource_repo, db_path, archive_dir


def test_import_bibtex_creates_and_updates_records(tmp_path: Path) -> None:
    service, _, _, _ = _bootstrap(tmp_path)

    bib_path = tmp_path / "refs.bib"
    bib_path.write_text(
        """
@article{AlphaKey,
  title={Alpha Title},
  author={One Author},
  year={2020}
}
@book{BetaKey,
  title={Beta Book},
  author={Second Author},
  year={2021},
  url={https://example.org/beta}
}
""",
        encoding="utf-8",
    )

    summary_first = service.import_bibtex(bib_path)
    assert summary_first.entries_seen == 2
    assert summary_first.mappings_created == 2
    assert summary_first.references_inserted == 2
    assert summary_first.references_updated == 0

    summary_second = service.import_bibtex(bib_path)
    assert summary_second.entries_seen == 2
    assert summary_second.mappings_created == 0
    assert summary_second.references_inserted == 0
    assert summary_second.references_updated == 2


def test_link_reference_to_resource(tmp_path: Path) -> None:
    service, resource_repo, db_path, archive_dir = _bootstrap(tmp_path)

    bib_path = tmp_path / "refs.bib"
    bib_path.write_text(
        """
@article{GammaKey,
  title={Gamma Title},
  year={2022}
}
""",
        encoding="utf-8",
    )
    service.import_bibtex(bib_path)

    source_file = tmp_path / "source.pdf"
    source_file.write_text("dummy content", encoding="utf-8")

    ingest = IngestionService(
        resource_repo=resource_repo,
        archive_store=ArchiveStore(archive_dir),
    )
    result = ingest.ingest_file(source_file)

    citation = CitationRepo(db_path).get_by_normalized_key("gammakey")
    assert citation is not None

    service.link_reference_to_resource(citation.cite_id, result.resource.digest_sha256)

    with get_connection(db_path) as conn:
        linked = conn.execute("SELECT COUNT(*) AS c FROM reference_resources").fetchone()
    assert linked["c"] == 1
