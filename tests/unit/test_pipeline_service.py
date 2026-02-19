from pathlib import Path

from stemmacodicum.application.services.extraction_service import ExtractionService
from stemmacodicum.application.services.ingestion_service import IngestionService
from stemmacodicum.application.services.pipeline_service import FinancialPipelineService
from stemmacodicum.infrastructure.archive.store import ArchiveStore
from stemmacodicum.infrastructure.db.repos.extraction_repo import ExtractionRepo
from stemmacodicum.infrastructure.db.repos.resource_repo import ResourceRepo
from stemmacodicum.infrastructure.db.sqlite import initialize_schema


def test_mass_import_pipeline_recurses_and_processes_supported_files(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()

    # Candidate
    f1 = root / "annual_report_2025.pdf"
    f1.write_text("fake pdf bytes", encoding="utf-8")

    # Also candidate (supported extension, no keyword requirement)
    f2 = root / "random_note.txt"
    f2.write_text("hello", encoding="utf-8")

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

    resource_repo = ResourceRepo(db_path)
    extraction_repo = ExtractionRepo(db_path)
    archive_dir = tmp_path / "archive"

    ingestion = IngestionService(resource_repo=resource_repo, archive_store=ArchiveStore(archive_dir))
    extraction = ExtractionService(resource_repo=resource_repo, extraction_repo=extraction_repo, archive_dir=archive_dir)

    service = FinancialPipelineService(
        ingestion_service=ingestion,
        extraction_service=extraction,
        extraction_repo=extraction_repo,
        state_path=tmp_path / "state.json",
        log_path=tmp_path / "run.log.jsonl",
    )

    stats = service.run(root=root, max_files=10, run_extraction=False)

    assert stats.candidates == 2
    assert stats.already_processed == 0
    assert stats.processed == 2
    assert stats.ingested == 2
    assert stats.failed == 0
    assert stats.remaining_unprocessed == 0


def test_financial_pipeline_reports_already_processed(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()

    f1 = root / "annual_report_2025.pdf"
    f1.write_text("fake pdf bytes", encoding="utf-8")

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

    resource_repo = ResourceRepo(db_path)
    extraction_repo = ExtractionRepo(db_path)
    archive_dir = tmp_path / "archive"

    ingestion = IngestionService(resource_repo=resource_repo, archive_store=ArchiveStore(archive_dir))
    extraction = ExtractionService(resource_repo=resource_repo, extraction_repo=extraction_repo, archive_dir=archive_dir)

    service = FinancialPipelineService(
        ingestion_service=ingestion,
        extraction_service=extraction,
        extraction_repo=extraction_repo,
        state_path=tmp_path / "state.json",
        log_path=tmp_path / "run.log.jsonl",
    )

    first = service.run(root=root, max_files=10, run_extraction=False)
    second = service.run(root=root, max_files=10, run_extraction=False)

    assert first.processed == 1
    assert second.processed == 0
    assert second.already_processed == 1
    assert second.state_entries_before == 1
    assert second.state_entries_after == 1
