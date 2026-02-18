from pathlib import Path

from stemmacodicum.application.services.extraction_service import ExtractionService
from stemmacodicum.application.services.ingestion_service import IngestionService
from stemmacodicum.application.services.pipeline_service import FinancialPipelineService
from stemmacodicum.infrastructure.archive.store import ArchiveStore
from stemmacodicum.infrastructure.db.repos.extraction_repo import ExtractionRepo
from stemmacodicum.infrastructure.db.repos.resource_repo import ResourceRepo
from stemmacodicum.infrastructure.db.sqlite import initialize_schema


def test_financial_pipeline_filters_and_processes(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()

    # Financial candidate
    f1 = root / "annual_report_2025.pdf"
    f1.write_text("fake pdf bytes", encoding="utf-8")

    # Non-financial file
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

    assert stats.candidates == 1
    assert stats.processed == 1
    assert stats.ingested == 1
    assert stats.failed == 0
