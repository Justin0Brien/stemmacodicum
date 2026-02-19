from pathlib import Path

from stemmacodicum.application.services.health_service import HealthService
from stemmacodicum.application.services.ingestion_service import IngestionService
from stemmacodicum.infrastructure.archive.store import ArchiveStore
from stemmacodicum.infrastructure.db.repos.resource_repo import ResourceRepo
from stemmacodicum.infrastructure.db.sqlite import initialize_schema


def test_doctor_passes_for_basic_clean_state(tmp_path: Path) -> None:
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

    archive_dir = tmp_path / "archive"
    source = tmp_path / "source.txt"
    source.write_text("ok", encoding="utf-8")

    ingest = IngestionService(ResourceRepo(db_path), ArchiveStore(archive_dir))
    ingest.ingest_file(source)

    report = HealthService(db_path=db_path, archive_dir=archive_dir).run_doctor()
    assert report.ok is True
    assert report.checks_run == 5
    assert report.db_runtime["journal_mode"] == "wal"
    assert report.db_runtime["foreign_keys"] is True
    assert int(report.db_runtime["busy_timeout_ms"]) >= 30_000
    assert report.issues == []
