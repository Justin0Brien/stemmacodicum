import json
import plistlib
from pathlib import Path

from stemmacodicum.application.services.ingestion_service import (
    IngestionService,
    _MACOS_WHEREFROMS_ATTR,
)
from stemmacodicum.infrastructure.archive.store import ArchiveStore
from stemmacodicum.infrastructure.db.repos.resource_repo import ResourceRepo
from stemmacodicum.infrastructure.db.sqlite import initialize_schema


def _bootstrap(tmp_path: Path) -> tuple[IngestionService, ResourceRepo]:
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
    repo = ResourceRepo(db_path)
    service = IngestionService(resource_repo=repo, archive_store=ArchiveStore(tmp_path / "archive"))
    return service, repo


def test_ingest_captures_macos_download_url_metadata(tmp_path: Path, monkeypatch) -> None:
    service, repo = _bootstrap(tmp_path)
    source = tmp_path / "statement.pdf"
    source.write_bytes(b"%PDF-1.4 fake")

    where_froms = [
        "https://example.com/files/statement.pdf",
        "https://example.com/reports",
    ]

    def fake_getxattr(path: Path, attr: str) -> bytes:
        if attr == _MACOS_WHEREFROMS_ATTR:
            return plistlib.dumps(where_froms)
        raise OSError("missing")

    monkeypatch.setattr(IngestionService, "_is_macos", staticmethod(lambda: True))
    monkeypatch.setattr("stemmacodicum.application.services.ingestion_service.os.getxattr", fake_getxattr, raising=False)

    result = service.ingest_file(source)

    assert result.status == "ingested"
    assert result.resource.download_url == "https://example.com/files/statement.pdf"
    assert json.loads(result.resource.download_urls_json or "[]") == where_froms

    persisted = repo.get_by_id(result.resource.id)
    assert persisted is not None
    assert persisted.download_url == "https://example.com/files/statement.pdf"
    assert json.loads(persisted.download_urls_json or "[]") == where_froms


def test_duplicate_ingest_backfills_missing_download_url_metadata(tmp_path: Path) -> None:
    service, repo = _bootstrap(tmp_path)
    source = tmp_path / "duplicate.pdf"
    source.write_bytes(b"%PDF-1.4 duplicate")

    first = service.ingest_file(source, source_uri=f"upload:{source.name}")
    assert first.status == "ingested"
    assert first.resource.download_url is None
    assert first.resource.download_urls_json is None

    second = service.ingest_file(source, source_uri="https://publisher.example/report.pdf")
    assert second.status == "duplicate"
    assert second.resource.download_url == "https://publisher.example/report.pdf"
    assert json.loads(second.resource.download_urls_json or "[]") == [
        "https://publisher.example/report.pdf"
    ]

    persisted = repo.get_by_id(first.resource.id)
    assert persisted is not None
    assert persisted.download_url == "https://publisher.example/report.pdf"
