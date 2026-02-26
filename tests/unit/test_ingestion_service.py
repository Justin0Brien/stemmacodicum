import json
import plistlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from stemmacodicum.application.services.ingestion_service import (
    IngestionService,
    _MACOS_WHEREFROMS_ATTR,
)
from stemmacodicum.core.errors import EmptySourceFileError
from stemmacodicum.infrastructure.archive.store import ArchiveStore
from stemmacodicum.infrastructure.db.repos.resource_repo import ResourceRepo
from stemmacodicum.infrastructure.db.sqlite import initialize_schema


def _bootstrap(
    tmp_path: Path,
    *,
    wayback_enabled: bool = False,
    wayback_async: bool = True,
) -> tuple[IngestionService, ResourceRepo]:
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
    service = IngestionService(
        resource_repo=repo,
        archive_store=ArchiveStore(tmp_path / "archive"),
        wayback_enabled=wayback_enabled,
        wayback_async=wayback_async,
    )
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

    first = service.ingest_file(source, source_uri="https://publisher.example/first.pdf")
    assert first.status == "ingested"
    assert first.resource.download_url == "https://publisher.example/first.pdf"

    second = service.ingest_file(source, source_uri="https://publisher.example/report.pdf")
    assert second.status == "duplicate"
    assert second.resource.download_url == "https://publisher.example/first.pdf"
    assert json.loads(second.resource.download_urls_json or "[]") == [
        "https://publisher.example/first.pdf",
        "https://publisher.example/report.pdf"
    ]

    persisted = repo.get_by_id(first.resource.id)
    assert persisted is not None
    assert persisted.download_url == "https://publisher.example/first.pdf"


def test_ingest_uses_xattr_cli_fallback_when_os_getxattr_unavailable(tmp_path: Path, monkeypatch) -> None:
    service, repo = _bootstrap(tmp_path)
    source = tmp_path / "cli-xattr.pdf"
    source.write_bytes(b"%PDF-1.4 fallback")
    where_froms = ["https://example.com/fallback/report.pdf"]
    plist_hex = plistlib.dumps(where_froms).hex()

    def fake_run(cmd, check=False, capture_output=True, text=True):
        if cmd[:3] == ["xattr", "-px", _MACOS_WHEREFROMS_ATTR]:
            return SimpleNamespace(returncode=0, stdout=plist_hex, stderr="")
        return SimpleNamespace(returncode=1, stdout="", stderr="missing")

    monkeypatch.setattr(IngestionService, "_is_macos", staticmethod(lambda: True))
    monkeypatch.setattr("stemmacodicum.application.services.ingestion_service.subprocess.run", fake_run)
    monkeypatch.delattr(
        "stemmacodicum.application.services.ingestion_service.os.getxattr",
        raising=False,
    )

    result = service.ingest_file(source)
    assert result.status == "ingested"
    assert result.resource.download_url == "https://example.com/fallback/report.pdf"

    persisted = repo.get_by_id(result.resource.id)
    assert persisted is not None
    assert persisted.download_url == "https://example.com/fallback/report.pdf"


def test_parse_raw_xattr_urls_handles_quoted_plaintext_arrays() -> None:
    raw = b'(\n  "https://example.com/report.pdf",\n  "https://example.com/reports"\n)\n'
    urls = IngestionService._parse_raw_xattr_urls(raw)
    assert urls == [
        "https://example.com/report.pdf",
        "https://example.com/reports",
    ]


def test_ingest_uses_source_path_fallback_for_url_discovery(tmp_path: Path, monkeypatch) -> None:
    service, repo = _bootstrap(tmp_path)
    staged = tmp_path / "staged-copy.pdf"
    staged.write_bytes(b"%PDF-1.4 staged")
    original = tmp_path / "original.pdf"
    original.write_bytes(b"%PDF-1.4 original")

    where_froms = [
        "https://example.com/files/original.pdf",
        "https://example.com/reports",
    ]

    def fake_getxattr(path: Path, attr: str) -> bytes:
        if path == original and attr == _MACOS_WHEREFROMS_ATTR:
            return plistlib.dumps(where_froms)
        raise OSError("missing")

    monkeypatch.setattr(IngestionService, "_is_macos", staticmethod(lambda: True))
    monkeypatch.setattr(
        "stemmacodicum.application.services.ingestion_service.os.getxattr",
        fake_getxattr,
        raising=False,
    )

    result = service.ingest_file(staged, source_paths=[original])
    assert result.status == "ingested"
    assert result.resource.download_url == "https://example.com/files/original.pdf"
    assert json.loads(result.resource.download_urls_json or "[]") == where_froms

    persisted = repo.get_by_id(result.resource.id)
    assert persisted is not None
    assert persisted.download_url == "https://example.com/files/original.pdf"


def test_ingest_uses_mdls_wherefroms_fallback(tmp_path: Path, monkeypatch) -> None:
    service, repo = _bootstrap(tmp_path)
    source = tmp_path / "mdls-fallback.pdf"
    source.write_bytes(b"%PDF-1.4 mdls")
    where_froms = ["https://example.com/mdls/report.pdf", "https://example.com/mdls/ref"]

    def fake_getxattr(path: Path, attr: str) -> bytes:
        raise OSError("missing")

    mdls_plist = plistlib.dumps({"kMDItemWhereFroms": where_froms})

    def fake_run(cmd, check=False, capture_output=True, text=False):
        if cmd[:3] == ["mdls", "-name", "kMDItemWhereFroms"]:
            return SimpleNamespace(returncode=0, stdout=mdls_plist, stderr=b"")
        return SimpleNamespace(returncode=1, stdout=b"", stderr=b"missing")

    monkeypatch.setattr(IngestionService, "_is_macos", staticmethod(lambda: True))
    monkeypatch.setattr(
        "stemmacodicum.application.services.ingestion_service.os.getxattr",
        fake_getxattr,
        raising=False,
    )
    monkeypatch.setattr("stemmacodicum.application.services.ingestion_service.subprocess.run", fake_run)

    result = service.ingest_file(source)
    assert result.status == "ingested"
    assert result.resource.download_url == "https://example.com/mdls/report.pdf"
    assert json.loads(result.resource.download_urls_json or "[]") == where_froms

    persisted = repo.get_by_id(result.resource.id)
    assert persisted is not None
    assert persisted.download_url == "https://example.com/mdls/report.pdf"


def test_ingest_rejects_zero_byte_file(tmp_path: Path) -> None:
    service, _repo = _bootstrap(tmp_path)
    source = tmp_path / "empty.pdf"
    source.write_bytes(b"")

    with pytest.raises(EmptySourceFileError):
        service.ingest_file(source, source_uri="https://example.com/empty.pdf")


def test_ingest_dispatches_wayback_registration_async(tmp_path: Path) -> None:
    service, _repo = _bootstrap(tmp_path, wayback_enabled=True, wayback_async=True)
    source = tmp_path / "async-wayback.pdf"
    source.write_bytes(b"%PDF-1.4 async")

    captured: list[tuple[object, tuple[object, ...], dict[str, object]]] = []

    class FakeExecutor:
        def submit(self, fn, *args, **kwargs):
            captured.append((fn, args, kwargs))
            return object()

    service._wayback_executor = FakeExecutor()  # type: ignore[assignment]
    service._probe_wayback_reachability = lambda force=False: {  # type: ignore[method-assign]
        "checked_epoch": 0.0,
        "reachable": True,
        "resolved_ip": "127.0.0.1",
        "error": None,
    }
    result = service.ingest_file(source, source_uri="https://www.openai.com/example/async-wayback.pdf")
    service._wayback_executor = None

    assert result.status == "ingested"
    assert len(captured) == 1
    submitted_fn, submitted_args, submitted_kwargs = captured[0]
    assert submitted_fn == service._register_with_wayback_request
    assert submitted_args == ("https://www.openai.com/example/async-wayback.pdf",)
    assert submitted_kwargs == {}
