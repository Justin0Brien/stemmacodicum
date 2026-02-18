from pathlib import Path

from stemmacodicum.application.services.extraction_service import ExtractionService
from stemmacodicum.application.services.ingestion_service import IngestionService
from stemmacodicum.infrastructure.archive.store import ArchiveStore
from stemmacodicum.infrastructure.db.repos.extraction_repo import ExtractionRepo
from stemmacodicum.infrastructure.db.repos.resource_repo import ResourceRepo
from stemmacodicum.infrastructure.db.sqlite import initialize_schema
from stemmacodicum.infrastructure.parsers.docling_adapter import ParsedCell, ParsedTable


def _bootstrap(tmp_path: Path) -> tuple[ResourceRepo, ExtractionRepo, Path]:
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
    return ResourceRepo(db_path), ExtractionRepo(db_path), tmp_path / "archive"


def test_table_id_is_deterministic() -> None:
    t1 = ParsedTable(
        page_index=0,
        caption="Table 5",
        row_headers=["A", "B"],
        col_headers=["Col1", "Col2"],
        cells=[ParsedCell(row_index=0, col_index=0, value="1")],
        bbox=None,
    )
    t2 = ParsedTable(
        page_index=0,
        caption="Table 5",
        row_headers=["A", "B"],
        col_headers=["Col1", "Col2"],
        cells=[ParsedCell(row_index=1, col_index=1, value="2")],
        bbox=None,
    )

    assert ExtractionService.derive_table_id(t1) == ExtractionService.derive_table_id(t2)


def test_extract_resource_persists_tables_and_text_layers(tmp_path: Path) -> None:
    resource_repo, extraction_repo, archive_dir = _bootstrap(tmp_path)

    source = tmp_path / "report.md"
    source.write_text(
        """
| Item | Value |
|---|---:|
| Cash | 5631 |
""",
        encoding="utf-8",
    )

    ingest = IngestionService(resource_repo=resource_repo, archive_store=ArchiveStore(archive_dir))
    ingested = ingest.ingest_file(source)

    service = ExtractionService(
        resource_repo=resource_repo,
        extraction_repo=extraction_repo,
        archive_dir=archive_dir,
    )
    summary = service.extract_resource(ingested.resource.id)

    assert summary.tables_found == 1
    assert summary.text_chars > 0
    assert summary.segments_persisted > 0
    assert summary.annotations_persisted > 0

    tables = extraction_repo.list_tables_for_resource(ingested.resource.id)
    assert len(tables) == 1
    assert tables[0].table_id.startswith("sha256:")

    doc_text = service.get_document_text(ingested.resource.id)
    assert doc_text is not None
    assert "Cash" in doc_text.text_content
    assert doc_text.char_count == len(doc_text.text_content)

    segments = service.list_segments(ingested.resource.id, limit=100)
    assert any(s.segment_type == "layout:document" for s in segments)
    assert any(s.segment_type == "structure:sentence" for s in segments)

    annotations = service.list_annotations(ingested.resource.id, limit=200)
    assert any(a["category"] == "quantity" for a in annotations)
    assert any(a["category"] == "metric" for a in annotations)

    dump = service.build_dump(ingested.resource.id)
    assert dump["document_text"] is not None
    assert len(dump["tables"]) == 1
