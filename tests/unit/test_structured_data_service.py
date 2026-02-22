from __future__ import annotations

from pathlib import Path

from stemmacodicum.application.services.ingestion_policy_service import IngestionPolicyService
from stemmacodicum.application.services.ingestion_service import IngestionService
from stemmacodicum.application.services.structured_data_service import StructuredDataService
from stemmacodicum.infrastructure.archive.store import ArchiveStore
from stemmacodicum.infrastructure.db.repos.resource_repo import ResourceRepo
from stemmacodicum.infrastructure.db.repos.structured_data_repo import StructuredDataRepo
from stemmacodicum.infrastructure.db.sqlite import initialize_schema


def _bootstrap(tmp_path: Path) -> tuple[ResourceRepo, StructuredDataService, IngestionService]:
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
    structured_repo = StructuredDataRepo(db_path)
    archive_dir = tmp_path / "archive"
    service = StructuredDataService(
        resource_repo=resource_repo,
        structured_repo=structured_repo,
        archive_dir=archive_dir,
        policy_service=IngestionPolicyService(),
        max_profile_rows=2000,
        sample_rows=10,
    )
    ingest = IngestionService(resource_repo=resource_repo, archive_store=ArchiveStore(archive_dir))
    return resource_repo, service, ingest


def test_profile_csv_with_metadata_preamble_and_lookup(tmp_path: Path) -> None:
    _resource_repo, service, ingest = _bootstrap(tmp_path)
    source = tmp_path / "hesa-table.csv"
    source.write_text(
        "\n".join(
            [
                "Title,Graduate activities",
                "Data source,HESA",
                "",
                "UKPRN,Academic year,Activity,Number",
                "10008071,2022/23,Total,30",
                "10008072,2022/23,Total,44",
            ]
        ),
        encoding="utf-8",
    )

    resource = ingest.ingest_file(source).resource
    profile = service.profile_resource(resource.id)
    assert profile.status == "success"
    assert profile.table_count == 1
    assert profile.row_count_observed == 2

    catalog = service.get_catalog(resource.id)
    assert catalog is not None
    assert catalog["data_format"] == "csv"

    tables = service.list_tables(resource.id)
    assert len(tables) == 1
    assert tables[0]["columns"] == ["UKPRN", "Academic year", "Activity", "Number"]

    match = service.resolve_data_cell(
        resource.id,
        {
            "value_column": "Number",
            "filters": {
                "UKPRN": "10008072",
                "Academic year": "2022/23",
                "Activity": "Total",
            },
        },
    )
    assert match.value_raw == "44"
    assert match.column_name == "Number"
    assert match.column_index == 4
    assert match.row_number >= 5


def test_profile_non_structured_resource_is_skipped(tmp_path: Path) -> None:
    _resource_repo, service, ingest = _bootstrap(tmp_path)
    source = tmp_path / "note.md"
    source.write_text("# narrative", encoding="utf-8")
    resource = ingest.ingest_file(source).resource
    profile = service.profile_resource(resource.id)
    assert profile.status == "skipped"
    assert "not classified as structured data" in str(profile.error)
