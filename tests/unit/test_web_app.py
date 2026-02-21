import json
from pathlib import Path
import time

from fastapi.testclient import TestClient

from stemmacodicum.core.config import AppPaths
from stemmacodicum.infrastructure.db.sqlite import get_connection
from stemmacodicum.web.app import create_app


def test_web_app_end_to_end_smoke(tmp_path: Path) -> None:
    project_root = tmp_path / "proj"
    project_root.mkdir(parents=True, exist_ok=True)
    paths = AppPaths(
        project_root=project_root,
        stemma_dir=project_root / ".stemma",
        db_path=project_root / ".stemma" / "stemma.db",
        archive_dir=project_root / ".stemma" / "archive",
        vector_dir=project_root / ".stemma" / "vector",
        qdrant_dir=project_root / ".stemma" / "vector" / "qdrant",
    )

    app = create_app(paths)
    client = TestClient(app)

    # Init
    r = client.post("/api/init")
    assert r.status_code == 200

    # Ingest by path
    source = tmp_path / "report.md"
    source.write_text("| Item | Value |\n|---|---:|\n| Cash | 5631 |\n", encoding="utf-8")
    r = client.post("/api/ingest/path", json={"path": str(source)})
    assert r.status_code == 200
    ingest_payload = r.json()
    assert ingest_payload["status"] in {"ingested", "duplicate"}
    assert "extraction" in ingest_payload
    assert ingest_payload["extraction"]["status"] in {"extracted", "skipped", "failed"}
    assert "download_url" in ingest_payload["resource"]
    assert "download_urls_json" in ingest_payload["resource"]

    # Upload ingest and dedupe check
    files = {"file": ("upload.txt", b"hello world", "text/plain")}
    r1 = client.post("/api/ingest/upload", files=files)
    r2 = client.post("/api/ingest/upload", files=files)
    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r1.json()["status"] in {"ingested", "duplicate"}
    assert r2.json()["status"] in {"ingested", "duplicate"}
    second_upload = r2.json()
    assert "extraction" in second_upload
    if second_upload["extraction"].get("reason") == "already_extracted":
        assert "summary" in second_upload["extraction"]
        assert second_upload["extraction"]["summary"]["text_chars"] >= 0

    # Stream ingest (SSE) by path
    r = client.post("/api/ingest/path/stream", json={"path": str(source)})
    assert r.status_code == 200
    stream_text = r.text
    assert "event: stage" in stream_text
    assert "event: payload" in stream_text
    assert '"ok": true' in stream_text

    # Stream ingest (SSE) by upload
    stream_files = {"file": ("upload-stream.txt", b"stream hello", "text/plain")}
    r = client.post("/api/ingest/upload/stream", files=stream_files)
    assert r.status_code == 200
    stream_text = r.text
    assert "event: stage" in stream_text
    assert "event: payload" in stream_text
    assert '"ok": true' in stream_text

    # Resource listing
    r = client.get("/api/resources?limit=10")
    assert r.status_code == 200
    assert r.json()["count"] >= 1
    resource_payload = r.json()["resources"][0]
    resource_id = resource_payload["id"]

    r = client.get(f"/api/resources/{resource_id}")
    assert r.status_code == 200
    assert r.json()["resource"]["id"] == resource_id

    r = client.post(
        f"/api/resources/{resource_id}/title",
        json={
            "title": "Example University - Annual Report - 2024/25",
            "title_candidates": [
                "Example University - Annual Report - 2024/25",
                "Example University Annual Report 2024/25",
            ],
        },
    )
    assert r.status_code == 200
    assert r.json()["resource"]["display_title"] == "Example University - Annual Report - 2024/25"

    r = client.get(f"/api/resources/{resource_id}/content")
    assert r.status_code == 200
    assert r.headers.get("content-disposition", "").startswith("inline;")

    r = client.get(f"/api/viewer/document?resource_id={resource_id}")
    assert r.status_code == 200
    viewer_payload = r.json()
    assert viewer_payload["ok"] is True
    assert viewer_payload["resource"]["id"] == resource_id
    assert "content" in viewer_payload
    assert "metadata" in viewer_payload
    assert "extraction" in viewer_payload
    assert viewer_payload["metadata"]["display_title"] == "Example University - Annual Report - 2024/25"

    # Import now auto-attempts extraction+vector indexing; confirm vector status exists.
    r = client.get(f"/api/vector/status?resource_id={resource_id}")
    assert r.status_code == 200
    assert "runs" in r.json()

    # Extract run on first resource
    r = client.post("/api/extract/run", json={"resource_id": resource_id})
    assert r.status_code == 200

    r = client.get(f"/api/vector/status?resource_id={resource_id}")
    assert r.status_code == 200

    r = client.get(f"/api/extract/text?resource_id={resource_id}")
    assert r.status_code == 200
    assert r.json()["document_text"] is not None

    r = client.get(f"/api/extract/segments?resource_id={resource_id}&limit=50")
    assert r.status_code == 200
    assert r.json()["count"] >= 1

    r = client.get(f"/api/extract/annotations?resource_id={resource_id}&limit=50")
    assert r.status_code == 200
    assert r.json()["count"] >= 1

    r = client.get(f"/api/extract/dump?resource_id={resource_id}")
    assert r.status_code == 200
    assert r.json()["dump"]["document_text"] is not None

    # Duplicate re-import should include persisted extraction summary (not empty stats).
    r = client.post("/api/ingest/path", json={"path": str(source)})
    assert r.status_code == 200
    reingest = r.json()
    assert "extraction" in reingest
    assert reingest["extraction"]["status"] in {"extracted", "failed", "skipped"}
    if reingest["extraction"].get("reason") == "already_extracted":
        summary = reingest["extraction"].get("summary", {})
        assert summary.get("text_chars", 0) >= 0
        assert summary.get("tables_found", 0) >= 0

    # Claims import/list
    claims = tmp_path / "claims.json"
    claims.write_text(
        '{"claims":[{"claim_type":"narrative","narrative_text":"Liquidity improved."}]}',
        encoding="utf-8",
    )
    r = client.post(
        "/api/claims/import",
        json={"file_path": str(claims), "fmt": "json", "claim_set": "web-smoke"},
    )
    assert r.status_code == 200

    r = client.get("/api/claims?claim_set=web-smoke")
    assert r.status_code == 200
    assert r.json()["count"] == 1

    # CEAPF proposition path
    r = client.post(
        "/api/ceapf/proposition",
        json={"proposition": {"subject": "org:X", "predicate": "ceapf:asserts", "object": "p"}},
    )
    assert r.status_code == 200

    # Doctor
    r = client.get("/api/doctor")
    assert r.status_code == 200
    payload = r.json()
    assert "db_runtime" in payload
    assert payload["db_runtime"]["journal_mode"] == "wal"

    r = client.get("/api/dashboard/summary")
    assert r.status_code == 200
    payload = r.json()
    assert payload["ok"] is True
    assert "counts" in payload
    assert "vector" in payload
    assert payload["health"] is None

    r = client.get("/api/dashboard/summary?include_doctor=true")
    assert r.status_code == 200
    payload = r.json()
    assert payload["ok"] is True
    assert payload["health"]["db_runtime"]["journal_mode"] == "wal"

    # Database explorer APIs
    r = client.get("/api/db/tables")
    assert r.status_code == 200
    db_payload = r.json()
    assert db_payload["count"] >= 1
    first_table = db_payload["tables"][0]["name"]

    r = client.get("/api/db/tables/")
    assert r.status_code == 200

    r = client.get(f"/api/db/table?name={first_table}&limit=10&offset=0")
    assert r.status_code == 200
    assert r.json()["table"] == first_table

    r = client.get(f"/api/db/table/?name={first_table}&limit=10&offset=0")
    assert r.status_code == 200

    r = client.get(
        f"/api/db/table/locate?name=resources&column=id&value={resource_id}&page_size=50"
    )
    assert r.status_code == 200
    locate = r.json()
    assert locate["ok"] is True
    assert locate["found"] is True
    assert locate["offset"] >= 0
    assert locate["page_size"] == 50

    # Mass import endpoint (controlled fixture)
    fin_root = tmp_path / "institution"
    fin_root.mkdir(parents=True, exist_ok=True)
    (fin_root / "annual_report_2024.pdf").write_text("fake", encoding="utf-8")
    r = client.post(
        "/api/import/mass",
        json={"root": str(fin_root), "max_files": 5, "skip_extraction": True},
    )
    assert r.status_code == 200
    assert r.json()["stats"]["processed"] >= 1

    # Legacy pipeline alias remains available.
    r = client.post(
        "/api/pipeline/financial-pass",
        json={"root": str(fin_root), "max_files": 5, "skip_extraction": True},
    )
    assert r.status_code == 200
    assert r.json()["stats"]["processed"] >= 0
    assert r.json()["stats"]["already_processed"] >= 1


def test_viewer_document_repairs_legacy_docling_table_payload(tmp_path: Path) -> None:
    project_root = tmp_path / "proj"
    project_root.mkdir(parents=True, exist_ok=True)
    paths = AppPaths(
        project_root=project_root,
        stemma_dir=project_root / ".stemma",
        db_path=project_root / ".stemma" / "stemma.db",
        archive_dir=project_root / ".stemma" / "archive",
        vector_dir=project_root / ".stemma" / "vector",
        qdrant_dir=project_root / ".stemma" / "vector" / "qdrant",
    )
    client = TestClient(create_app(paths))

    source = tmp_path / "report.md"
    source.write_text("| Metric | Value |\n|---|---:|\n| Cash | 5631 |\n", encoding="utf-8")
    ingest_resp = client.post("/api/ingest/path", json={"path": str(source)})
    assert ingest_resp.status_code == 200
    resource_id = ingest_resp.json()["resource"]["id"]

    with get_connection(paths.db_path) as conn:
        table_row = conn.execute(
            "SELECT id FROM extracted_tables WHERE resource_id = ? ORDER BY created_at ASC LIMIT 1",
            (resource_id,),
        ).fetchone()
        assert table_row is not None
        legacy_table_cells = (
            "[TableCell(bbox=BoundingBox(l=10.0, t=20.0, r=90.0, b=30.0, coord_origin=<CoordOrigin.TOPLEFT: "
            "'TOPLEFT'>), row_span=1, col_span=2, start_row_offset_idx=0, end_row_offset_idx=1, "
            "start_col_offset_idx=0, end_col_offset_idx=2, text='Metric', column_header=True, row_header=False, "
            "row_section=False, fillable=False), "
            "TableCell(bbox=BoundingBox(l=10.0, t=31.0, r=50.0, b=40.0, coord_origin=<CoordOrigin.TOPLEFT: "
            "'TOPLEFT'>), row_span=1, col_span=1, start_row_offset_idx=1, end_row_offset_idx=2, "
            "start_col_offset_idx=0, end_col_offset_idx=1, text='Cash', column_header=False, row_header=True, "
            "row_section=False, fillable=False), "
            "TableCell(bbox=BoundingBox(l=51.0, t=31.0, r=90.0, b=40.0, coord_origin=<CoordOrigin.TOPLEFT: "
            "'TOPLEFT'>), row_span=1, col_span=1, start_row_offset_idx=1, end_row_offset_idx=2, "
            "start_col_offset_idx=1, end_col_offset_idx=2, text='5631', column_header=False, row_header=False, "
            "row_section=False, fillable=False)]"
        )
        conn.execute(
            """
            UPDATE extracted_tables
            SET
              row_headers_json = ?,
              col_headers_json = ?,
              cells_json = ?,
              bbox_json = ?
            WHERE id = ?
            """,
            (
                json.dumps(["num_rows", "num_cols"]),
                json.dumps(["table_cells", legacy_table_cells]),
                json.dumps(
                    [
                        {"row_index": 0, "col_index": 0, "value": "num_rows"},
                        {"row_index": 0, "col_index": 1, "value": "2"},
                        {"row_index": 1, "col_index": 0, "value": "num_cols"},
                        {"row_index": 1, "col_index": 1, "value": "2"},
                    ]
                ),
                json.dumps({"x0": 10.0, "y0": 960.0, "x1": 90.0, "y1": 970.0}),
                str(table_row["id"]),
            ),
        )
        conn.commit()

    viewer_resp = client.get(f"/api/viewer/document?resource_id={resource_id}")
    assert viewer_resp.status_code == 200
    payload = viewer_resp.json()
    assert payload["ok"] is True
    assert len(payload["tables"]) >= 1
    table = payload["tables"][0]
    assert table["col_headers"][0] == "Metric"
    assert any(str(cell.get("value")) == "5631" for cell in table["cells"])
    assert isinstance(table["bbox"], dict)
    assert table["bbox"]["x0"] == 10.0
    assert table["bbox"]["y0"] == 20.0
    assert table["bbox"]["x1"] == 90.0
    assert table["bbox"]["y1"] == 40.0


def test_viewer_document_infers_segment_page_index_from_overlapping_anchor(tmp_path: Path) -> None:
    project_root = tmp_path / "proj"
    project_root.mkdir(parents=True, exist_ok=True)
    paths = AppPaths(
        project_root=project_root,
        stemma_dir=project_root / ".stemma",
        db_path=project_root / ".stemma" / "stemma.db",
        archive_dir=project_root / ".stemma" / "archive",
        vector_dir=project_root / ".stemma" / "vector",
        qdrant_dir=project_root / ".stemma" / "vector" / "qdrant",
    )
    client = TestClient(create_app(paths))

    source = tmp_path / "segment-page-source.txt"
    source.write_text(
        "Paragraph one. Sentence one.\nParagraph two. Sentence two.\nParagraph three.\n",
        encoding="utf-8",
    )
    ingest_resp = client.post("/api/ingest/path", json={"path": str(source)})
    assert ingest_resp.status_code == 200
    resource_id = ingest_resp.json()["resource"]["id"]

    with get_connection(paths.db_path) as conn:
        rows = conn.execute(
            """
            SELECT id, start_offset, end_offset
            FROM text_segments
            WHERE resource_id = ?
            ORDER BY start_offset ASC, end_offset ASC
            LIMIT 2
            """,
            (resource_id,),
        ).fetchall()
        assert len(rows) >= 2
        anchor_id = str(rows[0]["id"])
        target_id = str(rows[1]["id"])

        anchor_start = int(rows[0]["start_offset"] or 0)
        anchor_end = max(anchor_start + 200, int(rows[1]["end_offset"] or (anchor_start + 200)))
        target_start = anchor_start + 10
        target_end = target_start + 20

        conn.execute(
            """
            UPDATE text_segments
            SET
              segment_type = 'layout:paragraph',
              start_offset = ?,
              end_offset = ?,
              page_index = 4
            WHERE id = ?
            """,
            (anchor_start, anchor_end, anchor_id),
        )
        conn.execute(
            """
            UPDATE text_segments
            SET
              segment_type = 'structure:sentence',
              start_offset = ?,
              end_offset = ?,
              page_index = NULL
            WHERE id = ?
            """,
            (target_start, target_end, target_id),
        )
        conn.commit()

    viewer_resp = client.get(f"/api/viewer/document?resource_id={resource_id}")
    assert viewer_resp.status_code == 200
    payload = viewer_resp.json()
    assert payload["ok"] is True
    inferred = next((segment for segment in payload["segments"] if segment.get("id") == target_id), None)
    assert inferred is not None
    assert inferred["page_index"] == 4
    assert inferred["page_number"] == 5


def test_viewer_document_reads_segment_page_from_attrs_provenance(tmp_path: Path) -> None:
    project_root = tmp_path / "proj"
    project_root.mkdir(parents=True, exist_ok=True)
    paths = AppPaths(
        project_root=project_root,
        stemma_dir=project_root / ".stemma",
        db_path=project_root / ".stemma" / "stemma.db",
        archive_dir=project_root / ".stemma" / "archive",
        vector_dir=project_root / ".stemma" / "vector",
        qdrant_dir=project_root / ".stemma" / "vector" / "qdrant",
    )
    client = TestClient(create_app(paths))

    source = tmp_path / "segment-attrs-page-source.txt"
    source.write_text("Alpha.\nBravo.\nCharlie.\n", encoding="utf-8")
    ingest_resp = client.post("/api/ingest/path", json={"path": str(source)})
    assert ingest_resp.status_code == 200
    resource_id = ingest_resp.json()["resource"]["id"]

    with get_connection(paths.db_path) as conn:
        row = conn.execute(
            """
            SELECT id
            FROM text_segments
            WHERE resource_id = ?
            ORDER BY start_offset ASC, end_offset ASC
            LIMIT 1
            """,
            (resource_id,),
        ).fetchone()
        assert row is not None
        target_id = str(row["id"])
        conn.execute(
            """
            UPDATE text_segments
            SET page_index = NULL, attrs_json = ?
            WHERE id = ?
            """,
            (json.dumps({"prov": [{"page_no": 3}]}), target_id),
        )
        conn.commit()

    viewer_resp = client.get(f"/api/viewer/document?resource_id={resource_id}")
    assert viewer_resp.status_code == 200
    payload = viewer_resp.json()
    assert payload["ok"] is True
    inferred = next((segment for segment in payload["segments"] if segment.get("id") == target_id), None)
    assert inferred is not None
    assert inferred["page_index"] == 2
    assert inferred["page_number"] == 3


def test_background_import_queue_persists_across_client_reopen(tmp_path: Path) -> None:
    project_root = tmp_path / "proj"
    project_root.mkdir(parents=True, exist_ok=True)
    paths = AppPaths(
        project_root=project_root,
        stemma_dir=project_root / ".stemma",
        db_path=project_root / ".stemma" / "stemma.db",
        archive_dir=project_root / ".stemma" / "archive",
        vector_dir=project_root / ".stemma" / "vector",
        qdrant_dir=project_root / ".stemma" / "vector" / "qdrant",
    )

    queued_job_id = None
    with TestClient(create_app(paths)) as client:
        files = {"file": ("queued-upload.txt", b"background import queue smoke", "text/plain")}
        enqueue_resp = client.post("/api/import/queue/enqueue-upload", files=files)
        assert enqueue_resp.status_code == 200
        enqueue_payload = enqueue_resp.json()
        assert enqueue_payload.get("ok") is True
        queued_job_id = enqueue_payload["job"]["id"]

        deadline = time.time() + 20.0
        latest_job = None
        while time.time() < deadline:
            status_resp = client.get("/api/import/queue/status?limit=50")
            assert status_resp.status_code == 200
            payload = status_resp.json()
            jobs = payload.get("jobs", [])
            latest_job = next((job for job in jobs if job.get("id") == queued_job_id), None)
            if latest_job and latest_job.get("status") in {"done", "failed"}:
                break
            time.sleep(0.15)

        assert latest_job is not None
        assert latest_job.get("status") == "done"

    assert queued_job_id is not None
    with TestClient(create_app(paths)) as reopened_client:
        status_resp = reopened_client.get("/api/import/queue/status?limit=50")
        assert status_resp.status_code == 200
        payload = status_resp.json()
        jobs = payload.get("jobs", [])
        reopened_job = next((job for job in jobs if job.get("id") == queued_job_id), None)
        assert reopened_job is not None
        assert reopened_job.get("status") == "done"


def test_background_import_queue_status_supports_large_limit(tmp_path: Path) -> None:
    project_root = tmp_path / "proj"
    project_root.mkdir(parents=True, exist_ok=True)
    paths = AppPaths(
        project_root=project_root,
        stemma_dir=project_root / ".stemma",
        db_path=project_root / ".stemma" / "stemma.db",
        archive_dir=project_root / ".stemma" / "archive",
        vector_dir=project_root / ".stemma" / "vector",
        qdrant_dir=project_root / ".stemma" / "vector" / "qdrant",
    )

    with TestClient(create_app(paths)) as client:
        files = {"file": ("queued-upload.txt", b"background import queue smoke", "text/plain")}
        enqueue_resp = client.post("/api/import/queue/enqueue-upload", files=files)
        assert enqueue_resp.status_code == 200
        enqueue_payload = enqueue_resp.json()
        assert enqueue_payload.get("ok") is True
        queued_job_id = enqueue_payload["job"]["id"]

        status_resp = client.get("/api/import/queue/status?limit=10000")
        assert status_resp.status_code == 200
        payload = status_resp.json()
        assert payload.get("ok") is True
        jobs = payload.get("jobs", [])
        assert isinstance(jobs, list)
        assert any(str(job.get("id")) == queued_job_id for job in jobs)


def test_sources_panel_api_recovery_and_primary_update(tmp_path: Path) -> None:
    project_root = tmp_path / "proj"
    project_root.mkdir(parents=True, exist_ok=True)
    paths = AppPaths(
        project_root=project_root,
        stemma_dir=project_root / ".stemma",
        db_path=project_root / ".stemma" / "stemma.db",
        archive_dir=project_root / ".stemma" / "archive",
        vector_dir=project_root / ".stemma" / "vector",
        qdrant_dir=project_root / ".stemma" / "vector" / "qdrant",
    )
    client = TestClient(create_app(paths))

    source = tmp_path / "origin-check.txt"
    source.write_text("Source recovery endpoint test.\n", encoding="utf-8")
    ingest_resp = client.post(
        "/api/ingest/path",
        json={"path": str(source), "source_uri": "https://example.org/documents/origin-check"},
    )
    assert ingest_resp.status_code == 200
    resource_id = ingest_resp.json()["resource"]["id"]

    with get_connection(paths.db_path) as conn:
        conn.execute(
            """
            UPDATE resources
            SET download_url = NULL, download_urls_json = NULL
            WHERE id = ?
            """,
            (resource_id,),
        )
        conn.commit()

    list_resp = client.get("/api/sources/resources?limit=100&missing_only=true")
    assert list_resp.status_code == 200
    list_payload = list_resp.json()
    assert list_payload["ok"] is True
    assert any(str(item.get("id")) == resource_id for item in list_payload.get("resources", []))

    detail_resp = client.get(f"/api/sources/resources/{resource_id}")
    assert detail_resp.status_code == 200
    detail_payload = detail_resp.json()
    assert detail_payload["ok"] is True
    assert detail_payload["sources"]["has_external_source"] is False

    recover_resp = client.post(
        f"/api/sources/resources/{resource_id}/recover",
        json={
            "enable_web_search": False,
            "enable_wayback_lookup": False,
            "use_manifest_scan": False,
        },
    )
    assert recover_resp.status_code == 200
    recover_payload = recover_resp.json()
    assert recover_payload["ok"] is True
    assert recover_payload["result"]["resolved_url"] == "https://example.org/documents/origin-check"

    primary_resp = client.post(
        f"/api/sources/resources/{resource_id}/primary",
        json={"url": "10.1234/example.doi.record"},
    )
    assert primary_resp.status_code == 200
    primary_payload = primary_resp.json()
    assert primary_payload["ok"] is True
    assert primary_payload["resource"]["download_url"] == "https://doi.org/10.1234/example.doi.record"
