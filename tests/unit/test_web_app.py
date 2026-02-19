from pathlib import Path

from fastapi.testclient import TestClient

from stemmacodicum.core.config import AppPaths
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

    # Upload ingest and dedupe check
    files = {"file": ("upload.txt", b"hello world", "text/plain")}
    r1 = client.post("/api/ingest/upload", files=files)
    r2 = client.post("/api/ingest/upload", files=files)
    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r1.json()["status"] in {"ingested", "duplicate"}
    assert r2.json()["status"] in {"ingested", "duplicate"}

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
