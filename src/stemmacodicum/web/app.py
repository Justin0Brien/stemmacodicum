from __future__ import annotations

import json
import queue
import tempfile
import threading
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

from fastapi import Body, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from stemmacodicum.application.services.ceapf_service import CEAPFService
from stemmacodicum.application.services.claim_service import ClaimService
from stemmacodicum.application.services.evidence_binding_service import EvidenceBindingService
from stemmacodicum.application.services.extraction_service import ExtractionService
from stemmacodicum.application.services.health_service import HealthService
from stemmacodicum.application.services.ingestion_service import IngestionService
from stemmacodicum.application.services.pipeline_service import FinancialPipelineService
from stemmacodicum.application.services.project_service import ProjectService
from stemmacodicum.application.services.reference_service import ReferenceService
from stemmacodicum.application.services.reporting_service import ReportingService
from stemmacodicum.application.services.trace_service import TraceService
from stemmacodicum.application.services.vector_service import VectorIndexingService
from stemmacodicum.application.services.verification_service import VerificationService
from stemmacodicum.core.config import AppPaths
from stemmacodicum.core.time import now_utc_iso
from stemmacodicum.infrastructure.archive.store import ArchiveStore
from stemmacodicum.infrastructure.db.repos.citation_repo import CitationRepo
from stemmacodicum.infrastructure.db.repos.claim_repo import ClaimRepo
from stemmacodicum.infrastructure.db.repos.evidence_repo import EvidenceRepo
from stemmacodicum.infrastructure.db.repos.extraction_repo import ExtractionRepo
from stemmacodicum.infrastructure.db.repos.reference_repo import ReferenceRepo
from stemmacodicum.infrastructure.db.repos.resource_repo import ResourceRepo
from stemmacodicum.infrastructure.db.repos.vector_repo import VectorRepo
from stemmacodicum.infrastructure.db.repos.verification_repo import VerificationRepo
from stemmacodicum.infrastructure.parsers.docling_adapter import DoclingAdapter
from stemmacodicum.infrastructure.db.sqlite import get_connection
from stemmacodicum.infrastructure.vector.chunking import VectorChunker
from stemmacodicum.infrastructure.vector.embeddings import EmbeddingConfig, SentenceTransformerEmbedder
from stemmacodicum.infrastructure.vector.qdrant_store import QdrantLocalStore


class IngestPathRequest(BaseModel):
    path: str
    source_uri: str | None = None


class LinkReferenceRequest(BaseModel):
    cite_id: str
    resource_digest: str


class ExtractRunRequest(BaseModel):
    resource_id: str | None = None
    resource_digest: str | None = None
    profile: str = "default"


class ClaimsImportRequest(BaseModel):
    file_path: str
    fmt: str
    claim_set: str
    description: str | None = None


class BindAddRequest(BaseModel):
    claim_id: str
    resource_id: str | None = None
    resource_digest: str | None = None
    role: str
    selectors: list[dict[str, Any]]
    page_index: int | None = None
    note: str | None = None


class VerifyClaimRequest(BaseModel):
    claim_id: str
    policy: str = "strict"


class VerifySetRequest(BaseModel):
    claim_set: str
    policy: str = "strict"


class CEAPFPropositionRequest(BaseModel):
    proposition: dict[str, Any]


class CEAPFAssertionRequest(BaseModel):
    proposition_id: str
    asserting_agent: str
    modality: str
    evidence_id: str | None = None


class CEAPFRelationRequest(BaseModel):
    relation_type: str
    from_node_type: str
    from_node_id: str
    to_node_type: str
    to_node_id: str


class PipelineRequest(BaseModel):
    root: str
    max_files: int | None = None
    skip_extraction: bool = False
    extract_timeout_seconds: int | None = 300


class VectorSearchRequest(BaseModel):
    query: str
    limit: int = 10
    resource_id: str | None = None
    resource_digest: str | None = None
    extraction_run_id: str | None = None


class VectorIndexRequest(BaseModel):
    resource_id: str | None = None
    resource_digest: str | None = None
    extraction_run_id: str | None = None
    force: bool = False


class VectorBackfillRequest(BaseModel):
    limit_resources: int = 100000
    max_process: int | None = None


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value


def _quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def create_app(paths: AppPaths) -> FastAPI:
    app = FastAPI(title="Stemma Codicum", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    project_service = ProjectService(paths)
    project_service.init_project()
    vector_service_cache: VectorIndexingService | None = None

    def get_resource_repo() -> ResourceRepo:
        return ResourceRepo(paths.db_path)

    def get_extraction_repo() -> ExtractionRepo:
        return ExtractionRepo(paths.db_path)

    def get_claim_repo() -> ClaimRepo:
        return ClaimRepo(paths.db_path)

    def get_evidence_repo() -> EvidenceRepo:
        return EvidenceRepo(paths.db_path)

    def get_verification_repo() -> VerificationRepo:
        return VerificationRepo(paths.db_path)

    def get_ingestion_service() -> IngestionService:
        return IngestionService(resource_repo=get_resource_repo(), archive_store=ArchiveStore(paths.archive_dir))

    def get_extraction_service() -> ExtractionService:
        return ExtractionService(
            resource_repo=get_resource_repo(),
            extraction_repo=get_extraction_repo(),
            archive_dir=paths.archive_dir,
            vector_indexing_service=get_vector_service(),
        )

    def get_vector_service() -> VectorIndexingService:
        nonlocal vector_service_cache
        if vector_service_cache is None:
            vector_service_cache = VectorIndexingService(
                resource_repo=get_resource_repo(),
                extraction_repo=get_extraction_repo(),
                vector_repo=VectorRepo(paths.db_path),
                vector_store=QdrantLocalStore(storage_path=paths.qdrant_dir),
                embedder=SentenceTransformerEmbedder(config=EmbeddingConfig()),
                chunker=VectorChunker(),
            )
        return vector_service_cache

    def get_reference_service() -> ReferenceService:
        return ReferenceService(
            citation_repo=CitationRepo(paths.db_path),
            reference_repo=ReferenceRepo(paths.db_path),
            resource_repo=get_resource_repo(),
        )

    def get_claim_service() -> ClaimService:
        return ClaimService(get_claim_repo())

    def get_binding_service() -> EvidenceBindingService:
        return EvidenceBindingService(
            claim_repo=get_claim_repo(),
            resource_repo=get_resource_repo(),
            evidence_repo=get_evidence_repo(),
        )

    def get_verify_service() -> VerificationService:
        return VerificationService(
            claim_repo=get_claim_repo(),
            evidence_repo=get_evidence_repo(),
            extraction_repo=get_extraction_repo(),
            verification_repo=get_verification_repo(),
            binding_service=get_binding_service(),
        )

    def get_reporting_service() -> ReportingService:
        return ReportingService(get_verification_repo())

    def emit_stage(
        progress_callback: Callable[[dict[str, object]], None] | None,
        *,
        stage: str,
        state: str,
        progress: int,
        detail: str,
        stats: str | None = None,
    ) -> None:
        if progress_callback is None:
            return
        payload: dict[str, object] = {
            "stage": stage,
            "state": state,
            "progress": max(0, min(100, int(progress))),
            "detail": detail,
        }
        if stats:
            payload["stats"] = stats
        progress_callback(payload)

    def maybe_extract_after_import(
        resource_id: str,
        media_type: str,
        original_filename: str | None = None,
        progress_callback: Callable[[dict[str, object]], None] | None = None,
    ) -> dict[str, Any]:
        if not DoclingAdapter.supports(media_type, original_filename):
            emit_stage(
                progress_callback,
                stage="extract",
                state="skipped",
                progress=100,
                detail="Media type is not extractable.",
            )
            emit_stage(
                progress_callback,
                stage="tables",
                state="skipped",
                progress=100,
                detail="No extraction means no table parsing.",
            )
            emit_stage(
                progress_callback,
                stage="vector",
                state="skipped",
                progress=100,
                detail="No extraction means no vector indexing.",
            )
            return {"attempted": False, "status": "skipped", "reason": "media_type_not_extractable"}

        existing = get_extraction_repo().list_recent_runs(resource_id, limit=1)
        if existing:
            emit_stage(
                progress_callback,
                stage="extract",
                state="done",
                progress=100,
                detail="Existing extraction reused.",
                stats=f"run {existing[0].id}",
            )
            emit_stage(
                progress_callback,
                stage="tables",
                state="done",
                progress=100,
                detail="Structured extraction data already available.",
            )
            vector_summary: dict[str, Any] | None = None
            try:
                vector_summary = _jsonable(
                    get_vector_service().index_extraction(
                        resource_id=resource_id,
                        extraction_run_id=existing[0].id,
                        force=False,
                        progress_callback=progress_callback,
                    )
                )
            except Exception as exc:
                vector_summary = {
                    "status": "failed",
                    "error": str(exc),
                    "chunks_total": 0,
                    "chunks_indexed": 0,
                }
            return {
                "attempted": False,
                "status": "skipped",
                "reason": "already_extracted",
                "extraction_run_id": existing[0].id,
                "vector": vector_summary,
            }

        try:
            summary = get_extraction_service().extract_resource(
                resource_id,
                progress_callback=progress_callback,
            )
        except Exception as exc:
            emit_stage(
                progress_callback,
                stage="extract",
                state="error",
                progress=100,
                detail=f"Extraction failed: {exc}",
            )
            emit_stage(
                progress_callback,
                stage="tables",
                state="skipped",
                progress=100,
                detail="Skipped due to extraction failure.",
            )
            emit_stage(
                progress_callback,
                stage="vector",
                state="skipped",
                progress=100,
                detail="Skipped due to extraction failure.",
            )
            return {"attempted": True, "status": "failed", "error": str(exc)}
        return {
            "attempted": True,
            "status": "extracted",
            "summary": _jsonable(summary),
        }

    def _sse_event(event: str, payload: dict[str, Any]) -> str:
        return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=True)}\n\n"

    def _stream_import_job(
        runner: Callable[[Callable[[dict[str, object]], None]], dict[str, Any]]
    ) -> StreamingResponse:
        event_queue: queue.Queue[tuple[str, dict[str, Any]] | None] = queue.Queue()

        def emit(event: str, payload: dict[str, Any]) -> None:
            event_queue.put((event, payload))

        def worker() -> None:
            try:
                emit(
                    "stage",
                    {
                        "stage": "archive",
                        "state": "active",
                        "progress": 10,
                        "detail": "Hashing file and running dedupe checks.",
                    },
                )
                payload = runner(lambda update: emit("stage", dict(update)))
                emit("payload", payload)
                emit("done", {"ok": True})
            except Exception as exc:
                emit("error", {"error": str(exc)})
            finally:
                event_queue.put(None)

        def iterator() -> Iterator[str]:
            while True:
                item = event_queue.get()
                if item is None:
                    break
                event, payload = item
                yield _sse_event(event, payload)

        threading.Thread(target=worker, daemon=True).start()
        return StreamingResponse(
            iterator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    def _run_import_file(
        *,
        file_path: Path,
        source_uri: str | None,
        uploaded_filename: str | None,
        progress_callback: Callable[[dict[str, object]], None] | None = None,
    ) -> dict[str, Any]:
        result = get_ingestion_service().ingest_file(file_path, source_uri=source_uri)
        archive_detail = (
            "Duplicate detected; existing archive reused."
            if result.status == "duplicate"
            else "Document archived and resource record created."
        )
        archive_stats = (
            f"digest {str(result.resource.digest_sha256)[:10]}..."
            if result.resource.digest_sha256
            else None
        )
        emit_stage(
            progress_callback,
            stage="archive",
            state="done",
            progress=100,
            detail=archive_detail,
            stats=archive_stats,
        )
        extraction = maybe_extract_after_import(
            resource_id=result.resource.id,
            media_type=result.resource.media_type,
            original_filename=result.resource.original_filename,
            progress_callback=progress_callback,
        )
        payload: dict[str, Any] = {
            "ok": True,
            "status": result.status,
            "resource": _jsonable(result.resource),
            "extraction": extraction,
        }
        if uploaded_filename:
            payload["uploaded_filename"] = uploaded_filename
        return payload

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        page = Path(__file__).resolve().parent / "static" / "index.html"
        return HTMLResponse(page.read_text(encoding="utf-8"))

    @app.post("/api/init")
    def api_init() -> dict[str, Any]:
        result = project_service.init_project()
        return {
            "ok": True,
            "db_path": str(result.db_path),
            "paths_created": [str(p) for p in result.paths_created],
        }

    @app.post("/api/ingest/path")
    def api_ingest_path(req: IngestPathRequest) -> dict[str, Any]:
        try:
            result = get_ingestion_service().ingest_file(Path(req.path), source_uri=req.source_uri)
            extraction = maybe_extract_after_import(
                resource_id=result.resource.id,
                media_type=result.resource.media_type,
                original_filename=result.resource.original_filename,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "ok": True,
            "status": result.status,
            "resource": _jsonable(result.resource),
            "extraction": extraction,
        }

    @app.post("/api/ingest/path/stream")
    def api_ingest_path_stream(req: IngestPathRequest) -> StreamingResponse:
        source_path = Path(req.path).expanduser().resolve()
        return _stream_import_job(
            lambda progress_callback: _run_import_file(
                file_path=source_path,
                source_uri=req.source_uri,
                uploaded_filename=None,
                progress_callback=progress_callback,
            )
        )

    @app.post("/api/ingest/upload")
    async def api_ingest_upload(file: UploadFile = File(...)) -> dict[str, Any]:
        suffix = Path(file.filename or "upload.bin").suffix
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                temp_path = Path(tmp.name)
                content = await file.read()
                tmp.write(content)

            result = get_ingestion_service().ingest_file(
                temp_path,
                source_uri=f"upload:{file.filename or temp_path.name}",
            )
            extraction = maybe_extract_after_import(
                resource_id=result.resource.id,
                media_type=result.resource.media_type,
                original_filename=result.resource.original_filename,
            )
            return {
                "ok": True,
                "status": result.status,
                "resource": _jsonable(result.resource),
                "uploaded_filename": file.filename,
                "extraction": extraction,
            }
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            if temp_path and temp_path.exists():
                temp_path.unlink(missing_ok=True)

    @app.post("/api/ingest/upload/stream")
    async def api_ingest_upload_stream(file: UploadFile = File(...)) -> StreamingResponse:
        suffix = Path(file.filename or "upload.bin").suffix
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                temp_path = Path(tmp.name)
                content = await file.read()
                tmp.write(content)
        except Exception as exc:
            if temp_path and temp_path.exists():
                temp_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        def runner(progress_callback: Callable[[dict[str, object]], None]) -> dict[str, Any]:
            assert temp_path is not None
            try:
                return _run_import_file(
                    file_path=temp_path,
                    source_uri=f"upload:{file.filename or temp_path.name}",
                    uploaded_filename=file.filename,
                    progress_callback=progress_callback,
                )
            finally:
                if temp_path.exists():
                    temp_path.unlink(missing_ok=True)

        return _stream_import_job(runner)

    @app.get("/api/resources")
    def api_resources(limit: int = Query(default=100, ge=1, le=100000)) -> dict[str, Any]:
        resources = [_jsonable(r) for r in get_resource_repo().list(limit=limit)]
        return {"ok": True, "count": len(resources), "resources": resources}

    def _db_tables_payload(limit: int) -> dict[str, Any]:
        with get_connection(paths.db_path) as conn:
            rows = conn.execute(
                """
                SELECT name, sql
                FROM sqlite_master
                WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
                ORDER BY name ASC
                """
            ).fetchall()

            out: list[dict[str, Any]] = []
            for row in rows[:limit]:
                table_name = str(row["name"])
                quoted = _quote_identifier(table_name)
                row_count = conn.execute(f"SELECT COUNT(*) AS c FROM {quoted}").fetchone()["c"]
                col_rows = conn.execute(f"PRAGMA table_info({quoted})").fetchall()
                columns = [
                    {
                        "name": c["name"],
                        "type": c["type"],
                        "notnull": bool(c["notnull"]),
                        "pk": bool(c["pk"]),
                    }
                    for c in col_rows
                ]
                out.append(
                    {
                        "name": table_name,
                        "row_count": int(row_count),
                        "columns": columns,
                        "create_sql": row["sql"],
                    }
                )

        return {"ok": True, "count": len(out), "tables": out}

    def _db_table_payload(name: str, limit: int, offset: int) -> dict[str, Any]:
        with get_connection(paths.db_path) as conn:
            table_rows = conn.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
                ORDER BY name ASC
                """
            ).fetchall()
            allowed = {str(r["name"]) for r in table_rows}

            if name not in allowed:
                raise HTTPException(status_code=404, detail=f"Table not found: {name}")

            quoted = _quote_identifier(name)
            total_rows = conn.execute(f"SELECT COUNT(*) AS c FROM {quoted}").fetchone()["c"]
            col_rows = conn.execute(f"PRAGMA table_info({quoted})").fetchall()
            columns = [c["name"] for c in col_rows]

            data_rows = conn.execute(
                f"SELECT * FROM {quoted} LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
            rows = [{k: row[k] for k in row.keys()} for row in data_rows]

            schema = conn.execute(
                """
                SELECT sql
                FROM sqlite_master
                WHERE type = 'table' AND name = ?
                """,
                (name,),
            ).fetchone()

        return {
            "ok": True,
            "table": name,
            "limit": limit,
            "offset": offset,
            "total_rows": int(total_rows),
            "columns": columns,
            "rows": rows,
            "create_sql": schema["sql"] if schema else None,
        }

    @app.get("/api/db/tables")
    @app.get("/api/db/tables/")
    def api_db_tables(limit: int = Query(default=500, ge=1, le=2000)) -> dict[str, Any]:
        return _db_tables_payload(limit=limit)

    @app.get("/api/db/table")
    @app.get("/api/db/table/")
    def api_db_table(
        name: str = Query(..., min_length=1),
        limit: int = Query(default=50, ge=1, le=1000),
        offset: int = Query(default=0, ge=0),
    ) -> dict[str, Any]:
        return _db_table_payload(name=name, limit=limit, offset=offset)

    @app.get("/api/dashboard/summary")
    def api_dashboard_summary(include_doctor: bool = Query(default=False)) -> dict[str, Any]:
        with get_connection(paths.db_path) as conn:
            counts = {
                "resources": int(conn.execute("SELECT COUNT(*) AS c FROM resources").fetchone()["c"]),
                "citations": int(conn.execute("SELECT COUNT(*) AS c FROM citations").fetchone()["c"]),
                "references": int(conn.execute("SELECT COUNT(*) AS c FROM reference_entries").fetchone()["c"]),
                "claims": int(conn.execute("SELECT COUNT(*) AS c FROM claims").fetchone()["c"]),
                "evidence_items": int(conn.execute("SELECT COUNT(*) AS c FROM evidence_items").fetchone()["c"]),
                "claim_evidence_bindings": int(
                    conn.execute("SELECT COUNT(*) AS c FROM claim_evidence_bindings").fetchone()["c"]
                ),
                "extraction_runs": int(conn.execute("SELECT COUNT(*) AS c FROM extraction_runs").fetchone()["c"]),
                "verification_runs": int(conn.execute("SELECT COUNT(*) AS c FROM verification_runs").fetchone()["c"]),
                "vector_index_runs": int(conn.execute("SELECT COUNT(*) AS c FROM vector_index_runs").fetchone()["c"]),
                "vector_chunks": int(conn.execute("SELECT COUNT(*) AS c FROM vector_chunks").fetchone()["c"]),
            }

            latest_extraction_row = conn.execute(
                """
                SELECT id, resource_id, parser_name, status, created_at
                FROM extraction_runs
                ORDER BY created_at DESC
                LIMIT 1
                """
            ).fetchone()
            latest_verification_row = conn.execute(
                """
                SELECT id, claim_set_id, policy_profile, created_at
                FROM verification_runs
                ORDER BY created_at DESC
                LIMIT 1
                """
            ).fetchone()

            verification_status_counts: dict[str, int] = {}
            if latest_verification_row is not None:
                rows = conn.execute(
                    """
                    SELECT status, COUNT(*) AS c
                    FROM verification_results
                    WHERE run_id = ?
                    GROUP BY status
                    ORDER BY status ASC
                    """,
                    (str(latest_verification_row["id"]),),
                ).fetchall()
                verification_status_counts = {str(row["status"]): int(row["c"]) for row in rows}

        vector_payload = get_vector_service().status(limit_runs=25)
        run_status_counts: dict[str, int] = {}
        for run in vector_payload.get("runs", []):
            status = str(run.get("status") or "unknown")
            run_status_counts[status] = run_status_counts.get(status, 0) + 1

        health_payload: dict[str, Any] | None = None
        if include_doctor:
            report = HealthService(paths.db_path, paths.archive_dir).run_doctor()
            issues = [_jsonable(i) for i in report.issues]
            health_payload = {
                "ok": report.ok,
                "checks_run": report.checks_run,
                "warning_count": sum(1 for i in report.issues if i.level == "warning"),
                "error_count": sum(1 for i in report.issues if i.level == "error"),
                "db_runtime": _jsonable(report.db_runtime),
                "issues": issues,
            }

        return {
            "ok": True,
            "generated_at": now_utc_iso(),
            "project": {
                "root": str(paths.project_root),
                "db_path": str(paths.db_path),
                "archive_dir": str(paths.archive_dir),
                "qdrant_dir": str(paths.qdrant_dir),
            },
            "counts": counts,
            "latest": {
                "extraction_run": (
                    {
                        "id": latest_extraction_row["id"],
                        "resource_id": latest_extraction_row["resource_id"],
                        "parser_name": latest_extraction_row["parser_name"],
                        "status": latest_extraction_row["status"],
                        "created_at": latest_extraction_row["created_at"],
                    }
                    if latest_extraction_row
                    else None
                ),
                "verification_run": (
                    {
                        "id": latest_verification_row["id"],
                        "claim_set_id": latest_verification_row["claim_set_id"],
                        "policy_profile": latest_verification_row["policy_profile"],
                        "created_at": latest_verification_row["created_at"],
                        "result_status_counts": verification_status_counts,
                    }
                    if latest_verification_row
                    else None
                ),
            },
            "vector": {
                "backend": vector_payload.get("backend"),
                "collection_name": vector_payload.get("collection_name"),
                "embedding_model": vector_payload.get("embedding_model"),
                "vector_chunk_rows": vector_payload.get("vector_chunk_rows"),
                "distinct_chunk_ids": vector_payload.get("distinct_chunk_ids"),
                "qdrant_points": vector_payload.get("qdrant_points"),
                "run_status_counts": run_status_counts,
                "runs": vector_payload.get("runs", []),
            },
            "health": health_payload,
        }

    @app.post("/api/refs/import-bib")
    def api_refs_import_bib(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        path = payload.get("bib_path")
        if not path:
            raise HTTPException(status_code=400, detail="bib_path is required")
        try:
            summary = get_reference_service().import_bibtex(Path(str(path)))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, "summary": _jsonable(summary)}

    @app.get("/api/refs")
    def api_refs(limit: int = Query(default=100, ge=1, le=100000)) -> dict[str, Any]:
        refs = [_jsonable(r) for r in ReferenceRepo(paths.db_path).list(limit=limit)]
        return {"ok": True, "count": len(refs), "references": refs}

    @app.get("/api/citations")
    def api_citations(limit: int = Query(default=100, ge=1, le=100000)) -> dict[str, Any]:
        citations = [_jsonable(c) for c in CitationRepo(paths.db_path).list(limit=limit)]
        return {"ok": True, "count": len(citations), "citations": citations}

    @app.post("/api/refs/link-resource")
    def api_refs_link(req: LinkReferenceRequest) -> dict[str, Any]:
        try:
            get_reference_service().link_reference_to_resource(req.cite_id, req.resource_digest)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True}

    @app.post("/api/extract/run")
    def api_extract_run(req: ExtractRunRequest) -> dict[str, Any]:
        try:
            if req.resource_id:
                resource_id = req.resource_id
            elif req.resource_digest:
                resource = get_resource_repo().get_by_digest(req.resource_digest)
                if resource is None:
                    raise ValueError(f"Resource digest not found: {req.resource_digest}")
                resource_id = resource.id
            else:
                raise ValueError("Provide resource_id or resource_digest")

            summary = get_extraction_service().extract_resource(resource_id, parser_profile=req.profile)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, "summary": _jsonable(summary)}

    @app.post("/api/vector/index")
    def api_vector_index(req: VectorIndexRequest) -> dict[str, Any]:
        try:
            if req.resource_id:
                resource_id = req.resource_id
            elif req.resource_digest:
                resource = get_resource_repo().get_by_digest(req.resource_digest)
                if resource is None:
                    raise ValueError(f"Resource digest not found: {req.resource_digest}")
                resource_id = resource.id
            else:
                raise ValueError("Provide resource_id or resource_digest")

            if req.extraction_run_id:
                summary = get_vector_service().index_extraction(
                    resource_id=resource_id,
                    extraction_run_id=req.extraction_run_id,
                    force=req.force,
                )
            else:
                summary = get_vector_service().index_latest_for_resource(resource_id=resource_id, force=req.force)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, "summary": _jsonable(summary)}

    @app.post("/api/vector/search")
    def api_vector_search(req: VectorSearchRequest) -> dict[str, Any]:
        try:
            resource_id = req.resource_id
            if resource_id is None and req.resource_digest:
                resource = get_resource_repo().get_by_digest(req.resource_digest)
                if resource is None:
                    raise ValueError(f"Resource digest not found: {req.resource_digest}")
                resource_id = resource.id
            hits = get_vector_service().search(
                query=req.query,
                limit=req.limit,
                resource_id=resource_id,
                extraction_run_id=req.extraction_run_id,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, "count": len(hits), "hits": hits}

    @app.get("/api/vector/status")
    def api_vector_status(
        resource_id: str | None = None,
        resource_digest: str | None = None,
        limit_runs: int = Query(default=50, ge=1, le=1000),
    ) -> dict[str, Any]:
        try:
            resolved_resource_id = resource_id
            if resolved_resource_id is None and resource_digest:
                resource = get_resource_repo().get_by_digest(resource_digest)
                if resource is None:
                    raise ValueError(f"Resource digest not found: {resource_digest}")
                resolved_resource_id = resource.id
            payload = get_vector_service().status(resource_id=resolved_resource_id, limit_runs=limit_runs)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, **payload}

    @app.post("/api/vector/backfill")
    def api_vector_backfill(req: VectorBackfillRequest) -> dict[str, Any]:
        try:
            summary = get_vector_service().backfill_latest(
                limit_resources=req.limit_resources,
                max_process=req.max_process,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, "summary": _jsonable(summary)}

    def _resolve_extract_resource_id(resource_id: str | None, resource_digest: str | None) -> str:
        if resource_id:
            return resource_id
        if resource_digest:
            resource = get_resource_repo().get_by_digest(resource_digest)
            if resource is None:
                raise ValueError(f"Resource digest not found: {resource_digest}")
            return resource.id
        raise ValueError("Provide resource_id or resource_digest")

    @app.get("/api/extract/tables")
    def api_extract_tables(
        resource_id: str | None = None,
        resource_digest: str | None = None,
        limit: int = Query(default=100, ge=1, le=100000),
    ) -> dict[str, Any]:
        try:
            target_id = _resolve_extract_resource_id(resource_id, resource_digest)
            tables = [_jsonable(t) for t in get_extraction_service().list_tables(target_id, limit=limit)]
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, "count": len(tables), "tables": tables}

    @app.get("/api/extract/text")
    def api_extract_text(
        resource_id: str | None = None,
        resource_digest: str | None = None,
        extraction_run_id: str | None = None,
    ) -> dict[str, Any]:
        try:
            target_id = _resolve_extract_resource_id(resource_id, resource_digest)
            doc_text = get_extraction_service().get_document_text(
                resource_id=target_id,
                extraction_run_id=extraction_run_id,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, "document_text": _jsonable(doc_text) if doc_text else None}

    @app.get("/api/extract/segments")
    def api_extract_segments(
        resource_id: str | None = None,
        resource_digest: str | None = None,
        extraction_run_id: str | None = None,
        segment_type: str | None = None,
        limit: int = Query(default=1000, ge=1, le=200000),
    ) -> dict[str, Any]:
        try:
            target_id = _resolve_extract_resource_id(resource_id, resource_digest)
            segments = get_extraction_service().list_segments(
                resource_id=target_id,
                extraction_run_id=extraction_run_id,
                segment_type=segment_type,
                limit=limit,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        payload = [_jsonable(s) for s in segments]
        return {"ok": True, "count": len(payload), "segments": payload}

    @app.get("/api/extract/annotations")
    def api_extract_annotations(
        resource_id: str | None = None,
        resource_digest: str | None = None,
        extraction_run_id: str | None = None,
        layer: str | None = None,
        category: str | None = None,
        limit: int = Query(default=1000, ge=1, le=200000),
    ) -> dict[str, Any]:
        try:
            target_id = _resolve_extract_resource_id(resource_id, resource_digest)
            annotations = get_extraction_service().list_annotations(
                resource_id=target_id,
                extraction_run_id=extraction_run_id,
                layer=layer,
                category=category,
                limit=limit,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, "count": len(annotations), "annotations": annotations}

    @app.get("/api/extract/dump")
    def api_extract_dump(
        resource_id: str | None = None,
        resource_digest: str | None = None,
        extraction_run_id: str | None = None,
        segment_limit: int = Query(default=5000, ge=1, le=500000),
        annotation_limit: int = Query(default=5000, ge=1, le=500000),
        table_limit: int = Query(default=1000, ge=1, le=500000),
    ) -> dict[str, Any]:
        try:
            target_id = _resolve_extract_resource_id(resource_id, resource_digest)
            payload = get_extraction_service().build_dump(
                resource_id=target_id,
                extraction_run_id=extraction_run_id,
                segment_limit=segment_limit,
                annotation_limit=annotation_limit,
                table_limit=table_limit,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, "dump": payload}

    @app.post("/api/claims/import")
    def api_claims_import(req: ClaimsImportRequest) -> dict[str, Any]:
        try:
            summary = get_claim_service().import_claims(
                file_path=Path(req.file_path),
                fmt=req.fmt,
                claim_set_name=req.claim_set,
                claim_set_description=req.description,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, "summary": _jsonable(summary)}

    @app.get("/api/claims")
    def api_claims(
        claim_set: str | None = None,
        limit: int = Query(default=100, ge=1, le=100000),
    ) -> dict[str, Any]:
        try:
            claims = [_jsonable(c) for c in get_claim_service().list_claims(claim_set_name=claim_set, limit=limit)]
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, "count": len(claims), "claims": claims}

    @app.get("/api/claim-sets")
    def api_claim_sets(limit: int = Query(default=100, ge=1, le=100000)) -> dict[str, Any]:
        sets = [_jsonable(s) for s in get_claim_service().list_claim_sets(limit=limit)]
        return {"ok": True, "count": len(sets), "claim_sets": sets}

    @app.post("/api/bind/add")
    def api_bind_add(req: BindAddRequest) -> dict[str, Any]:
        try:
            if req.resource_id:
                resource_id = req.resource_id
            elif req.resource_digest:
                resource = get_resource_repo().get_by_digest(req.resource_digest)
                if resource is None:
                    raise ValueError(f"Resource digest not found: {req.resource_digest}")
                resource_id = resource.id
            else:
                raise ValueError("Provide resource_id or resource_digest")

            evidence_id = get_binding_service().bind_evidence(
                claim_id=req.claim_id,
                resource_id=resource_id,
                role=req.role,
                selectors=req.selectors,
                page_index=req.page_index,
                note=req.note,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, "evidence_id": evidence_id}

    @app.post("/api/bind/validate")
    def api_bind_validate(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        claim_id = payload.get("claim_id")
        if not claim_id:
            raise HTTPException(status_code=400, detail="claim_id is required")
        try:
            result = get_binding_service().validate_binding(str(claim_id))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, "result": _jsonable(result)}

    @app.post("/api/verify/claim")
    def api_verify_claim(req: VerifyClaimRequest) -> dict[str, Any]:
        try:
            outcome = get_verify_service().verify_claim(req.claim_id, policy_profile=req.policy)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, "outcome": _jsonable(outcome)}

    @app.post("/api/verify/set")
    def api_verify_set(req: VerifySetRequest) -> dict[str, Any]:
        try:
            outcome = get_verify_service().verify_claim_set(req.claim_set, policy_profile=req.policy)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, "outcome": _jsonable(outcome)}

    @app.get("/api/report/verification")
    def api_report_verification(
        run_id: str,
        json_out: str | None = None,
        md_out: str | None = None,
    ) -> dict[str, Any]:
        service = get_reporting_service()
        try:
            summary = service.build_run_summary(run_id)
            exports: dict[str, str] = {}
            if json_out:
                exports["json_out"] = str(service.export_json_report(run_id, Path(json_out)))
            if md_out:
                exports["md_out"] = str(service.export_markdown_report(run_id, Path(md_out)))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return {"ok": True, "summary": _jsonable(summary), "exports": exports}

    @app.get("/api/trace/claim")
    def api_trace_claim(claim_id: str) -> dict[str, Any]:
        try:
            trace = TraceService(paths.db_path).trace_claim(claim_id)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, "trace": _jsonable(trace)}

    @app.get("/api/trace/resource")
    def api_trace_resource(
        resource_id: str | None = None,
        resource_digest: str | None = None,
    ) -> dict[str, Any]:
        try:
            trace = TraceService(paths.db_path).trace_resource(
                resource_id=resource_id,
                digest_sha256=resource_digest,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, "trace": _jsonable(trace)}

    @app.get("/api/trace/citation")
    def api_trace_citation(cite_id: str) -> dict[str, Any]:
        try:
            trace = TraceService(paths.db_path).trace_citation(cite_id)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, "trace": _jsonable(trace)}

    @app.get("/api/doctor")
    def api_doctor() -> dict[str, Any]:
        report = HealthService(paths.db_path, paths.archive_dir).run_doctor()
        return {
            "ok": report.ok,
            "checks_run": report.checks_run,
            "db_runtime": _jsonable(report.db_runtime),
            "issues": [_jsonable(i) for i in report.issues],
        }

    @app.post("/api/ceapf/proposition")
    def api_ceapf_proposition(req: CEAPFPropositionRequest) -> dict[str, Any]:
        try:
            proposition_id = CEAPFService(paths.db_path).create_proposition(req.proposition)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, "proposition_id": proposition_id}

    @app.post("/api/ceapf/assertion")
    def api_ceapf_assertion(req: CEAPFAssertionRequest) -> dict[str, Any]:
        try:
            assertion_id = CEAPFService(paths.db_path).create_assertion_event(
                proposition_id=req.proposition_id,
                asserting_agent=req.asserting_agent,
                modality=req.modality,
                evidence_id=req.evidence_id,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, "assertion_id": assertion_id}

    @app.post("/api/ceapf/relation")
    def api_ceapf_relation(req: CEAPFRelationRequest) -> dict[str, Any]:
        try:
            relation_id = CEAPFService(paths.db_path).add_argument_relation(
                relation_type=req.relation_type,
                from_node_type=req.from_node_type,
                from_node_id=req.from_node_id,
                to_node_type=req.to_node_type,
                to_node_id=req.to_node_id,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, "relation_id": relation_id}

    @app.get("/api/ceapf/propositions")
    def api_ceapf_list(limit: int = Query(default=100, ge=1, le=100000)) -> dict[str, Any]:
        props = CEAPFService(paths.db_path).list_propositions(limit=limit)
        return {
            "ok": True,
            "count": len(props),
            "propositions": [
                {
                    "id": p.id,
                    "proposition": p.proposition,
                }
                for p in props
            ],
        }

    def run_mass_import(req: PipelineRequest) -> dict[str, Any]:
        pipeline = FinancialPipelineService(
            ingestion_service=get_ingestion_service(),
            extraction_service=get_extraction_service(),
            extraction_repo=get_extraction_repo(),
            state_path=paths.stemma_dir / "mass_import_state.json",
            log_path=paths.stemma_dir / "mass_import.log.jsonl",
        )
        try:
            stats = pipeline.run(
                root=Path(req.root),
                max_files=req.max_files,
                run_extraction=not req.skip_extraction,
                extract_timeout_seconds=req.extract_timeout_seconds,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return {"ok": True, "stats": _jsonable(stats)}

    @app.post("/api/import/mass")
    def api_import_mass(req: PipelineRequest) -> dict[str, Any]:
        return run_mass_import(req)

    @app.post("/api/pipeline/financial-pass")
    def api_pipeline_legacy(req: PipelineRequest) -> dict[str, Any]:
        return run_mass_import(req)

    return app
