from __future__ import annotations

import json
import importlib.util
import mimetypes
import os
import queue
import re
import shutil
import sys
import tempfile
import threading
from dataclasses import asdict, is_dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

from fastapi import Body, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from stemmacodicum.application.services.ceapf_service import CEAPFService
from stemmacodicum.application.services.claim_service import ClaimService
from stemmacodicum.application.services.background_import_queue_service import (
    BackgroundImportQueueService,
)
from stemmacodicum.application.services.evidence_binding_service import EvidenceBindingService
from stemmacodicum.application.services.extraction_service import (
    ExtractionCancelledError,
    ExtractionService,
)
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


class ImportQueueControlRequest(BaseModel):
    action: str


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


class ResourceTitleUpdateRequest(BaseModel):
    title: str | None = None
    title_candidates: list[str] | None = None


class SourceRecoverRequest(BaseModel):
    use_manifest_scan: bool = True
    manifest_root: str = "/Volumes/X10/data"
    manifest_max_files: int = 25000
    enable_wayback_lookup: bool = True
    enable_web_search: bool = True
    wayback_delay: float = 1.0
    shallow_search_dirs: list[str] | None = None
    deep_search_dirs: list[str] | None = None
    persist: bool = True


class SourcePrimaryUrlRequest(BaseModel):
    url: str


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
    static_dir = Path(__file__).resolve().parent / "static"

    def _env_bool(name: str, default: bool) -> bool:
        raw = os.getenv(name)
        if raw is None:
            return default
        value = str(raw).strip().lower()
        if value in {"1", "true", "yes", "on"}:
            return True
        if value in {"0", "false", "no", "off"}:
            return False
        return default

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    project_service = ProjectService(paths)
    project_service.init_project()
    vector_service_cache: VectorIndexingService | None = None
    source_recovery_module_cache: Any | None = None

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

    def get_source_recovery_module() -> Any:
        nonlocal source_recovery_module_cache
        if source_recovery_module_cache is not None:
            return source_recovery_module_cache
        candidate_paths = [
            (paths.project_root / "scripts" / "recover_reference_urls.py").resolve(),
            (Path(__file__).resolve().parents[3] / "scripts" / "recover_reference_urls.py").resolve(),
        ]
        script_path = next((path for path in candidate_paths if path.exists()), None)
        if script_path is None:
            raise RuntimeError(
                "Source recovery script not found in project scripts/ or repository scripts/."
            )
        spec = importlib.util.spec_from_file_location("stemma_recover_reference_urls", str(script_path))
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load source recovery script: {script_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
        source_recovery_module_cache = module
        return module

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
        page_count: int | None = None,
        page_current: int | None = None,
        resource_id: str | None = None,
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
        if isinstance(page_count, int) and page_count > 0:
            payload["page_count"] = page_count
        if isinstance(page_current, int) and page_current > 0:
            payload["page_current"] = page_current
        if resource_id:
            payload["resource_id"] = resource_id
        progress_callback(payload)

    def _safe_json_list_len(value: str | None) -> int:
        if not value:
            return 0
        try:
            parsed = json.loads(value)
        except Exception:
            return 0
        return len(parsed) if isinstance(parsed, list) else 0

    def _compute_text_stats(text: str) -> tuple[int, int, int]:
        word_count = len(re.findall(r"\b\w+\b", text))
        sentence_count = len(
            [m.group(0).strip() for m in re.finditer(r"[^.!?\n]+(?:[.!?]+|$)", text) if m.group(0).strip()]
        )
        paragraph_count = len([p for p in re.split(r"\n\s*\n+", text) if p.strip()])
        return word_count, sentence_count, paragraph_count

    def _is_existing_extraction_incomplete(resource_id: str, extraction_run_id: str) -> tuple[bool, str]:
        repo = get_extraction_repo()
        run = repo.get_run_by_id(extraction_run_id)
        if run is None:
            return True, "run_missing"
        if str(run.status).lower() != "success":
            return True, f"run_status={run.status}"
        doc_text = repo.get_document_text_for_run(extraction_run_id)
        if doc_text is None:
            return True, "document_text_missing"
        if int(doc_text.char_count or 0) > 0:
            segments = repo.list_segments_for_resource(
                resource_id=resource_id,
                extraction_run_id=extraction_run_id,
                limit=1,
            )
            if not segments:
                return True, "segments_missing_for_nonempty_text"
        return False, ""

    def _build_existing_extraction_summary(resource_id: str, extraction_run_id: str) -> dict[str, Any]:
        repo = get_extraction_repo()
        run = repo.get_run_by_id(extraction_run_id)
        if run is None:
            return {}

        doc_text = repo.get_document_text_for_run(extraction_run_id)
        text_content = str(doc_text.text_content) if doc_text and doc_text.text_content else ""
        text_chars = int(doc_text.char_count) if doc_text and doc_text.char_count is not None else len(text_content)
        text_words, text_sentences, text_paragraphs = _compute_text_stats(text_content)

        tables = repo.list_tables_for_run(extraction_run_id=extraction_run_id, limit=100000)
        table_rows_total = sum(_safe_json_list_len(t.row_headers_json) for t in tables)
        table_cols_total = sum(_safe_json_list_len(t.col_headers_json) for t in tables)
        table_cells_total = sum(_safe_json_list_len(t.cells_json) for t in tables)
        tables_found = len(tables)

        segments = repo.list_segments_for_resource(
            resource_id=resource_id,
            extraction_run_id=extraction_run_id,
            limit=200000,
        )
        annotations = repo.list_annotations_for_resource(
            resource_id=resource_id,
            extraction_run_id=extraction_run_id,
            limit=200000,
        )

        page_indexes: set[int] = set()
        for table in tables:
            if table.page_index is not None and int(table.page_index) >= 0:
                page_indexes.add(int(table.page_index))
        for segment in segments:
            if segment.page_index is not None and int(segment.page_index) >= 0:
                page_indexes.add(int(segment.page_index))
        page_count = (max(page_indexes) + 1) if page_indexes else None

        return {
            "run_id": run.id,
            "resource_id": run.resource_id,
            "tables_found": tables_found,
            "parser_name": run.parser_name,
            "parser_version": run.parser_version,
            "elapsed_seconds": None,
            "page_count": page_count,
            "pages_per_second": None,
            "timings": None,
            "text_chars": text_chars,
            "text_words": text_words,
            "text_sentences": text_sentences,
            "text_paragraphs": text_paragraphs,
            "segments_persisted": len(segments),
            "annotations_persisted": len(annotations),
            "table_rows_total": table_rows_total,
            "table_cols_total": table_cols_total,
            "table_cells_total": table_cells_total,
        }

    def maybe_extract_after_import(
        resource_id: str,
        media_type: str,
        original_filename: str | None = None,
        progress_callback: Callable[[dict[str, object]], None] | None = None,
        cancellation_check: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        def ensure_not_cancelled(context: str) -> None:
            if cancellation_check is not None and bool(cancellation_check()):
                raise ExtractionCancelledError(f"{context} cancelled by control request.")

        ensure_not_cancelled("extraction scheduling")
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
            run_id = existing[0].id
            incomplete, reason = _is_existing_extraction_incomplete(resource_id=resource_id, extraction_run_id=run_id)
            if incomplete:
                emit_stage(
                    progress_callback,
                    stage="extract",
                    state="active",
                    progress=30,
                    detail=f"Existing extraction incomplete ({reason}); re-running extraction.",
                )
                try:
                    summary = get_extraction_service().extract_resource(
                        resource_id,
                        progress_callback=progress_callback,
                        cancellation_check=cancellation_check,
                    )
                except ExtractionCancelledError:
                    raise
                except Exception as exc:
                    emit_stage(
                        progress_callback,
                        stage="extract",
                        state="error",
                        progress=100,
                        detail=f"Re-extraction failed: {exc}",
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
                    "reason": "reextracted_incomplete_previous",
                    "summary": _jsonable(summary),
                }

            existing_summary = _build_existing_extraction_summary(resource_id=resource_id, extraction_run_id=run_id)
            emit_stage(
                progress_callback,
                stage="extract",
                state="done",
                progress=100,
                detail="Existing extraction reused.",
                page_count=int(existing_summary.get("page_count") or 0) or None,
                stats=(
                    f"run {run_id} • "
                    f"{existing_summary.get('text_words', 0)} words • "
                    f"{existing_summary.get('text_sentences', 0)} sentences"
                ),
            )
            emit_stage(
                progress_callback,
                stage="tables",
                state="done",
                progress=100,
                detail="Structured extraction data already available.",
                stats=(
                    f"{existing_summary.get('tables_found', 0)} tables • "
                    f"{existing_summary.get('table_rows_total', 0)} rows • "
                    f"{existing_summary.get('table_cols_total', 0)} cols • "
                    f"{existing_summary.get('table_cells_total', 0)} cells"
                ),
            )
            vector_summary: dict[str, Any] | None = None
            try:
                ensure_not_cancelled("vector scheduling")
                vector_summary = _jsonable(
                    get_vector_service().index_extraction(
                        resource_id=resource_id,
                        extraction_run_id=run_id,
                        force=False,
                        progress_callback=progress_callback,
                    )
                )
                ensure_not_cancelled("vector indexing")
            except ExtractionCancelledError:
                raise
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
                "extraction_run_id": run_id,
                "summary": existing_summary,
                "vector": vector_summary,
            }

        try:
            summary = get_extraction_service().extract_resource(
                resource_id,
                progress_callback=progress_callback,
                cancellation_check=cancellation_check,
            )
        except ExtractionCancelledError:
            raise
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
        emit_seq = 0

        def emit(event: str, payload: dict[str, Any]) -> None:
            nonlocal emit_seq
            emit_seq += 1
            safe_payload = dict(payload or {})
            safe_payload.setdefault("emitted_at", now_utc_iso())
            safe_payload.setdefault("event_seq", emit_seq)
            safe_payload.setdefault("event_name", event)
            event_queue.put((event, safe_payload))

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
        resume_resource_id: str | None = None,
        resume_progress: dict[str, Any] | None = None,
        cancellation_check: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        def ensure_not_cancelled(context: str) -> None:
            if cancellation_check is not None and bool(cancellation_check()):
                raise ExtractionCancelledError(f"{context} cancelled by control request.")

        ensure_not_cancelled("import")
        resumed_from_existing_archive = False
        resource = None
        status = "unknown"
        existing_resource = get_resource_repo().get_by_id(resume_resource_id) if resume_resource_id else None
        if existing_resource is not None:
            resource = existing_resource
            status = "duplicate"
            resumed_from_existing_archive = True
        else:
            result = get_ingestion_service().ingest_file(file_path, source_uri=source_uri)
            resource = result.resource
            status = result.status
        archive_detail = (
            "Resuming from existing archived resource."
            if resumed_from_existing_archive
            else (
                "Duplicate detected; existing archive reused."
                if status == "duplicate"
                else "Document archived and resource record created."
            )
        )
        archive_stats = (
            f"digest {str(resource.digest_sha256)[:10]}..."
            if resource and resource.digest_sha256
            else None
        )
        if resumed_from_existing_archive and isinstance(resume_progress, dict):
            prior_status = str(resume_progress.get("status_line") or "").strip()
            if prior_status:
                archive_detail = f"{archive_detail} Last status: {prior_status}"
        emit_stage(
            progress_callback,
            stage="archive",
            state="done",
            progress=100,
            detail=archive_detail,
            stats=archive_stats,
            resource_id=resource.id if resource is not None else None,
        )
        ensure_not_cancelled("extraction start")
        extraction = maybe_extract_after_import(
            resource_id=resource.id,
            media_type=resource.media_type,
            original_filename=resource.original_filename,
            progress_callback=progress_callback,
            cancellation_check=cancellation_check,
        )
        ensure_not_cancelled("import finalization")
        payload: dict[str, Any] = {
            "ok": True,
            "status": status,
            "resource": _jsonable(resource),
            "extraction": extraction,
        }
        if resumed_from_existing_archive:
            payload["resumed"] = True
        if uploaded_filename:
            payload["uploaded_filename"] = uploaded_filename
        return payload

    import_queue_enabled = _env_bool("STEMMA_BACKGROUND_IMPORT_QUEUE_ENABLED", default=True)
    import_queue_service: BackgroundImportQueueService | None = None
    if import_queue_enabled:
        import_queue_service = BackgroundImportQueueService(
            db_path=paths.db_path,
            queue_dir=paths.stemma_dir / "import_queue",
            run_import_callback=_run_import_file,
        )

    @app.on_event("shutdown")
    def _shutdown_background_import_queue() -> None:
        if import_queue_service is not None:
            import_queue_service.shutdown()

    def _resource_archive_abspath(resource_id: str) -> tuple[dict[str, Any], Path]:
        resource = get_resource_repo().get_by_id(resource_id)
        if resource is None:
            raise HTTPException(status_code=404, detail=f"Resource not found: {resource_id}")

        archive_root = paths.archive_dir.resolve()
        resolved = (archive_root / resource.archived_relpath).resolve()
        if archive_root not in resolved.parents and resolved != archive_root:
            raise HTTPException(status_code=400, detail="Invalid archive path on resource")
        if not resolved.exists():
            raise HTTPException(status_code=404, detail=f"Archived file missing for resource: {resource_id}")
        return _jsonable(resource), resolved

    def _resource_image_abspath(resource_id: str, image_id: str) -> tuple[dict[str, Any], Path]:
        with get_connection(paths.db_path) as conn:
            has_images_table = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'resource_images'"
            ).fetchone()
            if not has_images_table:
                raise HTTPException(status_code=404, detail="No extracted images are stored in this project yet.")
            row = conn.execute(
                """
                SELECT *
                FROM resource_images
                WHERE id = ? AND resource_id = ?
                LIMIT 1
                """,
                (image_id, resource_id),
            ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail=f"Resource image not found: {image_id}")

        stemma_root = paths.stemma_dir.resolve()
        relpath = str(row["output_file_relpath"] or "").strip()
        if not relpath:
            raise HTTPException(status_code=404, detail=f"Resource image path is missing: {image_id}")
        resolved = (stemma_root / relpath).resolve()
        if stemma_root not in resolved.parents and resolved != stemma_root:
            raise HTTPException(status_code=400, detail="Invalid resource image path.")
        if not resolved.exists():
            raise HTTPException(status_code=404, detail=f"Extracted image file missing for image: {image_id}")
        return dict(row), resolved

    def _viewer_resource_images(
        resource_id: str,
        *,
        extraction_run_id: str | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        image_inventory: dict[str, Any] = {
            "table_available": False,
            "resource_id": resource_id,
            "active_run_id": extraction_run_id,
            "resource_total": 0,
            "active_run_total": 0,
            "linked_run_only_total": 0,
            "selected_total": 0,
        }
        with get_connection(paths.db_path) as conn:
            has_images_table = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'resource_images'"
            ).fetchone()
            if not has_images_table:
                return [], image_inventory
            image_inventory["table_available"] = True
            image_inventory["resource_total"] = int(
                conn.execute(
                    """
                    SELECT COUNT(*) AS c
                    FROM resource_images
                    WHERE resource_id = ?
                    """,
                    (resource_id,),
                ).fetchone()["c"]
            )

            active_run_total = 0
            linked_run_only_total = 0
            if extraction_run_id:
                active_run_total = int(
                    conn.execute(
                        """
                        SELECT COUNT(*) AS c
                        FROM resource_images
                        WHERE extraction_run_id = ? AND resource_id = ?
                        """,
                        (extraction_run_id, resource_id),
                    ).fetchone()["c"]
                )
                linked_run_only_total = int(
                    conn.execute(
                        """
                        SELECT COUNT(*) AS c
                        FROM resource_images
                        WHERE extraction_run_id = ? AND resource_id <> ?
                        """,
                        (extraction_run_id, resource_id),
                    ).fetchone()["c"]
                )
            image_inventory["active_run_total"] = active_run_total
            image_inventory["linked_run_only_total"] = linked_run_only_total

            rows = conn.execute(
                """
                SELECT
                  id,
                  resource_id,
                  extraction_run_id,
                  page_index,
                  image_index,
                  source_xref,
                  source_name,
                  source_format,
                  source_width_px,
                  source_height_px,
                  rendered_width_mm,
                  rendered_height_mm,
                  output_width_px,
                  output_height_px,
                  output_file_relpath,
                  output_file_sha256,
                  description_text,
                  description_model,
                  description_generated_at,
                  bbox_json,
                  metadata_json,
                  created_at
                FROM resource_images
                WHERE
                  resource_id = ?
                  OR (? IS NOT NULL AND extraction_run_id = ?)
                ORDER BY
                  CASE
                    WHEN ? IS NOT NULL AND extraction_run_id = ? THEN 0
                    ELSE 1
                  END,
                  page_index ASC,
                  image_index ASC,
                  created_at ASC
                """,
                (
                    resource_id,
                    extraction_run_id,
                    extraction_run_id,
                    extraction_run_id,
                    extraction_run_id,
                ),
            ).fetchall()

        deduped_rows: list[Any] = []
        seen_image_ids: set[str] = set()
        for row in rows:
            image_id = str(row["id"])
            if image_id in seen_image_ids:
                continue
            seen_image_ids.add(image_id)
            deduped_rows.append(row)

        images: list[dict[str, Any]] = []
        for row in deduped_rows:
            page_index = _coerce_non_negative_int(row["page_index"])
            page_number = (page_index + 1) if page_index is not None else None
            image_id = str(row["id"])
            row_resource_id = str(row["resource_id"])
            images.append(
                {
                    "id": image_id,
                    "resource_id": row_resource_id,
                    "extraction_run_id": row["extraction_run_id"],
                    "page_index": page_index,
                    "page_number": page_number,
                    "image_index": _coerce_non_negative_int(row["image_index"]),
                    "source_xref": _coerce_non_negative_int(row["source_xref"]),
                    "source_name": row["source_name"],
                    "source_format": row["source_format"],
                    "source_width_px": _coerce_non_negative_int(row["source_width_px"]),
                    "source_height_px": _coerce_non_negative_int(row["source_height_px"]),
                    "rendered_width_mm": row["rendered_width_mm"],
                    "rendered_height_mm": row["rendered_height_mm"],
                    "output_width_px": _coerce_non_negative_int(row["output_width_px"]),
                    "output_height_px": _coerce_non_negative_int(row["output_height_px"]),
                    "output_file_relpath": row["output_file_relpath"],
                    "output_file_sha256": row["output_file_sha256"],
                    "description_text": row["description_text"],
                    "description_model": row["description_model"],
                    "description_generated_at": row["description_generated_at"],
                    "bbox": _parse_json_blob(row["bbox_json"]),
                    "metadata": _parse_json_blob(row["metadata_json"]),
                    "created_at": row["created_at"],
                    "content_url": f"/api/resources/{row_resource_id}/images/{image_id}/content",
                }
            )
        image_inventory["selected_total"] = len(images)
        return images, image_inventory

    def _viewer_kind(media_type: str | None, original_filename: str | None) -> str:
        media = (media_type or "").lower()
        suffix = Path(original_filename or "").suffix.lower()
        if media == "application/pdf" or suffix == ".pdf":
            return "pdf"
        if media.startswith("image/"):
            return "image"
        if media in {
            "text/csv",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "application/vnd.oasis.opendocument.spreadsheet",
        } or suffix in {".csv", ".xlsx", ".ods"}:
            return "table"
        if media.startswith("text/") or suffix in {
            ".txt",
            ".md",
            ".rst",
            ".adoc",
            ".tex",
            ".json",
            ".xml",
            ".html",
            ".htm",
            ".xhtml",
        }:
            return "text"
        if DoclingAdapter.supports(media_type, original_filename):
            return "structured"
        return "binary"

    def _parse_json_blob(value: object) -> object:
        if value is None:
            return None
        if isinstance(value, (dict, list)):
            return value
        if not isinstance(value, str):
            return None
        try:
            return json.loads(value)
        except Exception:
            return None

    _LEGACY_DOCLING_TABLE_CELL_RE = re.compile(
        r"TableCell\(\s*"
        r"bbox=BoundingBox\(\s*l=(?P<l>-?\d+(?:\.\d+)?),\s*t=(?P<t>-?\d+(?:\.\d+)?),\s*"
        r"r=(?P<r>-?\d+(?:\.\d+)?),\s*b=(?P<b>-?\d+(?:\.\d+)?)"
        r".*?\)\s*,"
        r".*?start_row_offset_(?:idx|index)=(?P<row_start>\d+),\s*"
        r"end_row_offset_(?:idx|index)=(?P<row_end>\d+),\s*"
        r"start_col_offset_(?:idx|index)=(?P<col_start>\d+),\s*"
        r"end_col_offset_(?:idx|index)=(?P<col_end>\d+),\s*"
        r"text='(?P<text>(?:\\'|[^'])*)',\s*"
        r"column_header=(?P<column_header>True|False|true|false),\s*"
        r"row_header=(?P<row_header>True|False|true|false)",
        flags=re.DOTALL,
    )

    def _parse_legacy_docling_table_cells(col_headers: list[object]) -> list[dict[str, Any]]:
        if not isinstance(col_headers, list) or len(col_headers) < 2:
            return []
        raw = col_headers[1]
        if not isinstance(raw, str) or "TableCell(" not in raw:
            return []

        out: list[dict[str, Any]] = []
        for match in _LEGACY_DOCLING_TABLE_CELL_RE.finditer(raw):
            try:
                out.append(
                    {
                        "row_start": int(match.group("row_start")),
                        "row_end": int(match.group("row_end")),
                        "col_start": int(match.group("col_start")),
                        "col_end": int(match.group("col_end")),
                        "text": str(match.group("text")).replace("\\'", "'"),
                        "column_header": str(match.group("column_header")).lower() == "true",
                        "row_header": str(match.group("row_header")).lower() == "true",
                        "bbox": {
                            "x0": float(match.group("l")),
                            "y0": float(match.group("t")),
                            "x1": float(match.group("r")),
                            "y1": float(match.group("b")),
                        },
                    }
                )
            except Exception:
                continue
        return out

    def _extract_legacy_table_dimensions(cells: list[object]) -> tuple[int | None, int | None]:
        if not isinstance(cells, list):
            return None, None
        value_by_pos: dict[tuple[int, int], str] = {}
        for cell in cells:
            if not isinstance(cell, dict):
                continue
            try:
                r = int(cell.get("row_index"))
                c = int(cell.get("col_index"))
            except Exception:
                continue
            value_by_pos[(r, c)] = str(cell.get("value", "")).strip()

        row_count: int | None = None
        col_count: int | None = None
        if value_by_pos.get((0, 0), "").lower() == "num_rows":
            try:
                row_count = int(float(value_by_pos.get((0, 1), "")))
            except Exception:
                row_count = None
        if value_by_pos.get((1, 0), "").lower() == "num_cols":
            try:
                col_count = int(float(value_by_pos.get((1, 1), "")))
            except Exception:
                col_count = None
        return row_count, col_count

    def _repair_legacy_docling_table_payload(
        row_headers: object,
        col_headers: object,
        cells: object,
        bbox: object,
    ) -> tuple[list[str], list[str], list[dict[str, Any]], dict[str, float] | None]:
        safe_row_headers = [str(x) for x in row_headers] if isinstance(row_headers, list) else []
        safe_col_headers = [str(x) for x in col_headers] if isinstance(col_headers, list) else []
        safe_cells = [x for x in cells if isinstance(x, dict)] if isinstance(cells, list) else []
        safe_bbox = bbox if isinstance(bbox, dict) else None

        if not safe_col_headers or safe_col_headers[0] != "table_cells":
            return safe_row_headers, safe_col_headers, safe_cells, safe_bbox

        legacy_cells = _parse_legacy_docling_table_cells(safe_col_headers)
        if not legacy_cells:
            return safe_row_headers, safe_col_headers, safe_cells, safe_bbox

        fallback_rows, fallback_cols = _extract_legacy_table_dimensions(safe_cells)
        grid_rows = max((int(c["row_end"]) for c in legacy_cells), default=0)
        grid_cols = max((int(c["col_end"]) for c in legacy_cells), default=0)
        if isinstance(fallback_rows, int) and fallback_rows > 0:
            grid_rows = max(grid_rows, fallback_rows)
        if isinstance(fallback_cols, int) and fallback_cols > 0:
            grid_cols = max(grid_cols, fallback_cols)
        if grid_rows <= 0 or grid_cols <= 0:
            return safe_row_headers, safe_col_headers, safe_cells, safe_bbox

        grid = [["" for _ in range(grid_cols)] for _ in range(grid_rows)]
        for cell in legacy_cells:
            value = str(cell.get("text", "")).strip()
            if not value:
                continue
            row_start = max(0, min(grid_rows - 1, int(cell.get("row_start", 0))))
            row_end = max(row_start + 1, min(grid_rows, int(cell.get("row_end", row_start + 1))))
            col_start = max(0, min(grid_cols - 1, int(cell.get("col_start", 0))))
            col_end = max(col_start + 1, min(grid_cols, int(cell.get("col_end", col_start + 1))))
            for row_idx in range(row_start, row_end):
                for col_idx in range(col_start, col_end):
                    if not grid[row_idx][col_idx]:
                        grid[row_idx][col_idx] = value

        normalized_col_headers = [(grid[0][col] or f"C{col + 1}") for col in range(grid_cols)] if grid_rows else []
        normalized_row_headers = []
        normalized_cells: list[dict[str, Any]] = []
        for row_idx in range(1, grid_rows):
            normalized_row_headers.append(grid[row_idx][0] or f"R{row_idx}")
            for col_idx in range(grid_cols):
                value = grid[row_idx][col_idx]
                if not value:
                    continue
                normalized_cells.append({"row_index": row_idx - 1, "col_index": col_idx, "value": value})

        normalized_bbox = safe_bbox
        bbox_items = [c.get("bbox") for c in legacy_cells if isinstance(c.get("bbox"), dict)]
        xs0 = [float(b["x0"]) for b in bbox_items if all(k in b for k in ("x0", "y0", "x1", "y1"))]
        ys0 = [float(b["y0"]) for b in bbox_items if all(k in b for k in ("x0", "y0", "x1", "y1"))]
        xs1 = [float(b["x1"]) for b in bbox_items if all(k in b for k in ("x0", "y0", "x1", "y1"))]
        ys1 = [float(b["y1"]) for b in bbox_items if all(k in b for k in ("x0", "y0", "x1", "y1"))]
        if xs0 and ys0 and xs1 and ys1:
            # Prefer legacy cell-derived bounds. They are TOPLEFT coordinates and align with rendered PDF canvas space.
            normalized_bbox = {"x0": min(xs0), "y0": min(ys0), "x1": max(xs1), "y1": max(ys1)}

        if normalized_cells:
            return normalized_row_headers, normalized_col_headers, normalized_cells, normalized_bbox
        return safe_row_headers, safe_col_headers, safe_cells, safe_bbox

    def _annotation_snippet(text: str, start: int, end: int, radius: int = 44) -> str:
        safe_start = max(0, min(int(start), len(text)))
        safe_end = max(safe_start, min(int(end), len(text)))
        left = max(0, safe_start - radius)
        right = min(len(text), safe_end + radius)
        snippet = text[left:right].strip()
        if left > 0:
            snippet = "..." + snippet
        if right < len(text):
            snippet = snippet + "..."
        return snippet

    def _coerce_non_negative_int(value: object) -> int | None:
        try:
            parsed = int(value)  # type: ignore[arg-type]
        except Exception:
            return None
        return parsed if parsed >= 0 else None

    def _coerce_one_based_page_number(value: object) -> int | None:
        try:
            parsed = int(value)  # type: ignore[arg-type]
        except Exception:
            return None
        if parsed < 1:
            return None
        return parsed - 1

    def _segment_page_index_from_attrs(attrs: object) -> int | None:
        if not isinstance(attrs, dict):
            return None

        direct_keys = ("page_index", "pageIndex")
        for key in direct_keys:
            value = _coerce_non_negative_int(attrs.get(key))
            if value is not None:
                return value

        one_based_keys = ("page_number", "pageNumber", "page_no", "pageNo", "page")
        for key in one_based_keys:
            value = _coerce_one_based_page_number(attrs.get(key))
            if value is not None:
                return value

        prov = attrs.get("prov")
        if not isinstance(prov, list):
            prov = attrs.get("provenance")
        if isinstance(prov, list):
            for entry in prov:
                if not isinstance(entry, dict):
                    continue
                for key in direct_keys:
                    value = _coerce_non_negative_int(entry.get(key))
                    if value is not None:
                        return value
                for key in one_based_keys:
                    value = _coerce_one_based_page_number(entry.get(key))
                    if value is not None:
                        return value
        return None

    def _resolve_segment_page_indexes(segments: list[dict[str, Any]]) -> None:
        if not isinstance(segments, list) or not segments:
            return

        anchors: list[tuple[int, int, int]] = []
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            page_index = _coerce_non_negative_int(segment.get("page_index"))
            start_offset = _coerce_non_negative_int(segment.get("start_offset"))
            end_offset = _coerce_non_negative_int(segment.get("end_offset"))
            if page_index is None or start_offset is None or end_offset is None:
                continue
            start = min(start_offset, end_offset)
            end = max(start_offset, end_offset)
            anchors.append((start, end, page_index))

        if not anchors:
            return
        anchors.sort(key=lambda item: (item[0], item[1], item[2]))

        def infer_page_for_range(start: int, end: int) -> int | None:
            mid = (start + end) / 2.0
            overlap_candidates: list[tuple[int, float, int]] = []
            nearest_candidates: list[tuple[float, int, int]] = []
            for anchor_start, anchor_end, anchor_page in anchors:
                if anchor_end > start and anchor_start < end:
                    overlap_span = max(1, anchor_end - anchor_start)
                    overlap_mid_distance = abs(((anchor_start + anchor_end) / 2.0) - mid)
                    overlap_candidates.append((overlap_span, overlap_mid_distance, anchor_page))
                    continue
                if end <= anchor_start:
                    distance = anchor_start - end
                elif start >= anchor_end:
                    distance = start - anchor_end
                else:
                    distance = 0
                nearest_candidates.append((float(distance), abs(((anchor_start + anchor_end) / 2.0) - mid), anchor_page))
            if overlap_candidates:
                overlap_candidates.sort(key=lambda item: (item[0], item[1]))
                return overlap_candidates[0][2]
            if nearest_candidates:
                nearest_candidates.sort(key=lambda item: (item[0], item[1]))
                return nearest_candidates[0][2]
            return None

        for segment in segments:
            if not isinstance(segment, dict):
                continue
            if _coerce_non_negative_int(segment.get("page_index")) is not None:
                continue
            start_offset = _coerce_non_negative_int(segment.get("start_offset"))
            end_offset = _coerce_non_negative_int(segment.get("end_offset"))
            if start_offset is None or end_offset is None:
                continue
            start = min(start_offset, end_offset)
            end = max(start_offset, end_offset)
            inferred_page = infer_page_for_range(start, end)
            if inferred_page is None:
                continue
            segment["page_index"] = inferred_page
            segment["page_number"] = inferred_page + 1

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        page = static_dir / "index.html"
        return HTMLResponse(page.read_text(encoding="utf-8"))

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon() -> FileResponse:
        response = FileResponse(static_dir / "favicon.ico", media_type="image/x-icon")
        response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        response.headers["Content-Type"] = "image/x-icon"
        return response

    @app.get("/apple-touch-icon.png", include_in_schema=False)
    def apple_touch_icon() -> FileResponse:
        response = FileResponse(static_dir / "icons" / "apple-touch-icon.png", media_type="image/png")
        response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        return response

    @app.get("/site.webmanifest", include_in_schema=False)
    def web_manifest() -> FileResponse:
        response = FileResponse(static_dir / "site.webmanifest", media_type="application/manifest+json")
        response.headers["Cache-Control"] = "public, max-age=86400"
        return response

    @app.get("/browserconfig.xml", include_in_schema=False)
    def browser_config() -> FileResponse:
        response = FileResponse(static_dir / "browserconfig.xml", media_type="application/xml")
        response.headers["Cache-Control"] = "public, max-age=86400"
        return response

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

    @app.post("/api/import/queue/enqueue-upload")
    async def api_import_queue_enqueue_upload(file: UploadFile = File(...)) -> dict[str, Any]:
        if import_queue_service is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Background import queue is disabled. "
                    "Set STEMMA_BACKGROUND_IMPORT_QUEUE_ENABLED=1 to enable it."
                ),
            )
        try:
            content = await file.read()
            filename = file.filename or "upload.bin"
            job = import_queue_service.enqueue_upload(
                uploaded_filename=filename,
                content_bytes=content,
                source_uri=f"upload:{filename}",
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, "job": job}

    @app.get("/api/import/queue/status")
    def api_import_queue_status(limit: int = Query(default=200, ge=1, le=50000)) -> dict[str, Any]:
        if import_queue_service is None:
            return {
                "ok": True,
                "disabled": True,
                "queue": {
                    "total": 0,
                    "queued": 0,
                    "processing": 0,
                    "done": 0,
                    "failed": 0,
                    "skipped": 0,
                    "cancelled": 0,
                },
                "jobs": [],
            }
        return import_queue_service.status(limit=limit)

    @app.post("/api/import/queue/jobs/{job_id}/control")
    def api_import_queue_job_control(job_id: str, req: ImportQueueControlRequest) -> dict[str, Any]:
        if import_queue_service is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Background import queue is disabled. "
                    "Set STEMMA_BACKGROUND_IMPORT_QUEUE_ENABLED=1 to enable it."
                ),
            )
        try:
            job = import_queue_service.request_control(job_id=job_id, action=req.action)
        except ValueError as exc:
            detail = str(exc)
            status_code = 404 if detail.lower().startswith("import job not found") else 400
            raise HTTPException(status_code=status_code, detail=detail) from exc
        return {"ok": True, "job": job}

    @app.get("/api/import/queue/jobs/{job_id}/content")
    def api_import_queue_job_content(job_id: str, download: bool = False) -> FileResponse:
        with get_connection(paths.db_path) as conn:
            row = conn.execute(
                """
                SELECT id, original_filename, staged_relpath, resource_id
                FROM import_jobs
                WHERE id = ?
                """,
                (job_id,),
            ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail=f"Import job not found: {job_id}")

        resource_id = str(row["resource_id"] or "").strip()
        if resource_id:
            try:
                resource_payload, archived_path = _resource_archive_abspath(resource_id)
                resource_media = str(resource_payload.get("media_type") or "")
                media_type = (
                    resource_media
                    or mimetypes.guess_type(str(archived_path.name))[0]
                    or "application/octet-stream"
                )
                filename = str(resource_payload.get("original_filename") or archived_path.name)
                safe_filename = filename.replace('"', "")
                response = FileResponse(path=str(archived_path), media_type=media_type, filename=filename)
                response.headers["Content-Disposition"] = (
                    f'attachment; filename="{safe_filename}"'
                    if download
                    else f'inline; filename="{safe_filename}"'
                )
                response.headers["X-Stemma-Import-Job-Id"] = job_id
                return response
            except HTTPException:
                pass

        staged_relpath = str(row["staged_relpath"] or "").strip()
        queue_root = (paths.stemma_dir / "import_queue").resolve()
        staged_path = (queue_root / staged_relpath).resolve()
        if queue_root not in staged_path.parents:
            raise HTTPException(status_code=400, detail="Invalid queue job path.")
        if not staged_path.exists():
            raise HTTPException(status_code=404, detail=f"Queued file no longer available: {job_id}")

        filename = str(row["original_filename"] or staged_path.name)
        media_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        safe_filename = filename.replace('"', "")
        response = FileResponse(path=str(staged_path), media_type=media_type, filename=filename)
        response.headers["Content-Disposition"] = (
            f'attachment; filename="{safe_filename}"'
            if download
            else f'inline; filename="{safe_filename}"'
        )
        response.headers["X-Stemma-Import-Job-Id"] = job_id
        return response

    @app.get("/api/resources")
    def api_resources(limit: int = Query(default=100, ge=1, le=100000)) -> dict[str, Any]:
        resources = [_jsonable(r) for r in get_resource_repo().list(limit=limit)]
        return {"ok": True, "count": len(resources), "resources": resources}

    @app.get("/api/resources/{resource_id}")
    def api_resource_detail(resource_id: str) -> dict[str, Any]:
        resource = get_resource_repo().get_by_id(resource_id)
        if resource is None:
            raise HTTPException(status_code=404, detail=f"Resource not found: {resource_id}")
        return {"ok": True, "resource": _jsonable(resource)}

    @app.post("/api/resources/{resource_id}/title")
    def api_resource_set_title(resource_id: str, req: ResourceTitleUpdateRequest) -> dict[str, Any]:
        repo = get_resource_repo()
        resource = repo.get_by_id(resource_id)
        if resource is None:
            raise HTTPException(status_code=404, detail=f"Resource not found: {resource_id}")

        title = str(req.title or "").strip() or None
        candidates: list[str] = []
        for candidate in req.title_candidates or []:
            value = str(candidate or "").strip()
            if not value:
                continue
            if value in candidates:
                continue
            candidates.append(value)
            if len(candidates) >= 12:
                break
        candidates_json = json.dumps(candidates, ensure_ascii=True) if candidates else None
        repo.update_title_metadata(
            resource_id=resource_id,
            display_title=title,
            title_candidates_json=candidates_json,
        )
        updated = repo.get_by_id(resource_id)
        if updated is None:
            raise HTTPException(status_code=500, detail=f"Unable to read updated resource: {resource_id}")
        return {"ok": True, "resource": _jsonable(updated)}

    @app.get("/api/resources/{resource_id}/content")
    def api_resource_content(resource_id: str, download: bool = False) -> FileResponse:
        resource_payload, archived_path = _resource_archive_abspath(resource_id)
        resource_media = str(resource_payload.get("media_type") or "")
        media_type = resource_media or mimetypes.guess_type(str(archived_path.name))[0] or "application/octet-stream"
        filename = str(resource_payload.get("original_filename") or archived_path.name)
        safe_filename = filename.replace('"', "")
        response = FileResponse(path=str(archived_path), media_type=media_type, filename=filename)
        response.headers["Content-Disposition"] = (
            f'attachment; filename="{safe_filename}"'
            if download
            else f'inline; filename="{safe_filename}"'
        )
        response.headers["X-Stemma-Resource-Id"] = str(resource_payload.get("id") or "")
        return response

    @app.get("/api/resources/{resource_id}/images/{image_id}/content")
    def api_resource_image_content(resource_id: str, image_id: str, download: bool = False) -> FileResponse:
        row, image_path = _resource_image_abspath(resource_id, image_id)
        filename = Path(str(row.get("output_file_relpath") or image_path.name)).name
        media_type = mimetypes.guess_type(filename)[0] or (
            "image/avif" if filename.lower().endswith(".avif") else "application/octet-stream"
        )
        safe_filename = filename.replace('"', "")
        response = FileResponse(path=str(image_path), media_type=media_type, filename=filename)
        response.headers["Content-Disposition"] = (
            f'attachment; filename="{safe_filename}"'
            if download
            else f'inline; filename="{safe_filename}"'
        )
        response.headers["X-Stemma-Resource-Id"] = resource_id
        response.headers["X-Stemma-Resource-Image-Id"] = image_id
        return response

    @app.get("/api/viewer/document")
    def api_viewer_document(
        resource_id: str | None = None,
        resource_digest: str | None = None,
        table_limit: int = Query(default=800, ge=1, le=200000),
        segment_limit: int = Query(default=12000, ge=1, le=300000),
        annotation_limit: int = Query(default=12000, ge=1, le=300000),
    ) -> dict[str, Any]:
        try:
            target_id = _resolve_extract_resource_id(resource_id, resource_digest)
            resource = get_resource_repo().get_by_id(target_id)
            if resource is None:
                raise HTTPException(status_code=404, detail=f"Resource not found: {target_id}")

            _, archived_path = _resource_archive_abspath(target_id)
            stat = archived_path.stat()
        except Exception as exc:
            if isinstance(exc, HTTPException):
                raise
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        dump: dict[str, Any] | None = None
        extraction_error: str | None = None
        try:
            dump = get_extraction_service().build_dump(
                resource_id=target_id,
                segment_limit=segment_limit,
                annotation_limit=annotation_limit,
                table_limit=table_limit,
            )
        except Exception as exc:
            extraction_error = str(exc)

        run_payload = dump.get("run") if isinstance(dump, dict) else None
        doc_payload = dump.get("document_text") if isinstance(dump, dict) else None
        text_content = str(doc_payload.get("text_content") or "") if isinstance(doc_payload, dict) else ""

        parsed_tables: list[dict[str, Any]] = []
        parsed_segments: list[dict[str, Any]] = []
        parsed_annotations: list[dict[str, Any]] = []
        active_run_id = str(run_payload.get("id")) if isinstance(run_payload, dict) and run_payload.get("id") else None
        parsed_images, image_inventory = _viewer_resource_images(
            target_id,
            extraction_run_id=active_run_id,
        )

        pages_with_tables: set[int] = set()
        pages_with_geometry: set[int] = set()
        bbox_table_count = 0
        bbox_segment_count = 0

        if isinstance(dump, dict):
            for row in dump.get("tables", []) or []:
                row_headers = _parse_json_blob(row.get("row_headers_json"))
                col_headers = _parse_json_blob(row.get("col_headers_json"))
                cells = _parse_json_blob(row.get("cells_json"))
                bbox = _parse_json_blob(row.get("bbox_json"))
                fixed_row_headers, fixed_col_headers, fixed_cells, fixed_bbox = _repair_legacy_docling_table_payload(
                    row_headers=row_headers,
                    col_headers=col_headers,
                    cells=cells,
                    bbox=bbox,
                )
                page_index = row.get("page_index")
                page_number = (int(page_index) + 1) if isinstance(page_index, int) else None
                if isinstance(page_index, int):
                    pages_with_tables.add(page_index)
                if isinstance(fixed_bbox, dict):
                    bbox_table_count += 1
                    if isinstance(page_index, int):
                        pages_with_geometry.add(page_index)
                parsed_tables.append(
                    {
                        "id": row.get("id"),
                        "table_id": row.get("table_id"),
                        "caption": row.get("caption"),
                        "page_index": page_index,
                        "page_number": page_number,
                        "row_headers": fixed_row_headers,
                        "col_headers": fixed_col_headers,
                        "cells": fixed_cells,
                        "bbox": fixed_bbox if isinstance(fixed_bbox, dict) else None,
                        "created_at": row.get("created_at"),
                    }
                )

            for row in dump.get("segments", []) or []:
                bbox = _parse_json_blob(row.get("bbox_json"))
                attrs = _parse_json_blob(row.get("attrs_json"))
                page_index = _coerce_non_negative_int(row.get("page_index"))
                if page_index is None:
                    page_index = _segment_page_index_from_attrs(attrs)
                page_number = (int(page_index) + 1) if isinstance(page_index, int) else None
                if isinstance(bbox, dict):
                    bbox_segment_count += 1
                    if isinstance(page_index, int):
                        pages_with_geometry.add(page_index)
                parsed_segments.append(
                    {
                        "id": row.get("id"),
                        "segment_type": row.get("segment_type"),
                        "start_offset": row.get("start_offset"),
                        "end_offset": row.get("end_offset"),
                        "page_index": page_index,
                        "page_number": page_number,
                        "order_index": row.get("order_index"),
                        "bbox": bbox if isinstance(bbox, dict) else None,
                        "attrs": attrs if isinstance(attrs, dict) else None,
                        "created_at": row.get("created_at"),
                    }
                )
            _resolve_segment_page_indexes(parsed_segments)
            for segment in parsed_segments:
                if not isinstance(segment, dict):
                    continue
                page_index = _coerce_non_negative_int(segment.get("page_index"))
                if page_index is None:
                    continue
                if isinstance(segment.get("bbox"), dict):
                    pages_with_geometry.add(page_index)

            for row in dump.get("annotations", []) or []:
                spans = row.get("spans") if isinstance(row.get("spans"), list) else []
                snippet = None
                if spans:
                    first_span = spans[0]
                    if isinstance(first_span, dict):
                        start = first_span.get("start")
                        end = first_span.get("end")
                        if isinstance(start, int) and isinstance(end, int) and text_content:
                            snippet = _annotation_snippet(text_content, start, end)
                parsed_annotations.append(
                    {
                        "id": row.get("id"),
                        "layer": row.get("layer"),
                        "category": row.get("category"),
                        "label": row.get("label"),
                        "confidence": row.get("confidence"),
                        "source": row.get("source"),
                        "attrs": _parse_json_blob(row.get("attrs_json")),
                        "spans": spans,
                        "snippet": snippet,
                        "created_at": row.get("created_at"),
                    }
                )

        content_url = f"/api/resources/{target_id}/content"
        return {
            "ok": True,
            "resource": _jsonable(resource),
            "content": {
                "url": content_url,
                "kind": _viewer_kind(resource.media_type, resource.original_filename),
                "size_bytes": int(stat.st_size),
                "last_modified_epoch": int(stat.st_mtime),
            },
            "metadata": {
                "resource_id": resource.id,
                "digest_sha256": resource.digest_sha256,
                "media_type": resource.media_type,
                "original_filename": resource.original_filename,
                "display_title": resource.display_title,
                "title_candidates_json": resource.title_candidates_json,
                "source_uri": resource.source_uri,
                "download_url": resource.download_url,
                "download_urls_json": _parse_json_blob(resource.download_urls_json),
                "archived_relpath": resource.archived_relpath,
                "ingested_at": resource.ingested_at,
            },
            "extraction": {
                "available": dump is not None,
                "error": extraction_error,
                "run": run_payload,
                "document_text": (
                    {
                        "id": doc_payload.get("id"),
                        "char_count": doc_payload.get("char_count"),
                        "text_digest_sha256": doc_payload.get("text_digest_sha256"),
                        "created_at": doc_payload.get("created_at"),
                    }
                    if isinstance(doc_payload, dict)
                    else None
                ),
                "counts": {
                    "tables": len(parsed_tables),
                    "segments": len(parsed_segments),
                    "annotations": len(parsed_annotations),
                    "images": len(parsed_images),
                    "bbox_tables": bbox_table_count,
                    "bbox_segments": bbox_segment_count,
                },
                "image_inventory": image_inventory,
                "pages_with_tables": sorted(x + 1 for x in pages_with_tables),
                "pages_with_geometry": sorted(x + 1 for x in pages_with_geometry),
                "overlay_ready": bool(bbox_table_count or bbox_segment_count),
            },
            "tables": parsed_tables,
            "segments": parsed_segments,
            "annotations": parsed_annotations,
            "images": parsed_images,
            "document_text_excerpt": text_content[:4000],
        }

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

    def _db_table_locate_payload(name: str, column: str, value: str, page_size: int) -> dict[str, Any]:
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

            quoted_table = _quote_identifier(name)
            col_rows = conn.execute(f"PRAGMA table_info({quoted_table})").fetchall()
            columns = {str(c["name"]) for c in col_rows}
            if column not in columns:
                raise HTTPException(status_code=404, detail=f"Column not found on {name}: {column}")

            total_rows = int(conn.execute(f"SELECT COUNT(*) AS c FROM {quoted_table}").fetchone()["c"])
            quoted_col = _quote_identifier(column)

            # SQLite rowid gives us stable paging position for this read path (SELECT * LIMIT/OFFSET).
            hit = conn.execute(
                f"SELECT rowid AS rid FROM {quoted_table} WHERE {quoted_col} = ? LIMIT 1",
                (value,),
            ).fetchone()
            if hit is None:
                return {
                    "ok": True,
                    "table": name,
                    "column": column,
                    "value": value,
                    "found": False,
                    "offset": 0,
                    "page_size": page_size,
                    "total_rows": total_rows,
                }

            rid = int(hit["rid"])
            rows_before = int(
                conn.execute(
                    f"SELECT COUNT(*) AS c FROM {quoted_table} WHERE rowid < ?",
                    (rid,),
                ).fetchone()["c"]
            )
            offset = (rows_before // page_size) * page_size

        return {
            "ok": True,
            "table": name,
            "column": column,
            "value": value,
            "found": True,
            "offset": offset,
            "page_size": page_size,
            "total_rows": total_rows,
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

    @app.get("/api/db/table/locate")
    @app.get("/api/db/table/locate/")
    def api_db_table_locate(
        name: str = Query(..., min_length=1),
        column: str = Query(..., min_length=1),
        value: str = Query(...),
        page_size: int = Query(default=100, ge=1, le=1000),
    ) -> dict[str, Any]:
        return _db_table_locate_payload(name=name, column=column, value=value, page_size=page_size)

    @app.get("/api/dashboard/summary")
    def api_dashboard_summary(include_doctor: bool = Query(default=False)) -> dict[str, Any]:
        def _safe_ratio(numerator: int, denominator: int) -> float:
            if denominator <= 0:
                return 0.0
            return round((max(0, numerator) / denominator) * 100, 2)

        def _path_size_bytes(path: Path) -> int:
            try:
                if not path.exists():
                    return 0
                if path.is_file():
                    return int(path.stat().st_size)
            except OSError:
                return 0

            total = 0
            stack: list[Path] = [path]
            while stack:
                current = stack.pop()
                try:
                    with os.scandir(current) as entries:
                        for entry in entries:
                            try:
                                if entry.is_symlink():
                                    continue
                                if entry.is_file(follow_symlinks=False):
                                    total += int(entry.stat(follow_symlinks=False).st_size)
                                elif entry.is_dir(follow_symlinks=False):
                                    stack.append(Path(entry.path))
                            except OSError:
                                continue
                except OSError:
                    continue
            return total

        db_main_path = paths.db_path
        db_wal_path = Path(f"{db_main_path}-wal")
        db_shm_path = Path(f"{db_main_path}-shm")
        archive_path = paths.archive_dir
        qdrant_path = paths.qdrant_dir
        device_anchor = db_main_path.parent if db_main_path.parent.exists() else paths.project_root
        try:
            device_usage = shutil.disk_usage(device_anchor)
            device_total = int(device_usage.total)
            device_free = int(device_usage.free)
            device_used = int(device_total - device_free)
        except OSError:
            device_total = 0
            device_free = 0
            device_used = 0
        project_storage = {
            "db_main_path": str(db_main_path),
            "db_main_bytes": _path_size_bytes(db_main_path),
            "db_wal_path": str(db_wal_path),
            "db_wal_bytes": _path_size_bytes(db_wal_path),
            "db_shm_path": str(db_shm_path),
            "db_shm_bytes": _path_size_bytes(db_shm_path),
            "archive_dir": str(archive_path),
            "archive_bytes": _path_size_bytes(archive_path),
            "qdrant_dir": str(qdrant_path),
            "qdrant_bytes": _path_size_bytes(qdrant_path),
            "device_anchor_path": str(device_anchor),
            "device_total_bytes": device_total,
            "device_used_bytes": device_used,
            "device_free_bytes": device_free,
        }

        now_dt = datetime.now(timezone.utc).replace(microsecond=0)
        recent_7d = (now_dt - timedelta(days=7)).isoformat()
        recent_30d = (now_dt - timedelta(days=30)).isoformat()
        activity_window_days = 14
        activity_start_day = (now_dt - timedelta(days=activity_window_days - 1)).date().isoformat()
        activity_start_iso = f"{activity_start_day}T00:00:00+00:00"

        with get_connection(paths.db_path) as conn:
            counts = {
                "resources": int(conn.execute("SELECT COUNT(*) AS c FROM resources").fetchone()["c"]),
                "citations": int(conn.execute("SELECT COUNT(*) AS c FROM citations").fetchone()["c"]),
                "references": int(conn.execute("SELECT COUNT(*) AS c FROM reference_entries").fetchone()["c"]),
                "reference_resources": int(conn.execute("SELECT COUNT(*) AS c FROM reference_resources").fetchone()["c"]),
                "claim_sets": int(conn.execute("SELECT COUNT(*) AS c FROM claim_sets").fetchone()["c"]),
                "claims": int(conn.execute("SELECT COUNT(*) AS c FROM claims").fetchone()["c"]),
                "evidence_items": int(conn.execute("SELECT COUNT(*) AS c FROM evidence_items").fetchone()["c"]),
                "evidence_selectors": int(conn.execute("SELECT COUNT(*) AS c FROM evidence_selectors").fetchone()["c"]),
                "claim_evidence_bindings": int(
                    conn.execute("SELECT COUNT(*) AS c FROM claim_evidence_bindings").fetchone()["c"]
                ),
                "extraction_runs": int(conn.execute("SELECT COUNT(*) AS c FROM extraction_runs").fetchone()["c"]),
                "extracted_tables": int(conn.execute("SELECT COUNT(*) AS c FROM extracted_tables").fetchone()["c"]),
                "document_texts": int(conn.execute("SELECT COUNT(*) AS c FROM document_texts").fetchone()["c"]),
                "text_segments": int(conn.execute("SELECT COUNT(*) AS c FROM text_segments").fetchone()["c"]),
                "text_annotations": int(conn.execute("SELECT COUNT(*) AS c FROM text_annotations").fetchone()["c"]),
                "text_annotation_spans": int(conn.execute("SELECT COUNT(*) AS c FROM text_annotation_spans").fetchone()["c"]),
                "text_annotation_relations": int(
                    conn.execute("SELECT COUNT(*) AS c FROM text_annotation_relations").fetchone()["c"]
                ),
                "resource_images": int(conn.execute("SELECT COUNT(*) AS c FROM resource_images").fetchone()["c"]),
                "verification_runs": int(conn.execute("SELECT COUNT(*) AS c FROM verification_runs").fetchone()["c"]),
                "verification_results": int(conn.execute("SELECT COUNT(*) AS c FROM verification_results").fetchone()["c"]),
                "propositions": int(conn.execute("SELECT COUNT(*) AS c FROM propositions").fetchone()["c"]),
                "assertion_events": int(conn.execute("SELECT COUNT(*) AS c FROM assertion_events").fetchone()["c"]),
                "argument_relations": int(conn.execute("SELECT COUNT(*) AS c FROM argument_relations").fetchone()["c"]),
                "vector_index_runs": int(conn.execute("SELECT COUNT(*) AS c FROM vector_index_runs").fetchone()["c"]),
                "vector_chunks": int(conn.execute("SELECT COUNT(*) AS c FROM vector_chunks").fetchone()["c"]),
                "import_jobs": int(conn.execute("SELECT COUNT(*) AS c FROM import_jobs").fetchone()["c"]),
            }

            resource_profile_row = conn.execute(
                """
                SELECT
                  COALESCE(SUM(size_bytes), 0) AS resource_bytes_total,
                  COALESCE(AVG(size_bytes), 0) AS resource_bytes_avg,
                  MIN(ingested_at) AS first_ingested_at,
                  MAX(ingested_at) AS last_ingested_at,
                  COALESCE(SUM(CASE WHEN ingested_at >= ? THEN 1 ELSE 0 END), 0) AS resources_ingested_7d,
                  COALESCE(SUM(CASE WHEN ingested_at >= ? THEN 1 ELSE 0 END), 0) AS resources_ingested_30d,
                  COALESCE(SUM(
                    CASE
                      WHEN LOWER(COALESCE(download_url, '')) LIKE 'http://%'
                        OR LOWER(COALESCE(download_url, '')) LIKE 'https://%'
                      THEN 1 ELSE 0
                    END
                  ), 0) AS resources_with_external_source
                FROM resources
                """,
                (recent_7d, recent_30d),
            ).fetchone()

            text_volume_row = conn.execute(
                """
                SELECT
                  COALESCE(SUM(char_count), 0) AS chars_total,
                  COALESCE(AVG(char_count), 0) AS chars_avg,
                  COALESCE(MAX(char_count), 0) AS chars_max
                FROM document_texts
                """
            ).fetchone()

            coverage_row = conn.execute(
                """
                SELECT
                  (SELECT COUNT(DISTINCT resource_id) FROM extraction_runs WHERE LOWER(status) = 'success')
                    AS resources_with_successful_extraction,
                  (SELECT COUNT(DISTINCT resource_id) FROM document_texts) AS resources_with_text,
                  (SELECT COUNT(DISTINCT resource_id) FROM vector_chunks) AS resources_with_vector_chunks,
                  (SELECT COUNT(DISTINCT resource_id) FROM reference_resources) AS resources_with_references,
                  (SELECT COUNT(DISTINCT resource_id) FROM evidence_items) AS resources_with_evidence,
                  (SELECT COUNT(DISTINCT claim_id) FROM claim_evidence_bindings) AS claims_with_bindings,
                  (SELECT COUNT(DISTINCT evidence_id) FROM evidence_selectors) AS evidence_with_selectors,
                  (SELECT COUNT(DISTINCT claim_id) FROM verification_results) AS claims_with_verification
                """
            ).fetchone()

            documents_by_type_rows = conn.execute(
                """
                SELECT
                  CASE
                    WHEN LOWER(COALESCE(media_type, '')) LIKE 'application/pdf%' THEN 'PDF'
                    WHEN LOWER(COALESCE(media_type, '')) LIKE 'image/%' THEN 'Image'
                    WHEN LOWER(COALESCE(media_type, '')) IN (
                      'application/msword',
                      'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                      'application/vnd.oasis.opendocument.text',
                      'application/rtf'
                    ) THEN 'Word Document'
                    WHEN LOWER(COALESCE(media_type, '')) IN (
                      'application/vnd.ms-excel',
                      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                      'application/vnd.oasis.opendocument.spreadsheet',
                      'text/csv'
                    ) THEN 'Spreadsheet / CSV'
                    WHEN LOWER(COALESCE(media_type, '')) IN (
                      'application/vnd.ms-powerpoint',
                      'application/vnd.openxmlformats-officedocument.presentationml.presentation',
                      'application/vnd.oasis.opendocument.presentation'
                    ) THEN 'Presentation'
                    WHEN LOWER(COALESCE(media_type, '')) LIKE 'text/html%' THEN 'HTML'
                    WHEN LOWER(COALESCE(media_type, '')) IN ('application/json', 'application/xml', 'text/xml')
                      THEN 'Structured Text'
                    WHEN LOWER(COALESCE(media_type, '')) LIKE 'text/%' THEN 'Plain Text'
                    ELSE 'Other'
                  END AS label,
                  COUNT(*) AS c,
                  COALESCE(SUM(size_bytes), 0) AS bytes
                FROM resources
                GROUP BY label
                ORDER BY c DESC, bytes DESC, label ASC
                """
            ).fetchall()

            media_type_rows = conn.execute(
                """
                SELECT
                  CASE
                    WHEN TRIM(COALESCE(media_type, '')) = '' THEN 'unknown'
                    ELSE LOWER(media_type)
                  END AS media_type,
                  COUNT(*) AS c
                FROM resources
                GROUP BY media_type
                ORDER BY c DESC, media_type ASC
                LIMIT 8
                """
            ).fetchall()

            claim_status_rows = conn.execute(
                """
                SELECT LOWER(COALESCE(status, 'unknown')) AS status, COUNT(*) AS c
                FROM claims
                GROUP BY status
                ORDER BY c DESC, status ASC
                """
            ).fetchall()

            extraction_status_rows = conn.execute(
                """
                SELECT LOWER(COALESCE(status, 'unknown')) AS status, COUNT(*) AS c
                FROM extraction_runs
                GROUP BY status
                ORDER BY c DESC, status ASC
                """
            ).fetchall()

            vector_run_status_rows = conn.execute(
                """
                SELECT LOWER(COALESCE(status, 'unknown')) AS status, COUNT(*) AS c
                FROM vector_index_runs
                GROUP BY status
                ORDER BY c DESC, status ASC
                """
            ).fetchall()

            verification_result_rows = conn.execute(
                """
                SELECT LOWER(COALESCE(status, 'unknown')) AS status, COUNT(*) AS c
                FROM verification_results
                GROUP BY status
                ORDER BY c DESC, status ASC
                """
            ).fetchall()

            import_job_status_rows = conn.execute(
                """
                SELECT LOWER(COALESCE(status, 'unknown')) AS status, COUNT(*) AS c
                FROM import_jobs
                GROUP BY status
                ORDER BY c DESC, status ASC
                """
            ).fetchall()

            ingest_activity_rows = conn.execute(
                """
                SELECT substr(ingested_at, 1, 10) AS day, COUNT(*) AS c
                FROM resources
                WHERE ingested_at >= ?
                GROUP BY day
                ORDER BY day ASC
                """,
                (activity_start_iso,),
            ).fetchall()

            extraction_activity_rows = conn.execute(
                """
                SELECT substr(created_at, 1, 10) AS day, COUNT(*) AS c
                FROM extraction_runs
                WHERE created_at >= ?
                GROUP BY day
                ORDER BY day ASC
                """,
                (activity_start_iso,),
            ).fetchall()

            verification_activity_rows = conn.execute(
                """
                SELECT substr(created_at, 1, 10) AS day, COUNT(*) AS c
                FROM verification_runs
                WHERE created_at >= ?
                GROUP BY day
                ORDER BY day ASC
                """,
                (activity_start_iso,),
            ).fetchall()

            claim_activity_rows = conn.execute(
                """
                SELECT substr(created_at, 1, 10) AS day, COUNT(*) AS c
                FROM claims
                WHERE created_at >= ?
                GROUP BY day
                ORDER BY day ASC
                """,
                (activity_start_iso,),
            ).fetchall()

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

        resources_total = int(counts["resources"])
        claims_total = int(counts["claims"])
        evidence_total = int(counts["evidence_items"])

        resources_with_external_source = int(resource_profile_row["resources_with_external_source"] or 0)
        resources_with_successful_extraction = int(coverage_row["resources_with_successful_extraction"] or 0)
        resources_with_text = int(coverage_row["resources_with_text"] or 0)
        resources_with_vector_chunks = int(coverage_row["resources_with_vector_chunks"] or 0)
        resources_with_references = int(coverage_row["resources_with_references"] or 0)
        resources_with_evidence = int(coverage_row["resources_with_evidence"] or 0)
        claims_with_bindings = int(coverage_row["claims_with_bindings"] or 0)
        evidence_with_selectors = int(coverage_row["evidence_with_selectors"] or 0)
        claims_with_verification = int(coverage_row["claims_with_verification"] or 0)

        documents_by_type = [
            {
                "label": str(row["label"]),
                "count": int(row["c"]),
                "bytes": int(row["bytes"] or 0),
            }
            for row in documents_by_type_rows
        ]
        media_type_counts = [
            {
                "label": str(row["media_type"]),
                "count": int(row["c"]),
            }
            for row in media_type_rows
        ]

        def _rows_to_dict(rows: list[Any], key_name: str = "status") -> dict[str, int]:
            result: dict[str, int] = {}
            for row in rows:
                key = str(row[key_name])
                result[key] = int(row["c"])
            return result

        claim_status_counts = _rows_to_dict(claim_status_rows)
        extraction_status_counts = _rows_to_dict(extraction_status_rows)
        vector_run_status_counts_all = _rows_to_dict(vector_run_status_rows)
        verification_result_status_counts_all = _rows_to_dict(verification_result_rows)
        import_job_status_counts = _rows_to_dict(import_job_status_rows)

        ingest_activity_map = {str(row["day"]): int(row["c"]) for row in ingest_activity_rows if row["day"]}
        extraction_activity_map = {str(row["day"]): int(row["c"]) for row in extraction_activity_rows if row["day"]}
        verification_activity_map = {str(row["day"]): int(row["c"]) for row in verification_activity_rows if row["day"]}
        claim_activity_map = {str(row["day"]): int(row["c"]) for row in claim_activity_rows if row["day"]}

        activity_days = [
            (now_dt - timedelta(days=offset)).date().isoformat()
            for offset in range(activity_window_days - 1, -1, -1)
        ]
        activity_series: list[dict[str, int | str]] = []
        for day in activity_days:
            ingested = ingest_activity_map.get(day, 0)
            extracted = extraction_activity_map.get(day, 0)
            verified = verification_activity_map.get(day, 0)
            claims_added = claim_activity_map.get(day, 0)
            activity_series.append(
                {
                    "day": day,
                    "ingested": ingested,
                    "extracted": extracted,
                    "verified": verified,
                    "claims_added": claims_added,
                    "total": ingested + extracted + verified + claims_added,
                }
            )

        archive_payload = {
            "resources_total": resources_total,
            "resource_bytes_total": int(resource_profile_row["resource_bytes_total"] or 0),
            "resource_bytes_avg": int(round(float(resource_profile_row["resource_bytes_avg"] or 0))),
            "first_ingested_at": resource_profile_row["first_ingested_at"],
            "last_ingested_at": resource_profile_row["last_ingested_at"],
            "resources_ingested_7d": int(resource_profile_row["resources_ingested_7d"] or 0),
            "resources_ingested_30d": int(resource_profile_row["resources_ingested_30d"] or 0),
            "resources_with_external_source": resources_with_external_source,
            "resources_local_only": max(0, resources_total - resources_with_external_source),
            "resources_with_references": resources_with_references,
            "resources_with_successful_extraction": resources_with_successful_extraction,
            "resources_with_text": resources_with_text,
            "resources_with_vector_chunks": resources_with_vector_chunks,
            "resources_with_evidence": resources_with_evidence,
            "claims_with_bindings": claims_with_bindings,
            "claims_with_verification": claims_with_verification,
            "evidence_with_selectors": evidence_with_selectors,
            "text_char_total": int(text_volume_row["chars_total"] or 0),
            "text_char_avg": int(round(float(text_volume_row["chars_avg"] or 0))),
            "text_char_max": int(text_volume_row["chars_max"] or 0),
            "coverage": {
                "extraction_success_pct": _safe_ratio(resources_with_successful_extraction, resources_total),
                "text_coverage_pct": _safe_ratio(resources_with_text, resources_total),
                "vector_coverage_pct": _safe_ratio(resources_with_vector_chunks, resources_total),
                "reference_coverage_pct": _safe_ratio(resources_with_references, resources_total),
                "evidence_coverage_pct": _safe_ratio(resources_with_evidence, resources_total),
                "claim_binding_pct": _safe_ratio(claims_with_bindings, claims_total),
                "claim_verification_pct": _safe_ratio(claims_with_verification, claims_total),
                "evidence_selector_pct": _safe_ratio(evidence_with_selectors, evidence_total),
            },
            "documents_by_type": documents_by_type,
            "media_type_counts": media_type_counts,
            "claim_status_counts": claim_status_counts,
            "extraction_status_counts": extraction_status_counts,
            "verification_result_status_counts": verification_result_status_counts_all,
            "vector_run_status_counts": vector_run_status_counts_all,
            "import_job_status_counts": import_job_status_counts,
            "activity": {
                "window_days": activity_window_days,
                "series": activity_series,
            },
        }

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
                "storage": project_storage,
            },
            "counts": counts,
            "archive": archive_payload,
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

    @app.get("/api/sources/resources")
    def api_sources_resources(
        limit: int = Query(default=500, ge=1, le=100000),
        q: str | None = Query(default=None),
        missing_only: bool = Query(default=False),
    ) -> dict[str, Any]:
        # Ensure legacy projects are migrated with source metadata columns.
        get_resource_repo()
        where: list[str] = []
        params: list[object] = []
        if q:
            like = f"%{q.strip()}%"
            where.append(
                "("
                "r.id LIKE ? OR "
                "COALESCE(r.display_title, '') LIKE ? OR "
                "COALESCE(r.original_filename, '') LIKE ? OR "
                "COALESCE(r.download_url, '') LIKE ? OR "
                "COALESCE(r.source_uri, '') LIKE ?"
                ")"
            )
            params.extend([like, like, like, like, like])
        if missing_only:
            where.append(
                "("
                "LOWER(COALESCE(r.download_url, '')) NOT LIKE 'http://%' AND "
                "LOWER(COALESCE(r.download_url, '')) NOT LIKE 'https://%'"
                ")"
            )
        where_sql = f"WHERE {' AND '.join(where)}" if where else ""
        with get_connection(paths.db_path) as conn:
            has_document_texts = bool(
                conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'document_texts'"
                ).fetchone()
            )
            has_text_expr = (
                "EXISTS(SELECT 1 FROM document_texts dt WHERE dt.resource_id = r.id LIMIT 1)"
                if has_document_texts
                else "0"
            )
            rows = conn.execute(
                f"""
                SELECT
                  r.id,
                  r.display_title,
                  r.original_filename,
                  r.media_type,
                  r.source_uri,
                  r.download_url,
                  r.download_urls_json,
                  r.ingested_at,
                  {has_text_expr} AS has_text
                FROM resources r
                {where_sql}
                ORDER BY r.ingested_at DESC
                LIMIT ?
                """,
                (*params, limit),
            ).fetchall()
        resources: list[dict[str, Any]] = []
        for row in rows:
            urls_blob = _parse_json_blob(row["download_urls_json"])
            urls = urls_blob if isinstance(urls_blob, list) else []
            primary = str(row["download_url"] or "").strip()
            if primary and primary not in urls:
                urls.insert(0, primary)
            resources.append(
                {
                    "id": str(row["id"]),
                    "display_title": row["display_title"],
                    "original_filename": row["original_filename"],
                    "media_type": row["media_type"],
                    "source_uri": row["source_uri"],
                    "download_url": row["download_url"],
                    "download_urls": urls,
                    "ingested_at": row["ingested_at"],
                    "has_text": bool(row["has_text"]),
                    "has_external_source": bool(
                        str(row["download_url"] or "").lower().startswith("http://")
                        or str(row["download_url"] or "").lower().startswith("https://")
                    ),
                }
            )
        return {"ok": True, "count": len(resources), "resources": resources}

    @app.get("/api/sources/resources/{resource_id}")
    def api_sources_resource_detail(resource_id: str) -> dict[str, Any]:
        resource = get_resource_repo().get_by_id(resource_id)
        if resource is None:
            raise HTTPException(status_code=404, detail=f"Resource not found: {resource_id}")

        with get_connection(paths.db_path) as conn:
            has_reference_entries = bool(
                conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'reference_entries'"
                ).fetchone()
            )
            has_reference_resources = bool(
                conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'reference_resources'"
                ).fetchone()
            )
            has_document_texts = bool(
                conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'document_texts'"
                ).fetchone()
            )
            has_extraction_runs = bool(
                conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'extraction_runs'"
                ).fetchone()
            )

            if has_reference_entries and has_reference_resources:
                ref_rows = conn.execute(
                    """
                    SELECT re.id, re.cite_id, re.title, re.author, re.year, re.doi, re.url
                    FROM reference_entries re
                    INNER JOIN reference_resources rr ON rr.reference_id = re.id
                    WHERE rr.resource_id = ?
                    ORDER BY re.imported_at DESC, re.id DESC
                    LIMIT 100
                    """,
                    (resource_id,),
                ).fetchall()
            else:
                ref_rows = []

            if has_document_texts and has_extraction_runs:
                text_row = conn.execute(
                    """
                    SELECT dt.text_content, dt.char_count, dt.text_digest_sha256, er.id AS extraction_run_id, er.created_at
                    FROM document_texts dt
                    INNER JOIN extraction_runs er ON er.id = dt.extraction_run_id
                    WHERE dt.resource_id = ?
                    ORDER BY er.created_at DESC
                    LIMIT 1
                    """,
                    (resource_id,),
                ).fetchone()
            else:
                text_row = None

        parsed_urls = _parse_json_blob(resource.download_urls_json)
        source_candidates = parsed_urls if isinstance(parsed_urls, list) else []
        if resource.download_url and resource.download_url not in source_candidates:
            source_candidates.insert(0, resource.download_url)

        extracted_preview = ""
        extracted_char_count = 0
        extraction_run_id = None
        text_digest = None
        extraction_created_at = None
        if text_row is not None:
            extracted_full = str(text_row["text_content"] or "")
            extracted_char_count = int(text_row["char_count"] or len(extracted_full))
            extracted_preview = extracted_full[:4000]
            extraction_run_id = text_row["extraction_run_id"]
            text_digest = text_row["text_digest_sha256"]
            extraction_created_at = text_row["created_at"]

        references = [
            {
                "id": str(row["id"]),
                "cite_id": row["cite_id"],
                "title": row["title"],
                "author": row["author"],
                "year": row["year"],
                "doi": row["doi"],
                "url": row["url"],
            }
            for row in ref_rows
        ]
        return {
            "ok": True,
            "resource": _jsonable(resource),
            "sources": {
                "primary_url": resource.download_url,
                "source_uri": resource.source_uri,
                "candidates": source_candidates,
                "has_external_source": bool(
                    str(resource.download_url or "").lower().startswith("http://")
                    or str(resource.download_url or "").lower().startswith("https://")
                ),
            },
            "references": references,
            "extraction": {
                "available": text_row is not None,
                "char_count": extracted_char_count,
                "text_digest_sha256": text_digest,
                "extraction_run_id": extraction_run_id,
                "created_at": extraction_created_at,
                "text_preview": extracted_preview,
            },
        }

    @app.post("/api/sources/resources/{resource_id}/recover")
    def api_sources_resource_recover(resource_id: str, req: SourceRecoverRequest) -> dict[str, Any]:
        if get_resource_repo().get_by_id(resource_id) is None:
            raise HTTPException(status_code=404, detail=f"Resource not found: {resource_id}")
        try:
            recovery_mod = get_source_recovery_module()
            manifest_raw = str(req.manifest_root or "").strip() or "/Volumes/X10/data"
            manifest_root = Path(manifest_raw).expanduser()
            if not req.use_manifest_scan:
                manifest_root = Path("/__stemma_manifest_scan_disabled__")
            shallow = [Path(item).expanduser() for item in (req.shallow_search_dirs or []) if str(item).strip()] or None
            deep = [Path(item).expanduser() for item in (req.deep_search_dirs or []) if str(item).strip()] or None
            result = recovery_mod.recover_resource_by_id(
                db_path=paths.db_path,
                archive_dir=paths.archive_dir,
                resource_id=resource_id,
                manifest_root=manifest_root,
                manifest_max_files=max(100, int(req.manifest_max_files)),
                shallow_dirs=shallow,
                deep_dirs=deep,
                enable_wayback_lookup=bool(req.enable_wayback_lookup),
                wayback_delay=max(0.0, float(req.wayback_delay)),
                enable_web_search=bool(req.enable_web_search),
                persist=bool(req.persist),
                verbose=False,
            )
            reference_updates = int(recovery_mod.update_reference_urls_from_resources(paths.db_path))
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        updated = get_resource_repo().get_by_id(resource_id)
        if updated is None:
            raise HTTPException(status_code=404, detail=f"Resource not found after recovery: {resource_id}")
        return {
            "ok": True,
            "result": result,
            "reference_updates": reference_updates,
            "resource": _jsonable(updated),
        }

    @app.post("/api/sources/resources/{resource_id}/primary")
    def api_sources_resource_primary(resource_id: str, req: SourcePrimaryUrlRequest) -> dict[str, Any]:
        resource = get_resource_repo().get_by_id(resource_id)
        if resource is None:
            raise HTTPException(status_code=404, detail=f"Resource not found: {resource_id}")
        try:
            recovery_mod = get_source_recovery_module()
            raw = str(req.url or "").strip()
            normalized = raw
            if not recovery_mod.is_external_url(normalized):
                doi_url = recovery_mod.doi_to_url(raw)
                normalized = doi_url if doi_url else raw
            if not recovery_mod.is_external_url(normalized):
                raise ValueError("Provide an external URL or DOI.")
            merged_urls = recovery_mod.merge_urls(
                resource.download_urls_json,
                [resource.download_url or "", normalized],
            )
            primary = normalized
            get_resource_repo().update_download_metadata(
                resource_id=resource_id,
                download_url=primary,
                download_urls_json=json.dumps(merged_urls, ensure_ascii=True),
            )
            reference_updates = int(recovery_mod.update_reference_urls_from_resources(paths.db_path))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        updated = get_resource_repo().get_by_id(resource_id)
        if updated is None:
            raise HTTPException(status_code=404, detail=f"Resource not found after update: {resource_id}")
        return {"ok": True, "resource": _jsonable(updated), "reference_updates": reference_updates}

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
