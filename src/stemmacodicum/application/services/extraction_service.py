from __future__ import annotations

import json
import logging
import math
import multiprocessing as mp
import os
import queue
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from stemmacodicum.core.errors import ExtractionError
from stemmacodicum.core.hashing import compute_bytes_digest
from stemmacodicum.core.ids import new_uuid
from stemmacodicum.core.time import now_utc_iso
from stemmacodicum.domain.models.extraction import (
    AnnotationSpan,
    DocumentText,
    ExtractedTable,
    ExtractionRun,
    TextAnnotation,
    TextSegment,
)
from stemmacodicum.infrastructure.db.repos.extraction_repo import ExtractionRepo
from stemmacodicum.infrastructure.db.repos.resource_repo import ResourceRepo
from stemmacodicum.infrastructure.parsers.docling_adapter import (
    DoclingAdapter,
    DoclingRuntimeOptions,
    ParseResult,
    ParsedBlock,
    ParsedTable,
)
from stemmacodicum.application.services.vector_service import VectorIndexingService

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ExtractSummary:
    run_id: str
    resource_id: str
    tables_found: int
    parser_name: str | None = None
    parser_version: str | None = None
    elapsed_seconds: float | None = None
    page_count: int | None = None
    pages_per_second: float | None = None
    timings: dict[str, float] | None = None
    text_chars: int = 0
    text_words: int = 0
    text_sentences: int = 0
    text_paragraphs: int = 0
    segments_persisted: int = 0
    annotations_persisted: int = 0
    table_rows_total: int = 0
    table_cols_total: int = 0
    table_cells_total: int = 0
    vector_status: str | None = None
    vector_chunks_total: int = 0
    vector_chunks_indexed: int = 0
    vector_error: str | None = None


@dataclass(slots=True)
class _AnnotationSpec:
    layer: str
    category: str
    label: str | None
    spans: list[tuple[int, int]]
    confidence: float | None = None
    source: str | None = None
    attrs: dict[str, object] | None = None


class _PdfSubprocessCrashError(RuntimeError):
    pass


class ExtractionCancelledError(ExtractionError):
    pass


def _pdf_parse_worker_entry(
    *,
    file_path: str,
    media_type: str,
    parser_profile: str,
    runtime_options: DoclingRuntimeOptions | None,
    result_queue: Any,
) -> None:
    try:
        adapter = DoclingAdapter(profile=parser_profile, runtime_options=runtime_options)
        result = adapter.parse_resource(Path(file_path), media_type)
        result_queue.put({"ok": True, "result": result})
    except Exception as exc:
        result_queue.put({"ok": False, "error": f"{exc.__class__.__name__}: {exc}"})


class ExtractionService:
    _FINANCIAL_TERMS = [
        "cash",
        "cash flow",
        "revenue",
        "turnover",
        "profit",
        "ebitda",
        "asset",
        "liability",
        "equity",
        "capital",
        "debt",
        "expenditure",
        "income",
        "deficit",
        "surplus",
    ]

    _RE_CURRENCY = re.compile(r"[$£€]\s?\d[\d,]*(?:\.\d+)?")
    _RE_QUANTITY = re.compile(r"\b(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d+|\d{4,})\b")
    _RE_DATE = re.compile(
        r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}|(?:19|20)\d{2}(?:/\d{2,4})?)\b"
    )
    _RE_CITATION = re.compile(r"\[\d+\]|\([A-Z][A-Za-z\-]+,\s*(?:19|20)\d{2}\)")
    _RE_PERIOD = re.compile(r"\b(?:FY|Q[1-4]|H[12])\s?(?:19|20)?\d{2}(?:/\d{2,4})?\b")
    _RE_SENTENCE = re.compile(r"[^.!?\n]+(?:[.!?]+|$)")
    _REGEX_SCAN_CHUNK_CHARS = 16_384
    _REGEX_SCAN_OVERLAP_CHARS = 128
    _MAX_AUTO_ANNOTATIONS = 10_000
    _PDF_WORKER_JOIN_POLL_SECONDS = 0.2
    _DEFAULT_PDF_PARSE_MAX_RESTARTS = 2
    _DEFAULT_PDF_PARSE_ATTEMPT_TIMEOUT_SECONDS = 240
    _DEFAULT_NON_PDF_PARSE_MAX_RESTARTS = 1
    _DEFAULT_NON_PDF_PARSE_ATTEMPT_TIMEOUT_SECONDS = 180
    _DEFAULT_PDF_PAGE_SPLIT_MODE = True
    _DEFAULT_PDF_PAGE_SPLIT_MAX_WORKERS = 6

    def __init__(
        self,
        resource_repo: ResourceRepo,
        extraction_repo: ExtractionRepo,
        archive_dir: Path,
        docling_runtime_options: DoclingRuntimeOptions | None = None,
        vector_indexing_service: VectorIndexingService | None = None,
    ) -> None:
        self.resource_repo = resource_repo
        self.extraction_repo = extraction_repo
        self.archive_dir = archive_dir
        self.docling_runtime_options = docling_runtime_options
        self.vector_indexing_service = vector_indexing_service
        self.pdf_parse_subprocess_enabled = self._env_bool("STEMMA_PDF_PARSE_SUBPROCESS", default=True)
        self.pdf_parse_max_restarts = self._env_non_negative_int(
            "STEMMA_PDF_PARSE_MAX_RESTARTS",
            default=self._DEFAULT_PDF_PARSE_MAX_RESTARTS,
        )
        self.pdf_parse_attempt_timeout_seconds = self._env_non_negative_int(
            "STEMMA_PDF_PARSE_ATTEMPT_TIMEOUT_SECONDS",
            default=self._DEFAULT_PDF_PARSE_ATTEMPT_TIMEOUT_SECONDS,
        )
        self.non_pdf_parse_subprocess_enabled = self._env_bool(
            "STEMMA_NON_PDF_PARSE_SUBPROCESS",
            default=True,
        )
        self.non_pdf_parse_max_restarts = self._env_non_negative_int(
            "STEMMA_NON_PDF_PARSE_MAX_RESTARTS",
            default=self._DEFAULT_NON_PDF_PARSE_MAX_RESTARTS,
        )
        self.non_pdf_parse_attempt_timeout_seconds = self._env_non_negative_int(
            "STEMMA_NON_PDF_PARSE_ATTEMPT_TIMEOUT_SECONDS",
            default=self._DEFAULT_NON_PDF_PARSE_ATTEMPT_TIMEOUT_SECONDS,
        )
        self.pdf_page_split_mode = self._env_bool(
            "STEMMA_PDF_PAGE_SPLIT_MODE",
            default=self._DEFAULT_PDF_PAGE_SPLIT_MODE,
        )
        self.pdf_page_split_use_subprocess = self._env_bool(
            "STEMMA_PDF_PAGE_SPLIT_USE_SUBPROCESS",
            default=False,
        )
        self.pdf_page_split_workers = self._env_non_negative_int(
            "STEMMA_PDF_PAGE_SPLIT_WORKERS",
            default=0,
        )
        self.pdf_page_split_max_workers = self._env_non_negative_int(
            "STEMMA_PDF_PAGE_SPLIT_MAX_WORKERS",
            default=self._DEFAULT_PDF_PAGE_SPLIT_MAX_WORKERS,
        )
        self.pdf_page_split_threads_per_worker = self._env_non_negative_int(
            "STEMMA_PDF_PAGE_SPLIT_THREADS_PER_WORKER",
            default=0,
        )

    def extract_resource(
        self,
        resource_id: str,
        parser_profile: str = "default",
        progress_callback: Callable[[dict[str, object]], None] | None = None,
        cancellation_check: Callable[[], bool] | None = None,
    ) -> ExtractSummary:
        emit_lock = threading.Lock()

        def emit(payload: dict[str, object]) -> None:
            if progress_callback is None:
                return
            enriched = dict(payload)
            enriched.setdefault("emitted_at", now_utc_iso())
            with emit_lock:
                progress_callback(enriched)

        def ensure_not_cancelled(context: str = "extraction") -> None:
            if cancellation_check is not None and bool(cancellation_check()):
                raise ExtractionCancelledError(f"{context} cancelled by control request.")

        emit(
            {
                "stage": "extract",
                "state": "active",
                "progress": 4,
                "detail": "Starting text extraction.",
            }
        )
        ensure_not_cancelled("initialization")
        resource = self.resource_repo.get_by_id(resource_id)
        if resource is None:
            raise ExtractionError(f"Resource not found: {resource_id}")

        archived_path = self.archive_dir / resource.archived_relpath
        if not archived_path.exists():
            raise ExtractionError(f"Archived resource file missing: {archived_path}")

        adapter = DoclingAdapter(profile=parser_profile, runtime_options=self.docling_runtime_options)
        page_split_requested = resource.media_type == "application/pdf" and self.pdf_page_split_mode
        page_split_available = shutil.which("pdfseparate") is not None
        page_split_mode = page_split_requested and page_split_available
        if page_split_requested and not page_split_available:
            emit(
                {
                    "stage": "extract",
                    "state": "active",
                    "progress": 5,
                    "detail": "pdfseparate was not found; falling back to single-pass PDF parse.",
                    "component": "pdf_split",
                    "event": "split_unavailable",
                }
            )
        if page_split_mode:
            emit(
                {
                    "stage": "extract",
                    "state": "active",
                    "progress": 5,
                    "detail": "Per-page PDF parsing mode is enabled for this import.",
                    "component": "pdf_split",
                    "event": "split_mode_enabled",
                }
            )
        parse_started = time.perf_counter()
        parse_heartbeat_stop = threading.Event()
        parse_heartbeat_thread: threading.Thread | None = None
        if progress_callback is not None and not page_split_mode:
            def parse_heartbeat() -> None:
                while not parse_heartbeat_stop.wait(0.5):
                    if cancellation_check is not None and bool(cancellation_check()):
                        return
                    elapsed = max(0.0, time.perf_counter() - parse_started)
                    # Keep parse-stage heartbeat progress monotonic but slow-rising so long PDFs
                    # do not appear "stuck near completion" after ~20s.
                    parse_progress = 6 + int(49.0 * (1.0 - math.exp(-elapsed / 75.0)))
                    emit(
                        {
                            "stage": "extract",
                            "state": "active",
                            "progress": min(55, parse_progress),
                            "detail": "Extracting text and layout structure.",
                            "stats": f"Elapsed {elapsed:.1f}s • parser running (text/layout/table analysis)",
                        }
                    )

            parse_heartbeat_thread = threading.Thread(target=parse_heartbeat, daemon=True)
            parse_heartbeat_thread.start()
        try:
            ensure_not_cancelled("parse")
            if page_split_mode:
                parse_result = self._parse_pdf_split_pages(
                    archived_path=archived_path,
                    media_type=resource.media_type,
                    parser_profile=parser_profile,
                    emit=emit,
                    adapter=adapter,
                    cancellation_check=cancellation_check,
                )
            elif resource.media_type == "application/pdf" and self.pdf_parse_subprocess_enabled:
                parse_result = self._parse_pdf_with_restarts(
                    archived_path=archived_path,
                    media_type=resource.media_type,
                    parser_profile=parser_profile,
                    emit=emit,
                    cancellation_check=cancellation_check,
                )
            elif resource.media_type != "application/pdf" and self.non_pdf_parse_subprocess_enabled:
                parse_result = self._parse_non_pdf_with_restarts(
                    archived_path=archived_path,
                    media_type=resource.media_type,
                    parser_profile=parser_profile,
                    emit=emit,
                    cancellation_check=cancellation_check,
                )
            else:
                ensure_not_cancelled("parse")
                parse_result = adapter.parse_resource(archived_path, resource.media_type)
                ensure_not_cancelled("parse")
        except ExtractionCancelledError:
            raise
        except Exception as exc:
            raise ExtractionError(f"Extraction failed for {resource_id}: {exc}") from exc
        finally:
            parse_heartbeat_stop.set()
            if parse_heartbeat_thread is not None:
                parse_heartbeat_thread.join(timeout=0.2)

        ensure_not_cancelled("post-parse")
        full_text = parse_result.full_text or ""
        word_count = len(re.findall(r"\b\w+\b", full_text))
        sentence_count = len([m.group(0).strip() for m in self._RE_SENTENCE.finditer(full_text) if m.group(0).strip()])
        paragraph_count = len([p for p in re.split(r"\n\s*\n+", full_text) if p.strip()])
        table_rows_total = sum(len(t.row_headers) for t in parse_result.tables)
        table_cols_total = sum(len(t.col_headers) for t in parse_result.tables)
        table_cells_total = sum(len(t.cells) for t in parse_result.tables)
        emit(
            {
                "stage": "extract",
                "state": "active",
                "progress": 58,
                "detail": "Parsed document text and structure.",
                "page_count": parse_result.page_count,
                "stats": (
                    f"{parse_result.page_count or 0} pages • "
                    f"{word_count} words • "
                    f"{sentence_count} sentences • "
                    f"{paragraph_count} paragraphs"
                ),
            }
        )
        emit(
            {
                "stage": "tables",
                "state": "active",
                "progress": 66,
                "detail": "Consolidating table structures.",
                "stats": (
                    f"{len(parse_result.tables)} tables • "
                    f"{table_rows_total} rows • "
                    f"{table_cols_total} cols • "
                    f"{table_cells_total} cells"
                ),
            }
        )
        segment_specs = self._build_segment_specs(full_text, parse_result.blocks)
        annotation_specs = self._build_annotation_specs(
            full_text=full_text,
            parser_name=parse_result.parser_name,
        )

        output_payload = {
            "resource_id": resource_id,
            "text_content": full_text,
            "blocks": [self._block_payload(b) for b in parse_result.blocks],
            "segments": segment_specs,
            "annotations": [self._annotation_spec_payload(x) for x in annotation_specs],
            "tables": [self._table_payload(t) for t in parse_result.tables],
        }
        output_digest = compute_bytes_digest(json.dumps(output_payload, sort_keys=True).encode("utf-8"))

        run = ExtractionRun(
            id=new_uuid(),
            resource_id=resource_id,
            parser_name=parse_result.parser_name,
            parser_version=parse_result.parser_version,
            config_digest=parse_result.config_digest,
            output_digest=output_digest,
            status="success",
            created_at=now_utc_iso(),
        )
        self.extraction_repo.insert_run(run)

        total_tables = len(parse_result.tables)
        for table_index, parsed_table in enumerate(parse_result.tables, start=1):
            ensure_not_cancelled("table persistence")
            table_page_index = self._normalize_page_index(
                parsed_table.page_index,
                page_count=parse_result.page_count,
                clamp=False,
            )
            table_storage_page_index = self._normalize_page_index(
                parsed_table.page_index,
                page_count=parse_result.page_count,
                clamp=True,
            )
            table_id = self.derive_table_id(parsed_table, page_index=table_storage_page_index)
            table = ExtractedTable(
                id=new_uuid(),
                extraction_run_id=run.id,
                resource_id=resource_id,
                table_id=table_id,
                page_index=table_storage_page_index if table_storage_page_index is not None else 0,
                caption=parsed_table.caption,
                row_headers_json=json.dumps(parsed_table.row_headers, ensure_ascii=True),
                col_headers_json=json.dumps(parsed_table.col_headers, ensure_ascii=True),
                cells_json=json.dumps(
                    [
                        {
                            "row_index": c.row_index,
                            "col_index": c.col_index,
                            "value": c.value,
                        }
                        for c in parsed_table.cells
                    ],
                    ensure_ascii=True,
                ),
                bbox_json=json.dumps(parsed_table.bbox, ensure_ascii=True) if parsed_table.bbox else None,
                created_at=now_utc_iso(),
            )
            self.extraction_repo.insert_table(table)
            if total_tables and (
                table_index == 1
                or table_index == total_tables
                or table_index % max(1, total_tables // 4) == 0
            ):
                table_preview = self._preview_payload(
                    kind="table",
                    label=parsed_table.caption or f"Table {table_index}",
                    text=self._table_preview_text(parsed_table),
                    page_index=table_page_index,
                    page_count=parse_result.page_count,
                    bbox=parsed_table.bbox,
                )
                emit(
                    {
                        "stage": "tables",
                        "state": "active",
                        "progress": min(74, 66 + int((table_index / max(total_tables, 1)) * 8)),
                        "detail": "Persisting extracted tables.",
                        "stats": f"{table_index}/{total_tables} tables saved",
                        "page_count": parse_result.page_count,
                        "page_current": (table_page_index + 1) if table_page_index is not None else None,
                        "preview": table_preview,
                    }
                )

        document_text = DocumentText(
            id=new_uuid(),
            extraction_run_id=run.id,
            resource_id=resource_id,
            text_content=full_text,
            text_digest_sha256=compute_bytes_digest(full_text.encode("utf-8")),
            char_count=len(full_text),
            created_at=now_utc_iso(),
        )
        self.extraction_repo.insert_document_text(document_text)

        persisted_segments = 0
        total_segments = len(segment_specs)
        for segment_index, spec in enumerate(segment_specs, start=1):
            ensure_not_cancelled("segment persistence")
            segment_page_index = self._normalize_page_index(
                spec.get("page_index"),
                page_count=parse_result.page_count,
                clamp=False,
            )
            segment = TextSegment(
                id=new_uuid(),
                document_text_id=document_text.id,
                extraction_run_id=run.id,
                resource_id=resource_id,
                segment_type=str(spec["segment_type"]),
                start_offset=int(spec["start_offset"]),
                end_offset=int(spec["end_offset"]),
                page_index=segment_page_index,
                order_index=spec.get("order_index"),
                bbox_json=json.dumps(spec["bbox"], ensure_ascii=True)
                if isinstance(spec.get("bbox"), dict)
                else None,
                attrs_json=json.dumps(spec["attrs"], ensure_ascii=True, sort_keys=True)
                if isinstance(spec.get("attrs"), dict)
                else None,
                created_at=now_utc_iso(),
            )
            self.extraction_repo.insert_text_segment(segment)
            persisted_segments += 1
            if total_segments and (
                segment_index == 1
                or segment_index == total_segments
                or segment_index % 250 == 0
            ):
                segment_preview = self._preview_payload(
                    kind="segment",
                    label=str(spec.get("segment_type") or "segment"),
                    text=full_text[int(spec["start_offset"]) : int(spec["end_offset"])],
                    page_index=segment_page_index,
                    page_count=parse_result.page_count,
                    bbox=spec.get("bbox") if isinstance(spec.get("bbox"), dict) else None,
                )
                emit(
                    {
                        "stage": "extract",
                        "state": "active",
                        "progress": min(84, 74 + int((segment_index / max(total_segments, 1)) * 10)),
                        "detail": "Persisting text segments.",
                        "stats": f"{segment_index}/{total_segments} segments",
                        "page_count": parse_result.page_count,
                        "page_current": (segment_page_index + 1) if segment_page_index is not None else None,
                        "preview": segment_preview,
                    }
                )

        persisted_annotations = 0
        total_annotations = len(annotation_specs)
        for annotation_index, spec in enumerate(annotation_specs, start=1):
            ensure_not_cancelled("annotation persistence")
            annotation = TextAnnotation(
                id=new_uuid(),
                document_text_id=document_text.id,
                extraction_run_id=run.id,
                resource_id=resource_id,
                layer=spec.layer,
                category=spec.category,
                label=spec.label,
                confidence=spec.confidence,
                source=spec.source,
                attrs_json=json.dumps(spec.attrs, ensure_ascii=True, sort_keys=True) if spec.attrs else None,
                created_at=now_utc_iso(),
            )
            self.extraction_repo.insert_text_annotation(annotation)
            for span_order, (start, end) in enumerate(spec.spans):
                self.extraction_repo.insert_annotation_span(
                    AnnotationSpan(
                        id=new_uuid(),
                        annotation_id=annotation.id,
                        start_offset=start,
                        end_offset=end,
                        span_order=span_order,
                        created_at=now_utc_iso(),
                    )
                )
            persisted_annotations += 1
            if total_annotations and (
                annotation_index == 1
                or annotation_index == total_annotations
                or annotation_index % 250 == 0
            ):
                annotation_text = None
                if spec.spans:
                    first_start, first_end = spec.spans[0]
                    annotation_text = full_text[first_start:first_end]
                annotation_preview = self._preview_payload(
                    kind="annotation",
                    label=f"{spec.layer}:{spec.category}",
                    text=annotation_text,
                    page_index=None,
                    page_count=parse_result.page_count,
                    bbox=None,
                )
                emit(
                    {
                        "stage": "tables",
                        "state": "active",
                        "progress": min(96, 84 + int((annotation_index / max(total_annotations, 1)) * 12)),
                        "detail": "Persisting annotations.",
                        "stats": f"{annotation_index}/{total_annotations} annotations",
                        "page_count": parse_result.page_count,
                        "preview": annotation_preview,
                    }
                )

        emit(
            {
                "stage": "extract",
                "state": "done",
                "progress": 100,
                "detail": "Text extraction complete.",
                "page_count": parse_result.page_count,
                "stats": (
                    f"{word_count} words • "
                    f"{sentence_count} sentences • "
                    f"{paragraph_count} paragraphs"
                ),
            }
        )
        emit(
            {
                "stage": "tables",
                "state": "done",
                "progress": 100,
                "detail": "Table and annotation extraction complete.",
                "stats": (
                    f"{len(parse_result.tables)} tables • "
                    f"{persisted_annotations} annotations"
                ),
            }
        )

        vector_status: str | None = None
        vector_chunks_total = 0
        vector_chunks_indexed = 0
        vector_error: str | None = None
        if self.vector_indexing_service is not None:
            vector_summary = self.vector_indexing_service.index_extraction(
                resource_id=resource_id,
                extraction_run_id=run.id,
                progress_callback=progress_callback,
            )
            vector_status = vector_summary.status
            vector_chunks_total = vector_summary.chunks_total
            vector_chunks_indexed = vector_summary.chunks_indexed
            vector_error = vector_summary.error

        return ExtractSummary(
            run_id=run.id,
            resource_id=resource_id,
            tables_found=len(parse_result.tables),
            parser_name=parse_result.parser_name,
            parser_version=parse_result.parser_version,
            elapsed_seconds=parse_result.elapsed_seconds,
            page_count=parse_result.page_count,
            pages_per_second=(
                (parse_result.page_count / parse_result.elapsed_seconds)
                if parse_result.page_count
                and parse_result.elapsed_seconds is not None
                and parse_result.elapsed_seconds > 0
                else None
            ),
            timings=parse_result.timings,
            text_chars=len(full_text),
            text_words=word_count,
            text_sentences=sentence_count,
            text_paragraphs=paragraph_count,
            segments_persisted=persisted_segments,
            annotations_persisted=persisted_annotations,
            table_rows_total=table_rows_total,
            table_cols_total=table_cols_total,
            table_cells_total=table_cells_total,
            vector_status=vector_status,
            vector_chunks_total=vector_chunks_total,
            vector_chunks_indexed=vector_chunks_indexed,
            vector_error=vector_error,
        )

    def _parse_pdf_with_restarts(
        self,
        *,
        archived_path: Path,
        media_type: str,
        parser_profile: str,
        emit: Callable[[dict[str, object]], None],
        runtime_options_override: DoclingRuntimeOptions | None = None,
        page_current: int | None = None,
        page_count: int | None = None,
        cancellation_check: Callable[[], bool] | None = None,
    ) -> ParseResult:
        attempts_total = max(1, self.pdf_parse_max_restarts + 1)
        for attempt in range(1, attempts_total + 1):
            attempt_started = time.perf_counter()
            start_payload: dict[str, object] = {
                "stage": "extract",
                "state": "active",
                "progress": 6,
                "detail": (
                    f"Starting Docling parse attempt {attempt}/{attempts_total}."
                ),
                "stats": f"parser subprocess • {archived_path.name}",
                "component": "docling",
                "event": "attempt_start",
                "attempt": attempt,
                "attempts_total": attempts_total,
            }
            if page_current is not None:
                start_payload["page_current"] = page_current
            if page_count is not None:
                start_payload["page_count"] = page_count
            emit(start_payload)
            try:
                result = self._run_pdf_parse_subprocess_attempt(
                    archived_path=archived_path,
                    media_type=media_type,
                    parser_profile=parser_profile,
                    runtime_options_override=runtime_options_override,
                    timeout_seconds=self.pdf_parse_attempt_timeout_seconds,
                    cancellation_check=cancellation_check,
                )
                attempt_elapsed_ms = int(max(0.0, (time.perf_counter() - attempt_started) * 1000.0))
                done_payload: dict[str, object] = {
                    "stage": "extract",
                    "state": "active",
                    "progress": 55,
                    "detail": (
                        f"Docling parse attempt {attempt}/{attempts_total} completed."
                    ),
                    "stats": f"{attempt_elapsed_ms}ms • {result.page_count or 0} pages",
                    "page_count": result.page_count,
                    "component": "docling",
                    "event": "attempt_done",
                    "attempt": attempt,
                    "attempts_total": attempts_total,
                    "duration_ms": attempt_elapsed_ms,
                }
                if page_current is not None:
                    done_payload["page_current"] = page_current
                if page_count is not None:
                    done_payload["page_count"] = page_count
                emit(done_payload)
                return result
            except ExtractionCancelledError:
                raise
            except _PdfSubprocessCrashError as exc:
                if attempt >= attempts_total:
                    break
                logger.warning(
                    "PDF parser subprocess crashed for %s (attempt %s/%s): %s",
                    archived_path,
                    attempt,
                    attempts_total,
                    exc,
                )
                retry_payload: dict[str, object] = {
                    "stage": "extract",
                    "state": "active",
                    "progress": 6,
                    "detail": (
                        f"Parser subprocess crashed (attempt {attempt}/{attempts_total}); restarting."
                    ),
                    "stats": str(exc),
                    "component": "docling",
                    "event": "attempt_restart",
                    "attempt": attempt,
                    "attempts_total": attempts_total,
                }
                if page_current is not None:
                    retry_payload["page_current"] = page_current
                if page_count is not None:
                    retry_payload["page_count"] = page_count
                emit(retry_payload)
                time.sleep(min(0.2 * attempt, 1.0))
        raise ExtractionError(
            "PDF parser subprocess crashed repeatedly "
            f"({attempts_total}/{attempts_total} attempts) for {archived_path.name}."
        )

    def _parse_non_pdf_with_restarts(
        self,
        *,
        archived_path: Path,
        media_type: str,
        parser_profile: str,
        emit: Callable[[dict[str, object]], None],
        cancellation_check: Callable[[], bool] | None = None,
    ) -> ParseResult:
        attempts_total = max(1, self.non_pdf_parse_max_restarts + 1)
        for attempt in range(1, attempts_total + 1):
            attempt_started = time.perf_counter()
            emit(
                {
                    "stage": "extract",
                    "state": "active",
                    "progress": 6,
                    "detail": f"Starting parse attempt {attempt}/{attempts_total}.",
                    "stats": f"parser subprocess • {archived_path.name}",
                    "component": "docling",
                    "event": "attempt_start",
                    "attempt": attempt,
                    "attempts_total": attempts_total,
                }
            )
            try:
                result = self._run_pdf_parse_subprocess_attempt(
                    archived_path=archived_path,
                    media_type=media_type,
                    parser_profile=parser_profile,
                    timeout_seconds=self.non_pdf_parse_attempt_timeout_seconds,
                    cancellation_check=cancellation_check,
                )
                attempt_elapsed_ms = int(max(0.0, (time.perf_counter() - attempt_started) * 1000.0))
                emit(
                    {
                        "stage": "extract",
                        "state": "active",
                        "progress": 55,
                        "detail": f"Parse attempt {attempt}/{attempts_total} completed.",
                        "stats": f"{attempt_elapsed_ms}ms • {result.page_count or 0} pages",
                        "page_count": result.page_count,
                        "component": "docling",
                        "event": "attempt_done",
                        "attempt": attempt,
                        "attempts_total": attempts_total,
                        "duration_ms": attempt_elapsed_ms,
                    }
                )
                return result
            except ExtractionCancelledError:
                raise
            except _PdfSubprocessCrashError as exc:
                if attempt >= attempts_total:
                    break
                logger.warning(
                    "Parser subprocess crashed for %s (attempt %s/%s): %s",
                    archived_path,
                    attempt,
                    attempts_total,
                    exc,
                )
                emit(
                    {
                        "stage": "extract",
                        "state": "active",
                        "progress": 6,
                        "detail": (
                            f"Parser subprocess crashed (attempt {attempt}/{attempts_total}); restarting."
                        ),
                        "stats": str(exc),
                        "component": "docling",
                        "event": "attempt_restart",
                        "attempt": attempt,
                        "attempts_total": attempts_total,
                    }
                )
                time.sleep(min(0.2 * attempt, 1.0))
        raise ExtractionError(
            "Parser subprocess crashed repeatedly "
            f"({attempts_total}/{attempts_total} attempts) for {archived_path.name}."
        )

    def _parse_pdf_split_pages(
        self,
        *,
        archived_path: Path,
        media_type: str,
        parser_profile: str,
        emit: Callable[[dict[str, object]], None],
        adapter: DoclingAdapter,
        cancellation_check: Callable[[], bool] | None = None,
    ) -> ParseResult:
        if cancellation_check is not None and bool(cancellation_check()):
            raise ExtractionCancelledError("PDF split parse cancelled by control request.")
        split_started = time.perf_counter()
        emit(
            {
                "stage": "extract",
                "state": "active",
                "progress": 5,
                "detail": "Splitting PDF into one-page files for isolated parsing.",
                "stats": f"Source {archived_path.name}",
                "component": "pdf_split",
                "event": "split_start",
            }
        )
        with tempfile.TemporaryDirectory(prefix="stemma-pdf-pages-") as tmp_dir_raw:
            tmp_dir = Path(tmp_dir_raw)
            output_pattern = str(tmp_dir / "page-%06d.pdf")
            split_cmd = ["pdfseparate", str(archived_path), output_pattern]
            split_proc = subprocess.run(
                split_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if split_proc.returncode != 0:
                stderr = (split_proc.stderr or "").strip()
                raise ExtractionError(
                    "Failed to split PDF into one-page files "
                    f"(exit={split_proc.returncode}): {stderr[:320]}"
                )
            page_files = sorted(tmp_dir.glob("page-*.pdf"))
            total_pages = len(page_files)
            if total_pages <= 0:
                raise ExtractionError(f"PDF split produced no pages for {archived_path.name}")
            worker_count = self._resolve_pdf_page_split_workers(total_pages)
            use_subprocess = (
                self.pdf_page_split_use_subprocess
                or worker_count > 1
                or self.pdf_parse_subprocess_enabled
            )
            runtime_options_override = (
                self._scaled_docling_runtime_options_for_parallel(worker_count)
                if use_subprocess
                else None
            )
            split_elapsed_ms = int(max(0.0, (time.perf_counter() - split_started) * 1000.0))
            emit(
                {
                    "stage": "extract",
                    "state": "active",
                    "progress": 6,
                    "detail": "PDF split complete; parsing pages individually.",
                    "stats": (
                        f"{total_pages} page files ready • split {split_elapsed_ms}ms • "
                        f"workers {worker_count} • subprocess {str(use_subprocess).lower()}"
                    ),
                    "page_count": total_pages,
                    "component": "pdf_split",
                    "event": "split_done",
                    "duration_ms": split_elapsed_ms,
                    "workers": worker_count,
                    "subprocess": use_subprocess,
                    "runtime_threads_per_worker": (
                        runtime_options_override.num_threads
                        if isinstance(runtime_options_override, DoclingRuntimeOptions)
                        else None
                    ),
                }
            )

            parse_started = time.perf_counter()
            page_results: dict[int, tuple[ParseResult, int]] = {}
            merged_text_parts: list[str] = []
            merged_blocks: list[ParsedBlock] = []
            merged_tables: list[ParsedTable] = []
            merged_timings: dict[str, float] = {}
            config_digests: list[str] = []
            parser_names: list[str] = []
            parser_versions: list[str] = []
            merged_text_len = 0
            completed_pages = 0

            if worker_count > 1:
                executor = ThreadPoolExecutor(max_workers=worker_count)
                active_pages: dict[int, float] = {}
                active_lock = threading.Lock()
                wait_monitor_stop = threading.Event()
                future_map: dict[object, int] = {}
                force_abort = False
                stall_timeout_seconds = self._pdf_page_split_worker_stall_timeout_seconds()

                def wait_monitor() -> None:
                    while not wait_monitor_stop.wait(2.0):
                        with active_lock:
                            waiting = sorted(active_pages.items())
                        if not waiting:
                            continue
                        now_ts = time.perf_counter()
                        oldest_page = waiting[0][0]
                        oldest_age = max(0.0, now_ts - waiting[0][1])
                        waiting_pages = [idx for idx, _started in waiting[:8]]
                        emit(
                            {
                                "stage": "extract",
                                "state": "active",
                                "progress": min(
                                    55,
                                    6 + int((49.0 * completed_pages) / max(total_pages, 1)),
                                ),
                                "detail": (
                                    f"Waiting on {len(waiting)} page worker(s); "
                                    f"{completed_pages}/{total_pages} complete."
                                ),
                                "stats": (
                                    f"oldest page {oldest_page} waiting {oldest_age:.1f}s • "
                                    f"active {waiting_pages}"
                                ),
                                "page_count": total_pages,
                                "page_current": oldest_page,
                                "component": "scheduler",
                                "event": "page_wait",
                            }
                        )

                monitor_thread = threading.Thread(target=wait_monitor, daemon=True)
                monitor_thread.start()
                try:
                    future_map = {
                        executor.submit(
                            self._parse_pdf_split_single_page,
                            page_idx=page_idx,
                            total_pages=total_pages,
                            page_path=page_path,
                            media_type=media_type,
                            parser_profile=parser_profile,
                            emit=emit,
                            adapter=adapter,
                            use_subprocess=use_subprocess,
                            runtime_options_override=runtime_options_override,
                            cancellation_check=cancellation_check,
                        ): page_idx
                        for page_idx, page_path in enumerate(page_files, start=1)
                    }
                    with active_lock:
                        now_ts = time.perf_counter()
                        for page_idx in future_map.values():
                            active_pages[page_idx] = now_ts
                    pending = set(future_map.keys())
                    while pending:
                        if cancellation_check is not None and bool(cancellation_check()):
                            raise ExtractionCancelledError("PDF split parse cancelled by control request.")
                        done, pending = wait(
                            pending,
                            timeout=self._PDF_WORKER_JOIN_POLL_SECONDS,
                            return_when=FIRST_COMPLETED,
                        )
                        if not done:
                            with active_lock:
                                waiting = sorted(active_pages.items(), key=lambda item: item[1])
                            if waiting:
                                now_ts = time.perf_counter()
                                oldest_page = waiting[0][0]
                                oldest_age = max(0.0, now_ts - waiting[0][1])
                                if oldest_age >= float(stall_timeout_seconds):
                                    waiting_pages = ", ".join(str(idx) for idx, _ in waiting[:8])
                                    raise ExtractionError(
                                        "Per-page parser appears stalled: "
                                        f"page {oldest_page}/{total_pages} has been running for "
                                        f"{oldest_age:.1f}s (limit {stall_timeout_seconds}s). "
                                        f"Active pages: {waiting_pages}"
                                    )
                            continue
                        for future in done:
                            page_idx = future_map.get(future)
                            if page_idx is None:
                                continue
                            try:
                                parsed_page_idx, page_result, page_elapsed_ms = future.result()
                            except ExtractionCancelledError:
                                raise
                            except Exception as exc:
                                raise ExtractionError(
                                    f"Per-page parse failed for page {page_idx}/{total_pages}: {exc}"
                                ) from exc
                            with active_lock:
                                active_pages.pop(parsed_page_idx, None)
                            page_results[parsed_page_idx] = (page_result, page_elapsed_ms)
                            completed_pages += 1
                            page_text = page_result.full_text or ""
                            emit(
                                {
                                    "stage": "extract",
                                    "state": "active",
                                    "progress": min(
                                        55,
                                        6 + int((49.0 * completed_pages) / max(total_pages, 1)),
                                    ),
                                    "detail": (
                                        f"Completed {completed_pages}/{total_pages} pages "
                                        f"(latest page {parsed_page_idx}/{total_pages})."
                                    ),
                                    "stats": (
                                        f"{page_elapsed_ms}ms • "
                                        f"{len(page_result.tables)} tables • "
                                        f"{len(page_text)} chars"
                                    ),
                                    "page_count": total_pages,
                                    "page_current": parsed_page_idx,
                                    "component": "docling",
                                    "event": "page_parse_done",
                                    "duration_ms": page_elapsed_ms,
                                }
                            )
                except Exception:
                    force_abort = True
                    raise
                finally:
                    wait_monitor_stop.set()
                    monitor_thread.join(timeout=0.2)
                    if force_abort:
                        for future in future_map:
                            future.cancel()
                        executor.shutdown(wait=False, cancel_futures=True)
                    else:
                        executor.shutdown(wait=True, cancel_futures=False)
            else:
                for page_idx, page_path in enumerate(page_files, start=1):
                    try:
                        if cancellation_check is not None and bool(cancellation_check()):
                            raise ExtractionCancelledError("PDF split parse cancelled by control request.")
                        parsed_page_idx, page_result, page_elapsed_ms = self._parse_pdf_split_single_page(
                            page_idx=page_idx,
                            total_pages=total_pages,
                            page_path=page_path,
                            media_type=media_type,
                            parser_profile=parser_profile,
                            emit=emit,
                            adapter=adapter,
                            use_subprocess=use_subprocess,
                            runtime_options_override=runtime_options_override,
                            cancellation_check=cancellation_check,
                        )
                    except ExtractionCancelledError:
                        raise
                    except Exception as exc:
                        raise ExtractionError(
                            f"Per-page parse failed for page {page_idx}/{total_pages}: {exc}"
                        ) from exc
                    page_results[parsed_page_idx] = (page_result, page_elapsed_ms)
                    completed_pages += 1
                    page_text = page_result.full_text or ""
                    emit(
                        {
                            "stage": "extract",
                            "state": "active",
                            "progress": min(
                                55,
                                6 + int((49.0 * completed_pages) / max(total_pages, 1)),
                            ),
                            "detail": (
                                f"Completed {completed_pages}/{total_pages} pages "
                                f"(latest page {parsed_page_idx}/{total_pages})."
                            ),
                            "stats": (
                                f"{page_elapsed_ms}ms • "
                                f"{len(page_result.tables)} tables • "
                                f"{len(page_text)} chars"
                            ),
                            "page_count": total_pages,
                            "page_current": parsed_page_idx,
                            "component": "docling",
                            "event": "page_parse_done",
                            "duration_ms": page_elapsed_ms,
                        }
                    )

            for page_idx in range(1, total_pages + 1):
                if page_idx not in page_results:
                    raise ExtractionError(
                        f"Missing per-page parse result while merging output (page {page_idx}/{total_pages})."
                    )
                page_result, _page_elapsed_ms = page_results[page_idx]
                config_digests.append(str(page_result.config_digest))
                parser_names.append(str(page_result.parser_name or "docling"))
                parser_versions.append(str(page_result.parser_version or "auto"))

                if merged_text_parts:
                    merged_text_parts.append("\n\n")
                    merged_text_len += 2
                page_text = page_result.full_text or ""
                page_offset = merged_text_len
                merged_text_parts.append(page_text)
                merged_text_len += len(page_text)

                for block in page_result.blocks:
                    start = max(0, int(block.start_offset))
                    end = max(start, int(block.end_offset))
                    if end > len(page_text):
                        end = len(page_text)
                    merged_blocks.append(
                        ParsedBlock(
                            block_type=block.block_type,
                            start_offset=page_offset + start,
                            end_offset=page_offset + end,
                            page_index=page_idx - 1,
                            bbox=block.bbox,
                            attrs=block.attrs,
                        )
                    )

                for table in page_result.tables:
                    merged_tables.append(
                        ParsedTable(
                            page_index=page_idx - 1,
                            caption=table.caption,
                            row_headers=list(table.row_headers),
                            col_headers=list(table.col_headers),
                            cells=list(table.cells),
                            bbox=table.bbox,
                        )
                    )

                if isinstance(page_result.timings, dict):
                    for key, value in page_result.timings.items():
                        try:
                            parsed_value = float(value)
                        except Exception:
                            continue
                        if parsed_value < 0:
                            continue
                        merged_timings[str(key)] = merged_timings.get(str(key), 0.0) + parsed_value

            parse_elapsed_seconds = max(0.0, time.perf_counter() - parse_started)
            page_split_timings = dict(merged_timings)
            page_split_timings["page_parse_total_seconds"] = parse_elapsed_seconds
            page_split_timings["page_split_seconds"] = max(
                0.0,
                (split_elapsed_ms / 1000.0),
            )

            parser_name = parser_names[0] if parser_names else "docling"
            parser_version = parser_versions[0] if parser_versions else "auto"
            parser_config = {
                "profile": parser_profile,
                "mode": "pdf_page_split",
                "pages": total_pages,
                "source_name": archived_path.name,
                "workers": worker_count,
                "subprocess": use_subprocess,
                "runtime_threads_per_worker": (
                    runtime_options_override.num_threads
                    if isinstance(runtime_options_override, DoclingRuntimeOptions)
                    else None
                ),
                "page_config_digests": config_digests,
            }
            merged_text = "".join(merged_text_parts)
            emit(
                {
                    "stage": "extract",
                    "state": "active",
                    "progress": 56,
                    "detail": "Merged per-page parse output into a single document result.",
                    "stats": (
                        f"{total_pages} pages • "
                        f"{len(merged_tables)} tables • "
                        f"{len(merged_text)} chars"
                    ),
                    "page_count": total_pages,
                    "page_current": total_pages,
                    "component": "pdf_split",
                    "event": "merge_done",
                }
            )
            return ParseResult(
                parser_name=parser_name,
                parser_version=parser_version,
                config_digest=compute_bytes_digest(
                    json.dumps(parser_config, sort_keys=True).encode("utf-8")
                ),
                tables=merged_tables,
                full_text=merged_text,
                blocks=merged_blocks,
                elapsed_seconds=parse_elapsed_seconds,
                page_count=total_pages,
                timings=page_split_timings,
            )

    def _parse_pdf_split_single_page(
        self,
        *,
        page_idx: int,
        total_pages: int,
        page_path: Path,
        media_type: str,
        parser_profile: str,
        emit: Callable[[dict[str, object]], None],
        adapter: DoclingAdapter,
        use_subprocess: bool,
        runtime_options_override: DoclingRuntimeOptions | None,
        cancellation_check: Callable[[], bool] | None = None,
    ) -> tuple[int, ParseResult, int]:
        if cancellation_check is not None and bool(cancellation_check()):
            raise ExtractionCancelledError(f"Page parse cancelled before page {page_idx}/{total_pages}.")
        page_progress = 6 + int((49.0 * max(0, page_idx - 1)) / max(total_pages, 1))
        emit(
            {
                "stage": "extract",
                "state": "active",
                "progress": min(55, page_progress),
                "detail": f"Parsing page {page_idx}/{total_pages} with Docling.",
                "stats": (
                    f"Page file {page_path.name} • "
                    f"subprocess {str(use_subprocess).lower()}"
                ),
                "page_count": total_pages,
                "page_current": page_idx,
                "component": "docling",
                "event": "page_parse_start",
            }
        )
        page_started = time.perf_counter()
        if use_subprocess:
            page_result = self._parse_pdf_with_restarts(
                archived_path=page_path,
                media_type=media_type,
                parser_profile=parser_profile,
                emit=emit,
                runtime_options_override=runtime_options_override,
                page_current=page_idx,
                page_count=total_pages,
                cancellation_check=cancellation_check,
            )
        else:
            page_result = adapter.parse_resource(page_path, media_type)
            if cancellation_check is not None and bool(cancellation_check()):
                raise ExtractionCancelledError(f"Page parse cancelled after page {page_idx}/{total_pages}.")
        page_elapsed_ms = int(max(0.0, (time.perf_counter() - page_started) * 1000.0))
        return page_idx, page_result, page_elapsed_ms

    def _resolve_pdf_page_split_workers(self, total_pages: int) -> int:
        if total_pages <= 1:
            return 1
        cap_raw = self.pdf_page_split_max_workers
        cap = max(
            1,
            cap_raw if cap_raw > 0 else self._DEFAULT_PDF_PAGE_SPLIT_MAX_WORKERS,
        )
        requested = max(0, self.pdf_page_split_workers)
        if requested > 0:
            return max(1, min(requested, cap, total_pages))
        cpu_count = max(1, os.cpu_count() or 1)
        auto_workers = 1
        if total_pages >= 4:
            auto_workers = max(2, cpu_count // 4)
        if total_pages >= 16:
            auto_workers = max(auto_workers, cpu_count // 3)
        auto_workers = max(1, auto_workers)
        return min(auto_workers, cap, total_pages)

    def _scaled_docling_runtime_options_for_parallel(self, worker_count: int) -> DoclingRuntimeOptions | None:
        if worker_count <= 1:
            return self.docling_runtime_options
        base = self.docling_runtime_options or DoclingRuntimeOptions()
        cpu_count = max(1, os.cpu_count() or 1)
        explicit_threads = max(0, self.pdf_page_split_threads_per_worker)
        target_threads = max(1, explicit_threads) if explicit_threads > 0 else max(
            1,
            min(8, cpu_count // worker_count),
        )
        if base.num_threads is not None and base.num_threads > 0:
            target_threads = max(1, min(int(base.num_threads), target_threads))
        return DoclingRuntimeOptions(
            auto_tune=base.auto_tune,
            use_threaded_pipeline=base.use_threaded_pipeline,
            device=base.device,
            num_threads=target_threads,
            layout_batch_size=base.layout_batch_size,
            ocr_batch_size=base.ocr_batch_size,
            table_batch_size=base.table_batch_size,
            queue_max_size=base.queue_max_size,
            log_settings=base.log_settings,
        )

    def _pdf_page_split_worker_stall_timeout_seconds(self) -> int:
        attempt_timeout = max(0, int(self.pdf_parse_attempt_timeout_seconds))
        attempts_total = max(1, self.pdf_parse_max_restarts + 1)
        if attempt_timeout <= 0:
            return 900
        # Allow for parser timeout, restart backoff, and subprocess shutdown overhead.
        return max(90, int((attempt_timeout + 8) * attempts_total + 15))

    def _run_pdf_parse_subprocess_attempt(
        self,
        *,
        archived_path: Path,
        media_type: str,
        parser_profile: str,
        runtime_options_override: DoclingRuntimeOptions | None = None,
        timeout_seconds: int | None = None,
        cancellation_check: Callable[[], bool] | None = None,
    ) -> ParseResult:
        mp_ctx = mp.get_context("spawn")
        result_queue = mp_ctx.Queue(maxsize=1)
        process = mp_ctx.Process(
            target=_pdf_parse_worker_entry,
            kwargs={
                "file_path": str(archived_path),
                "media_type": media_type,
                "parser_profile": parser_profile,
                "runtime_options": (
                    runtime_options_override
                    if runtime_options_override is not None
                    else self.docling_runtime_options
                ),
                "result_queue": result_queue,
            },
        )
        process.start()
        attempt_timeout_seconds = max(
            0,
            int(self.pdf_parse_attempt_timeout_seconds if timeout_seconds is None else timeout_seconds),
        )
        wait_started = time.perf_counter()
        payload: dict[str, object] | None = None
        try:
            deadline = (
                wait_started + float(attempt_timeout_seconds)
                if attempt_timeout_seconds > 0
                else None
            )
            while True:
                now = time.perf_counter()
                if cancellation_check is not None and bool(cancellation_check()):
                    process.terminate()
                    process.join(timeout=5)
                    raise ExtractionCancelledError(
                        f"Parser worker cancelled while processing {archived_path.name}."
                    )
                if deadline is not None and now >= deadline:
                    process.terminate()
                    process.join(timeout=5)
                    raise _PdfSubprocessCrashError(
                        f"parser worker timed out after {attempt_timeout_seconds}s"
                    )
                wait_timeout = self._PDF_WORKER_JOIN_POLL_SECONDS
                if deadline is not None:
                    wait_timeout = max(0.01, min(wait_timeout, deadline - now))
                try:
                    queued_payload = result_queue.get(timeout=wait_timeout)
                    payload = queued_payload if isinstance(queued_payload, dict) else None
                    if payload is None:
                        payload = {"ok": False, "error": "invalid parser worker payload"}
                except queue.Empty:
                    if process.is_alive():
                        continue
                    break
                if process.is_alive():
                    process.join(timeout=self._PDF_WORKER_JOIN_POLL_SECONDS)
                if not process.is_alive():
                    break
        except BaseException:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
            raise
        if process.is_alive():
            process.join(timeout=1.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
        try:
            if payload is None:
                payload = result_queue.get_nowait()
        except queue.Empty:
            pass
        finally:
            try:
                result_queue.cancel_join_thread()
            except Exception:
                pass
            try:
                result_queue.close()
            except Exception:
                pass

        if process.exitcode == 0:
            if isinstance(payload, dict) and payload.get("ok") is True:
                result = payload.get("result")
                if isinstance(result, ParseResult):
                    return result
                raise ExtractionError("Parser subprocess returned an invalid result payload.")
            error = "unknown parser worker error"
            if isinstance(payload, dict) and isinstance(payload.get("error"), str):
                error = str(payload.get("error"))
            raise ExtractionError(error)

        raise _PdfSubprocessCrashError(self._format_worker_exit(process.exitcode))

    def list_tables(self, resource_id: str, limit: int = 100) -> list[ExtractedTable]:
        return self.extraction_repo.list_tables_for_resource(resource_id=resource_id, limit=limit)

    def get_document_text(
        self,
        resource_id: str,
        extraction_run_id: str | None = None,
    ) -> DocumentText | None:
        if extraction_run_id:
            doc_text = self.extraction_repo.get_document_text_for_run(extraction_run_id)
            if doc_text is None:
                return None
            if doc_text.resource_id != resource_id:
                raise ExtractionError(
                    f"Extraction run {extraction_run_id} does not belong to resource {resource_id}"
                )
            return doc_text
        return self.extraction_repo.get_latest_document_text_for_resource(resource_id)

    def list_segments(
        self,
        resource_id: str,
        *,
        extraction_run_id: str | None = None,
        segment_type: str | None = None,
        limit: int = 1000,
    ) -> list[TextSegment]:
        return self.extraction_repo.list_segments_for_resource(
            resource_id=resource_id,
            extraction_run_id=extraction_run_id,
            segment_type=segment_type,
            limit=limit,
        )

    def list_annotations(
        self,
        resource_id: str,
        *,
        extraction_run_id: str | None = None,
        layer: str | None = None,
        category: str | None = None,
        limit: int = 1000,
    ) -> list[dict[str, object]]:
        rows = self.extraction_repo.list_annotations_for_resource(
            resource_id=resource_id,
            extraction_run_id=extraction_run_id,
            layer=layer,
            category=category,
            limit=limit,
        )
        output: list[dict[str, object]] = []
        for annotation, spans in rows:
            output.append(
                {
                    "id": annotation.id,
                    "layer": annotation.layer,
                    "category": annotation.category,
                    "label": annotation.label,
                    "confidence": annotation.confidence,
                    "source": annotation.source,
                    "attrs_json": annotation.attrs_json,
                    "spans": [
                        {
                            "start": span.start_offset,
                            "end": span.end_offset,
                            "span_order": span.span_order,
                        }
                        for span in spans
                    ],
                    "created_at": annotation.created_at,
                }
            )
        return output

    def build_dump(
        self,
        resource_id: str,
        *,
        extraction_run_id: str | None = None,
        segment_limit: int = 5000,
        annotation_limit: int = 5000,
        table_limit: int = 1000,
    ) -> dict[str, object]:
        run = self._resolve_run(resource_id=resource_id, extraction_run_id=extraction_run_id)
        if run is None:
            raise ExtractionError(f"No extraction runs found for resource: {resource_id}")

        document_text = self.extraction_repo.get_document_text_for_run(run.id)
        tables = self.extraction_repo.list_tables_for_run(run.id, limit=table_limit)
        segments = self.list_segments(
            resource_id=resource_id,
            extraction_run_id=run.id,
            limit=segment_limit,
        )
        annotations = self.list_annotations(
            resource_id=resource_id,
            extraction_run_id=run.id,
            limit=annotation_limit,
        )

        return {
            "run": {
                "id": run.id,
                "resource_id": run.resource_id,
                "parser_name": run.parser_name,
                "parser_version": run.parser_version,
                "config_digest": run.config_digest,
                "output_digest": run.output_digest,
                "status": run.status,
                "created_at": run.created_at,
            },
            "document_text": {
                "id": document_text.id,
                "text_digest_sha256": document_text.text_digest_sha256,
                "char_count": document_text.char_count,
                "text_content": document_text.text_content,
                "created_at": document_text.created_at,
            }
            if document_text
            else None,
            "tables": [
                {
                    "id": table.id,
                    "table_id": table.table_id,
                    "page_index": table.page_index,
                    "caption": table.caption,
                    "row_headers_json": table.row_headers_json,
                    "col_headers_json": table.col_headers_json,
                    "cells_json": table.cells_json,
                    "bbox_json": table.bbox_json,
                    "created_at": table.created_at,
                }
                for table in tables
            ],
            "segments": [
                {
                    "id": segment.id,
                    "segment_type": segment.segment_type,
                    "start_offset": segment.start_offset,
                    "end_offset": segment.end_offset,
                    "page_index": segment.page_index,
                    "order_index": segment.order_index,
                    "bbox_json": segment.bbox_json,
                    "attrs_json": segment.attrs_json,
                    "created_at": segment.created_at,
                }
                for segment in segments
            ],
            "annotations": annotations,
        }

    def _resolve_run(self, resource_id: str, extraction_run_id: str | None) -> ExtractionRun | None:
        if extraction_run_id:
            run = self.extraction_repo.get_run_by_id(extraction_run_id)
            if run is None:
                raise ExtractionError(f"Extraction run not found: {extraction_run_id}")
            if run.resource_id != resource_id:
                raise ExtractionError(
                    f"Extraction run {extraction_run_id} does not belong to resource {resource_id}"
                )
            return run
        return self.extraction_repo.get_latest_run(resource_id)

    def _build_segment_specs(
        self,
        full_text: str,
        blocks: list[ParsedBlock],
    ) -> list[dict[str, object]]:
        specs: list[dict[str, object]] = []
        text_len = len(full_text)
        if text_len == 0:
            return specs

        has_document_block = False
        for block in blocks:
            span = self._normalize_span(block.start_offset, block.end_offset, text_len)
            if span is None:
                continue
            block_type = f"layout:{block.block_type.strip().lower()}"
            if block_type == "layout:document":
                has_document_block = True
            specs.append(
                {
                    "segment_type": block_type,
                    "start_offset": span[0],
                    "end_offset": span[1],
                    "page_index": block.page_index,
                    "order_index": (
                        self._optional_int(block.attrs.get("order_index"))
                        if isinstance(block.attrs, dict)
                        else None
                    ),
                    "bbox": block.bbox,
                    "attrs": block.attrs,
                }
            )
        if not has_document_block:
            specs.append(
                {
                    "segment_type": "layout:document",
                    "start_offset": 0,
                    "end_offset": text_len,
                    "page_index": None,
                    "order_index": 0,
                    "bbox": None,
                    "attrs": {"role": "root"},
                }
            )

        for idx, (start, end) in enumerate(self._sentence_spans(full_text)):
            specs.append(
                {
                    "segment_type": "structure:sentence",
                    "start_offset": start,
                    "end_offset": end,
                    "page_index": None,
                    "order_index": idx,
                    "bbox": None,
                    "attrs": None,
                }
            )

        deduped: list[dict[str, object]] = []
        seen: set[tuple[object, ...]] = set()
        for spec in specs:
            key = (
                spec["segment_type"],
                spec["start_offset"],
                spec["end_offset"],
                spec["page_index"],
                spec["order_index"],
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(spec)
        return deduped

    def _build_annotation_specs(
        self,
        *,
        full_text: str,
        parser_name: str,
    ) -> list[_AnnotationSpec]:
        if not full_text:
            return []

        out: list[_AnnotationSpec] = [
            _AnnotationSpec(
                layer="provenance",
                category="parser-source",
                label=parser_name,
                confidence=1.0,
                source=parser_name,
                attrs={"strategy": "parser_output"},
                spans=[(0, len(full_text))],
            )
        ]

        def add_pattern(
            *,
            regex: re.Pattern[str],
            layer: str,
            category: str,
            source: str,
            confidence: float,
        ) -> None:
            for start, end, label in self._iter_regex_matches_chunked(full_text, regex):
                span = self._normalize_span(start, end, len(full_text))
                if span is None:
                    continue
                out.append(
                    _AnnotationSpec(
                        layer=layer,
                        category=category,
                        label=label,
                        confidence=confidence,
                        source=source,
                        attrs=None,
                        spans=[span],
                    )
                )
                if len(out) >= self._MAX_AUTO_ANNOTATIONS:
                    return

        add_pattern(
            regex=self._RE_CURRENCY,
            layer="semantic_baseline",
            category="currency",
            source="regex",
            confidence=0.9,
        )
        if len(out) < self._MAX_AUTO_ANNOTATIONS:
            add_pattern(
                regex=self._RE_QUANTITY,
                layer="semantic_baseline",
                category="quantity",
                source="regex",
                confidence=0.8,
            )
        if len(out) < self._MAX_AUTO_ANNOTATIONS:
            add_pattern(
                regex=self._RE_DATE,
                layer="semantic_baseline",
                category="date",
                source="regex",
                confidence=0.8,
            )
        if len(out) < self._MAX_AUTO_ANNOTATIONS:
            add_pattern(
                regex=self._RE_CITATION,
                layer="semantic_baseline",
                category="citation-marker",
                source="regex",
                confidence=0.75,
            )
        if len(out) < self._MAX_AUTO_ANNOTATIONS:
            add_pattern(
                regex=self._RE_PERIOD,
                layer="domain_financial",
                category="period",
                source="regex",
                confidence=0.85,
            )

        if len(out) < self._MAX_AUTO_ANNOTATIONS:
            for term in self._FINANCIAL_TERMS:
                term_pattern = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
                for start, end, _label in self._iter_regex_matches_chunked(full_text, term_pattern):
                    span = self._normalize_span(start, end, len(full_text))
                    if span is not None:
                        out.append(
                            _AnnotationSpec(
                                layer="domain_financial",
                                category="metric",
                                label=full_text[span[0] : span[1]],
                                confidence=0.75,
                                source="keyword",
                                attrs={"term": term},
                                spans=[span],
                            )
                        )
                    if len(out) >= self._MAX_AUTO_ANNOTATIONS:
                        break
                if len(out) >= self._MAX_AUTO_ANNOTATIONS:
                    break
        return out

    @classmethod
    def _iter_regex_matches_chunked(
        cls,
        text: str,
        regex: re.Pattern[str],
    ):
        text_len = len(text)
        if text_len <= 0:
            return
        chunk_size = max(4096, int(cls._REGEX_SCAN_CHUNK_CHARS))
        overlap = max(128, min(int(cls._REGEX_SCAN_OVERLAP_CHARS), chunk_size // 3))
        stride = max(1, chunk_size - overlap)
        offset = 0
        while offset < text_len:
            chunk_end = min(text_len, offset + chunk_size)
            chunk = text[offset:chunk_end]
            min_local_start = 0 if offset == 0 else overlap
            for match in regex.finditer(chunk):
                local_start = int(match.start())
                local_end = int(match.end())
                if local_end <= local_start:
                    continue
                if offset > 0 and local_start < min_local_start:
                    continue
                global_start = offset + local_start
                global_end = min(text_len, offset + local_end)
                if global_end <= global_start:
                    continue
                yield global_start, global_end, match.group(0)
            if chunk_end >= text_len:
                break
            offset += stride
            # Keep background extraction cooperative so web requests can still run.
            time.sleep(0)

    @classmethod
    def _sentence_spans(cls, text: str) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        text_len = len(text)
        for start, end, sentence in cls._iter_regex_matches_chunked(text, cls._RE_SENTENCE):
            span = cls._normalize_span(start, end, text_len)
            if span is not None and sentence.strip():
                spans.append(span)
        return spans

    @staticmethod
    def _normalize_span(start: int, end: int, text_len: int) -> tuple[int, int] | None:
        s = max(0, int(start))
        e = min(int(end), text_len)
        if text_len <= 0:
            return None
        if e <= s:
            return None
        return (s, e)

    @staticmethod
    def _optional_int(value: object) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _normalize_page_index(
        page_index: object,
        *,
        page_count: int | None,
        clamp: bool,
    ) -> int | None:
        try:
            idx = int(page_index)  # type: ignore[arg-type]
        except Exception:
            return None
        if idx < 0:
            return None
        total = 0
        try:
            total = int(page_count) if page_count is not None else 0
        except Exception:
            total = 0
        if total > 0 and idx >= total:
            return total - 1 if clamp else None
        return idx

    @staticmethod
    def _truncate_preview_text(text: str | None, *, max_len: int = 220) -> str | None:
        if text is None:
            return None
        compact = " ".join(str(text).split()).strip()
        if not compact:
            return None
        if len(compact) <= max_len:
            return compact
        return compact[: max_len - 1].rstrip() + "…"

    @staticmethod
    def _safe_bbox_payload(bbox: object) -> dict[str, float] | None:
        if not isinstance(bbox, dict):
            return None
        try:
            x0 = float(bbox.get("x0"))
            y0 = float(bbox.get("y0"))
            x1 = float(bbox.get("x1"))
            y1 = float(bbox.get("y1"))
        except Exception:
            return None
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        return {"x0": x0, "y0": y0, "x1": x1, "y1": y1}

    @classmethod
    def _preview_payload(
        cls,
        *,
        kind: str,
        label: str | None,
        text: str | None,
        page_index: int | None,
        page_count: int | None,
        bbox: object,
    ) -> dict[str, object] | None:
        payload: dict[str, object] = {}
        safe_kind = str(kind or "").strip().lower()
        if safe_kind:
            payload["kind"] = safe_kind[:48]
        safe_label = cls._truncate_preview_text(label, max_len=96)
        if safe_label:
            payload["label"] = safe_label
        safe_text = cls._truncate_preview_text(text, max_len=220)
        if safe_text:
            payload["text"] = safe_text
        if page_count is not None and int(page_count) > 0:
            payload["page_count"] = int(page_count)
        normalized_page_index = cls._normalize_page_index(
            page_index,
            page_count=page_count,
            clamp=False,
        )
        if normalized_page_index is not None:
            payload["page_current"] = normalized_page_index + 1
        safe_bbox = cls._safe_bbox_payload(bbox)
        if safe_bbox is not None:
            payload["bbox"] = safe_bbox
        return payload if payload else None

    @classmethod
    def _table_preview_text(cls, table: ParsedTable) -> str | None:
        if table.caption and table.caption.strip():
            return cls._truncate_preview_text(table.caption, max_len=220)
        values: list[str] = []
        for cell in table.cells:
            value = cls._truncate_preview_text(cell.value, max_len=80)
            if not value:
                continue
            values.append(value)
            if len(values) >= 4:
                break
        if not values:
            return None
        return cls._truncate_preview_text(" | ".join(values), max_len=220)

    @staticmethod
    def derive_table_id(table: ParsedTable, page_index: int | None = None) -> str:
        table_page_index = page_index
        if table_page_index is None:
            try:
                table_page_index = int(table.page_index)
            except Exception:
                table_page_index = 0
        canonical = {
            "caption": (table.caption or "").strip().lower(),
            "page_index": table_page_index,
            "row_headers": [h.strip().lower() for h in table.row_headers],
            "col_headers": [h.strip().lower() for h in table.col_headers],
            "bbox": table.bbox or {},
        }
        digest = compute_bytes_digest(json.dumps(canonical, sort_keys=True).encode("utf-8"))
        return f"sha256:{digest}"

    @staticmethod
    def _table_payload(table: ParsedTable) -> dict[str, object]:
        return {
            "page_index": table.page_index,
            "caption": table.caption,
            "row_headers": table.row_headers,
            "col_headers": table.col_headers,
            "cells": [
                {"row_index": c.row_index, "col_index": c.col_index, "value": c.value}
                for c in table.cells
            ],
            "bbox": table.bbox,
        }

    @staticmethod
    def _block_payload(block: ParsedBlock) -> dict[str, object]:
        return {
            "block_type": block.block_type,
            "start_offset": block.start_offset,
            "end_offset": block.end_offset,
            "page_index": block.page_index,
            "bbox": block.bbox,
            "attrs": block.attrs,
        }

    @staticmethod
    def _annotation_spec_payload(spec: _AnnotationSpec) -> dict[str, object]:
        return {
            "layer": spec.layer,
            "category": spec.category,
            "label": spec.label,
            "confidence": spec.confidence,
            "source": spec.source,
            "attrs": spec.attrs,
            "spans": [{"start": s, "end": e} for s, e in spec.spans],
        }

    @staticmethod
    def _format_worker_exit(exitcode: int | None) -> str:
        if exitcode is None:
            return "parser worker exited with unknown status"
        if exitcode < 0:
            signal_number = -exitcode
            try:
                signal_name = signal.Signals(signal_number).name
            except ValueError:
                signal_name = "UNKNOWN"
            return f"parser worker terminated by signal {signal_number} ({signal_name})"
        return f"parser worker exited with code {exitcode}"

    @staticmethod
    def _env_bool(name: str, *, default: bool) -> bool:
        raw = os.getenv(name)
        if raw is None:
            return default
        normalized = raw.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
        return default

    @staticmethod
    def _env_non_negative_int(name: str, *, default: int) -> int:
        raw = os.getenv(name)
        if raw is None:
            return default
        try:
            parsed = int(raw.strip())
        except ValueError:
            return default
        return max(0, parsed)
