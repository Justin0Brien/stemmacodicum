from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

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
    ParsedBlock,
    ParsedTable,
)
from stemmacodicum.application.services.vector_service import VectorIndexingService


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
    _MAX_AUTO_ANNOTATIONS = 10_000

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

    def extract_resource(
        self,
        resource_id: str,
        parser_profile: str = "default",
        progress_callback: Callable[[dict[str, object]], None] | None = None,
    ) -> ExtractSummary:
        def emit(payload: dict[str, object]) -> None:
            if progress_callback is None:
                return
            progress_callback(payload)

        emit(
            {
                "stage": "extract",
                "state": "active",
                "progress": 4,
                "detail": "Starting text extraction.",
            }
        )
        resource = self.resource_repo.get_by_id(resource_id)
        if resource is None:
            raise ExtractionError(f"Resource not found: {resource_id}")

        archived_path = self.archive_dir / resource.archived_relpath
        if not archived_path.exists():
            raise ExtractionError(f"Archived resource file missing: {archived_path}")

        adapter = DoclingAdapter(profile=parser_profile, runtime_options=self.docling_runtime_options)
        try:
            parse_result = adapter.parse_resource(archived_path, resource.media_type)
        except Exception as exc:
            raise ExtractionError(f"Extraction failed for {resource_id}: {exc}") from exc

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

        for parsed_table in parse_result.tables:
            table_id = self.derive_table_id(parsed_table)
            table = ExtractedTable(
                id=new_uuid(),
                extraction_run_id=run.id,
                resource_id=resource_id,
                table_id=table_id,
                page_index=parsed_table.page_index,
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
        for spec in segment_specs:
            segment = TextSegment(
                id=new_uuid(),
                document_text_id=document_text.id,
                extraction_run_id=run.id,
                resource_id=resource_id,
                segment_type=str(spec["segment_type"]),
                start_offset=int(spec["start_offset"]),
                end_offset=int(spec["end_offset"]),
                page_index=spec.get("page_index"),
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

        persisted_annotations = 0
        for spec in annotation_specs:
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

        emit(
            {
                "stage": "extract",
                "state": "done",
                "progress": 100,
                "detail": "Text extraction complete.",
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
            for match in regex.finditer(full_text):
                span = self._normalize_span(match.start(), match.end(), len(full_text))
                if span is None:
                    continue
                out.append(
                    _AnnotationSpec(
                        layer=layer,
                        category=category,
                        label=match.group(0),
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
                for match in term_pattern.finditer(full_text):
                    span = self._normalize_span(match.start(), match.end(), len(full_text))
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
    def _sentence_spans(cls, text: str) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        text_len = len(text)
        for match in cls._RE_SENTENCE.finditer(text):
            span = cls._normalize_span(match.start(), match.end(), text_len)
            if span is not None and text[span[0] : span[1]].strip():
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
    def derive_table_id(table: ParsedTable) -> str:
        canonical = {
            "caption": (table.caption or "").strip().lower(),
            "page_index": table.page_index,
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
