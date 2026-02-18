from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ExtractionRun:
    id: str
    resource_id: str
    parser_name: str
    parser_version: str
    config_digest: str
    output_digest: str
    status: str
    created_at: str


@dataclass(slots=True)
class ExtractedTable:
    id: str
    extraction_run_id: str
    resource_id: str
    table_id: str
    page_index: int
    caption: str | None
    row_headers_json: str
    col_headers_json: str
    cells_json: str
    bbox_json: str | None
    created_at: str


@dataclass(slots=True)
class DocumentText:
    id: str
    extraction_run_id: str
    resource_id: str
    text_content: str
    text_digest_sha256: str
    char_count: int
    created_at: str


@dataclass(slots=True)
class TextSegment:
    id: str
    document_text_id: str
    extraction_run_id: str
    resource_id: str
    segment_type: str
    start_offset: int
    end_offset: int
    page_index: int | None
    order_index: int | None
    bbox_json: str | None
    attrs_json: str | None
    created_at: str


@dataclass(slots=True)
class TextAnnotation:
    id: str
    document_text_id: str
    extraction_run_id: str
    resource_id: str
    layer: str
    category: str
    label: str | None
    confidence: float | None
    source: str | None
    attrs_json: str | None
    created_at: str


@dataclass(slots=True)
class AnnotationSpan:
    id: str
    annotation_id: str
    start_offset: int
    end_offset: int
    span_order: int
    created_at: str


@dataclass(slots=True)
class AnnotationRelation:
    id: str
    document_text_id: str
    extraction_run_id: str
    resource_id: str
    relation_type: str
    from_annotation_id: str
    to_annotation_id: str
    attrs_json: str | None
    created_at: str
