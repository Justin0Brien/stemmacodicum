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
