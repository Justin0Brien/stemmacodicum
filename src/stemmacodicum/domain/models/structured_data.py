from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class StructuredDataRun:
    id: str
    resource_id: str
    data_format: str
    status: str
    scan_truncated: int
    table_count: int
    row_count_observed: int
    config_json: str | None
    error_message: str | None
    created_at: str
    finished_at: str | None


@dataclass(slots=True)
class StructuredDataTable:
    id: str
    structured_run_id: str
    resource_id: str
    table_name: str
    sheet_name: str | None
    header_row_index: int | None
    row_count_observed: int
    scan_truncated: int
    columns_json: str
    sample_rows_json: str
    created_at: str
