from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from stemmacodicum.core.errors import ExtractionError
from stemmacodicum.core.hashing import compute_bytes_digest
from stemmacodicum.core.ids import new_uuid
from stemmacodicum.core.time import now_utc_iso
from stemmacodicum.domain.models.extraction import ExtractedTable, ExtractionRun
from stemmacodicum.infrastructure.db.repos.extraction_repo import ExtractionRepo
from stemmacodicum.infrastructure.db.repos.resource_repo import ResourceRepo
from stemmacodicum.infrastructure.parsers.docling_adapter import (
    DoclingAdapter,
    DoclingRuntimeOptions,
    ParsedTable,
)


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


class ExtractionService:
    def __init__(
        self,
        resource_repo: ResourceRepo,
        extraction_repo: ExtractionRepo,
        archive_dir: Path,
        docling_runtime_options: DoclingRuntimeOptions | None = None,
    ) -> None:
        self.resource_repo = resource_repo
        self.extraction_repo = extraction_repo
        self.archive_dir = archive_dir
        self.docling_runtime_options = docling_runtime_options

    def extract_resource(self, resource_id: str, parser_profile: str = "default") -> ExtractSummary:
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

        output_payload = {
            "resource_id": resource_id,
            "tables": [self._table_payload(t) for t in parse_result.tables],
        }
        output_digest = compute_bytes_digest(
            json.dumps(output_payload, sort_keys=True).encode("utf-8")
        )

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
                bbox_json=json.dumps(parsed_table.bbox, ensure_ascii=True)
                if parsed_table.bbox
                else None,
                created_at=now_utc_iso(),
            )
            self.extraction_repo.insert_table(table)

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
        )

    def list_tables(self, resource_id: str, limit: int = 100) -> list[ExtractedTable]:
        return self.extraction_repo.list_tables_for_resource(resource_id=resource_id, limit=limit)

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
