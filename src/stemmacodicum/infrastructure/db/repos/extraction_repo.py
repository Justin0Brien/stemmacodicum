from __future__ import annotations

from pathlib import Path

from stemmacodicum.domain.models.extraction import ExtractedTable, ExtractionRun
from stemmacodicum.infrastructure.db.sqlite import get_connection


class ExtractionRepo:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def insert_run(self, run: ExtractionRun) -> None:
        with get_connection(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO extraction_runs (
                    id,
                    resource_id,
                    parser_name,
                    parser_version,
                    config_digest,
                    output_digest,
                    status,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run.id,
                    run.resource_id,
                    run.parser_name,
                    run.parser_version,
                    run.config_digest,
                    run.output_digest,
                    run.status,
                    run.created_at,
                ),
            )
            conn.commit()

    def insert_table(self, table: ExtractedTable) -> None:
        with get_connection(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO extracted_tables (
                    id,
                    extraction_run_id,
                    resource_id,
                    table_id,
                    page_index,
                    caption,
                    row_headers_json,
                    col_headers_json,
                    cells_json,
                    bbox_json,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    table.id,
                    table.extraction_run_id,
                    table.resource_id,
                    table.table_id,
                    table.page_index,
                    table.caption,
                    table.row_headers_json,
                    table.col_headers_json,
                    table.cells_json,
                    table.bbox_json,
                    table.created_at,
                ),
            )
            conn.commit()

    def list_tables_for_resource(self, resource_id: str, limit: int = 100) -> list[ExtractedTable]:
        with get_connection(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT * FROM extracted_tables
                WHERE resource_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (resource_id, limit),
            ).fetchall()
        return [self._to_table(row) for row in rows]

    def list_recent_runs(self, resource_id: str, limit: int = 20) -> list[ExtractionRun]:
        with get_connection(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT * FROM extraction_runs
                WHERE resource_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (resource_id, limit),
            ).fetchall()
        return [self._to_run(row) for row in rows]

    def get_table_by_table_id(self, resource_id: str, table_id: str) -> ExtractedTable | None:
        with get_connection(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT *
                FROM extracted_tables
                WHERE resource_id = ? AND table_id = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (resource_id, table_id),
            ).fetchone()
        return self._to_table(row) if row else None

    @staticmethod
    def _to_run(row) -> ExtractionRun:
        return ExtractionRun(
            id=row["id"],
            resource_id=row["resource_id"],
            parser_name=row["parser_name"],
            parser_version=row["parser_version"],
            config_digest=row["config_digest"],
            output_digest=row["output_digest"],
            status=row["status"],
            created_at=row["created_at"],
        )

    @staticmethod
    def _to_table(row) -> ExtractedTable:
        return ExtractedTable(
            id=row["id"],
            extraction_run_id=row["extraction_run_id"],
            resource_id=row["resource_id"],
            table_id=row["table_id"],
            page_index=row["page_index"],
            caption=row["caption"],
            row_headers_json=row["row_headers_json"],
            col_headers_json=row["col_headers_json"],
            cells_json=row["cells_json"],
            bbox_json=row["bbox_json"],
            created_at=row["created_at"],
        )
