from __future__ import annotations

from pathlib import Path

from stemmacodicum.domain.models.structured_data import StructuredDataRun, StructuredDataTable
from stemmacodicum.infrastructure.db.sqlite import get_connection


class StructuredDataRepo:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def insert_run(self, run: StructuredDataRun) -> None:
        with get_connection(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO structured_data_runs (
                    id,
                    resource_id,
                    data_format,
                    status,
                    scan_truncated,
                    table_count,
                    row_count_observed,
                    config_json,
                    error_message,
                    created_at,
                    finished_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run.id,
                    run.resource_id,
                    run.data_format,
                    run.status,
                    int(run.scan_truncated),
                    run.table_count,
                    run.row_count_observed,
                    run.config_json,
                    run.error_message,
                    run.created_at,
                    run.finished_at,
                ),
            )
            conn.commit()

    def finalize_run(
        self,
        run_id: str,
        *,
        status: str,
        scan_truncated: int,
        table_count: int,
        row_count_observed: int,
        error_message: str | None,
        finished_at: str,
    ) -> None:
        with get_connection(self.db_path) as conn:
            conn.execute(
                """
                UPDATE structured_data_runs
                SET status = ?,
                    scan_truncated = ?,
                    table_count = ?,
                    row_count_observed = ?,
                    error_message = ?,
                    finished_at = ?
                WHERE id = ?
                """,
                (
                    status,
                    int(scan_truncated),
                    table_count,
                    row_count_observed,
                    error_message,
                    finished_at,
                    run_id,
                ),
            )
            conn.commit()

    def insert_tables(self, tables: list[StructuredDataTable]) -> int:
        if not tables:
            return 0
        with get_connection(self.db_path) as conn:
            conn.executemany(
                """
                INSERT INTO structured_data_tables (
                    id,
                    structured_run_id,
                    resource_id,
                    table_name,
                    sheet_name,
                    header_row_index,
                    row_count_observed,
                    scan_truncated,
                    columns_json,
                    sample_rows_json,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        table.id,
                        table.structured_run_id,
                        table.resource_id,
                        table.table_name,
                        table.sheet_name,
                        table.header_row_index,
                        table.row_count_observed,
                        int(table.scan_truncated),
                        table.columns_json,
                        table.sample_rows_json,
                        table.created_at,
                    )
                    for table in tables
                ],
            )
            conn.commit()
        return len(tables)

    def upsert_catalog(
        self,
        *,
        resource_id: str,
        latest_run_id: str,
        data_format: str,
        table_count: int,
        row_count_observed: int,
        scan_truncated: int,
        updated_at: str,
    ) -> None:
        with get_connection(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO structured_data_catalog (
                    resource_id,
                    latest_run_id,
                    data_format,
                    table_count,
                    row_count_observed,
                    scan_truncated,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(resource_id) DO UPDATE SET
                    latest_run_id = excluded.latest_run_id,
                    data_format = excluded.data_format,
                    table_count = excluded.table_count,
                    row_count_observed = excluded.row_count_observed,
                    scan_truncated = excluded.scan_truncated,
                    updated_at = excluded.updated_at
                """,
                (
                    resource_id,
                    latest_run_id,
                    data_format,
                    table_count,
                    row_count_observed,
                    int(scan_truncated),
                    updated_at,
                ),
            )
            conn.commit()

    def get_latest_run(self, resource_id: str) -> StructuredDataRun | None:
        with get_connection(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT *
                FROM structured_data_runs
                WHERE resource_id = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (resource_id,),
            ).fetchone()
        return self._to_run(row) if row else None

    def get_run_by_id(self, run_id: str) -> StructuredDataRun | None:
        with get_connection(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT *
                FROM structured_data_runs
                WHERE id = ?
                LIMIT 1
                """,
                (run_id,),
            ).fetchone()
        return self._to_run(row) if row else None

    def list_tables_for_run(self, run_id: str, *, limit: int = 5000) -> list[StructuredDataTable]:
        with get_connection(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM structured_data_tables
                WHERE structured_run_id = ?
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (run_id, limit),
            ).fetchall()
        return [self._to_table(row) for row in rows]

    def list_tables_for_resource(self, resource_id: str, *, limit: int = 5000) -> list[StructuredDataTable]:
        with get_connection(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM structured_data_tables
                WHERE resource_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (resource_id, limit),
            ).fetchall()
        return [self._to_table(row) for row in rows]

    def get_catalog_row(self, resource_id: str):
        with get_connection(self.db_path) as conn:
            return conn.execute(
                """
                SELECT *
                FROM structured_data_catalog
                WHERE resource_id = ?
                LIMIT 1
                """,
                (resource_id,),
            ).fetchone()

    @staticmethod
    def _to_run(row) -> StructuredDataRun:
        return StructuredDataRun(
            id=row["id"],
            resource_id=row["resource_id"],
            data_format=row["data_format"],
            status=row["status"],
            scan_truncated=int(row["scan_truncated"] or 0),
            table_count=int(row["table_count"] or 0),
            row_count_observed=int(row["row_count_observed"] or 0),
            config_json=row["config_json"],
            error_message=row["error_message"],
            created_at=row["created_at"],
            finished_at=row["finished_at"],
        )

    @staticmethod
    def _to_table(row) -> StructuredDataTable:
        return StructuredDataTable(
            id=row["id"],
            structured_run_id=row["structured_run_id"],
            resource_id=row["resource_id"],
            table_name=row["table_name"],
            sheet_name=row["sheet_name"],
            header_row_index=row["header_row_index"],
            row_count_observed=int(row["row_count_observed"] or 0),
            scan_truncated=int(row["scan_truncated"] or 0),
            columns_json=row["columns_json"],
            sample_rows_json=row["sample_rows_json"],
            created_at=row["created_at"],
        )
