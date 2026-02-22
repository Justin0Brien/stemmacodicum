from __future__ import annotations

import csv
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from xml.etree import ElementTree as ET

from stemmacodicum.application.services.ingestion_policy_service import IngestionPolicyService
from stemmacodicum.core.ids import new_uuid
from stemmacodicum.core.time import now_utc_iso
from stemmacodicum.domain.models.structured_data import StructuredDataRun, StructuredDataTable
from stemmacodicum.infrastructure.db.repos.resource_repo import ResourceRepo
from stemmacodicum.infrastructure.db.repos.structured_data_repo import StructuredDataRepo


@dataclass(slots=True)
class StructuredProfileSummary:
    run_id: str
    resource_id: str
    status: str
    data_format: str
    table_count: int
    row_count_observed: int
    scan_truncated: bool
    error: str | None = None


@dataclass(slots=True)
class StructuredCellMatch:
    resource_id: str
    data_format: str
    table_name: str
    row_number: int
    column_name: str
    column_index: int
    value_raw: str
    filters: dict[str, str]


@dataclass(slots=True)
class _ProfiledTable:
    table_name: str
    sheet_name: str | None
    header_row_index: int | None
    row_count_observed: int
    scan_truncated: bool
    columns: list[str]
    sample_rows: list[dict[str, str]]


@dataclass(slots=True)
class _ProfilePayload:
    data_format: str
    scan_truncated: bool
    row_count_observed: int
    tables: list[_ProfiledTable]
    config: dict[str, object]


class StructuredDataService:
    _CSV_DIALECT_SCAN_BYTES = 131072
    _CSV_HEADER_SCAN_ROWS = 140
    _DEFAULT_MAX_PROFILE_ROWS = 20000
    _DEFAULT_SAMPLE_ROWS = 40
    _METADATA_PREFIXES = {
        "title",
        "location",
        "academic years",
        "data source",
        "sourced from",
        "data source link",
        "data file canonical link",
        "licence",
        "license",
        "code page",
        "disclaimer",
        "table ",
        "last updated",
    }

    def __init__(
        self,
        *,
        resource_repo: ResourceRepo,
        structured_repo: StructuredDataRepo,
        archive_dir: Path,
        policy_service: IngestionPolicyService | None = None,
        max_profile_rows: int = _DEFAULT_MAX_PROFILE_ROWS,
        sample_rows: int = _DEFAULT_SAMPLE_ROWS,
    ) -> None:
        self.resource_repo = resource_repo
        self.structured_repo = structured_repo
        self.archive_dir = archive_dir
        self.policy_service = policy_service or IngestionPolicyService()
        self.max_profile_rows = max(200, int(max_profile_rows))
        self.sample_rows = max(5, int(sample_rows))

    def profile_resource(self, resource_id: str, *, force: bool = False) -> StructuredProfileSummary:
        resource = self.resource_repo.get_by_id(resource_id)
        if resource is None:
            raise ValueError(f"Resource not found: {resource_id}")

        decision = self.policy_service.decide(
            media_type=resource.media_type,
            original_filename=resource.original_filename,
        )
        if decision.resource_kind != "structured_data":
            return StructuredProfileSummary(
                run_id="",
                resource_id=resource_id,
                status="skipped",
                data_format="non-structured",
                table_count=0,
                row_count_observed=0,
                scan_truncated=False,
                error="resource is not classified as structured data",
            )

        latest = self.structured_repo.get_latest_run(resource_id)
        if latest is not None and latest.status == "success" and not force:
            return StructuredProfileSummary(
                run_id=latest.id,
                resource_id=resource_id,
                status="skipped",
                data_format=latest.data_format,
                table_count=latest.table_count,
                row_count_observed=latest.row_count_observed,
                scan_truncated=bool(latest.scan_truncated),
                error=None,
            )

        run_id = new_uuid()
        data_format = self._resolve_data_format(resource.media_type, resource.original_filename)
        started_at = now_utc_iso()
        self.structured_repo.insert_run(
            StructuredDataRun(
                id=run_id,
                resource_id=resource_id,
                data_format=data_format,
                status="running",
                scan_truncated=0,
                table_count=0,
                row_count_observed=0,
                config_json=None,
                error_message=None,
                created_at=started_at,
                finished_at=None,
            )
        )

        archived_path = self.archive_dir / resource.archived_relpath
        if not archived_path.exists():
            self.structured_repo.finalize_run(
                run_id,
                status="failed",
                scan_truncated=0,
                table_count=0,
                row_count_observed=0,
                error_message=f"Archived resource file missing: {archived_path}",
                finished_at=now_utc_iso(),
            )
            return StructuredProfileSummary(
                run_id=run_id,
                resource_id=resource_id,
                status="failed",
                data_format=data_format,
                table_count=0,
                row_count_observed=0,
                scan_truncated=False,
                error=f"Archived resource file missing: {archived_path}",
            )

        try:
            payload = self._profile_path(archived_path=archived_path, data_format=data_format)
        except Exception as exc:
            self.structured_repo.finalize_run(
                run_id,
                status="failed",
                scan_truncated=0,
                table_count=0,
                row_count_observed=0,
                error_message=str(exc),
                finished_at=now_utc_iso(),
            )
            return StructuredProfileSummary(
                run_id=run_id,
                resource_id=resource_id,
                status="failed",
                data_format=data_format,
                table_count=0,
                row_count_observed=0,
                scan_truncated=False,
                error=str(exc),
            )

        created_at = now_utc_iso()
        self.structured_repo.insert_tables(
            [
                StructuredDataTable(
                    id=new_uuid(),
                    structured_run_id=run_id,
                    resource_id=resource_id,
                    table_name=table.table_name,
                    sheet_name=table.sheet_name,
                    header_row_index=table.header_row_index,
                    row_count_observed=table.row_count_observed,
                    scan_truncated=int(table.scan_truncated),
                    columns_json=json.dumps(table.columns, ensure_ascii=True),
                    sample_rows_json=json.dumps(table.sample_rows, ensure_ascii=True),
                    created_at=created_at,
                )
                for table in payload.tables
            ]
        )
        self.structured_repo.finalize_run(
            run_id,
            status="success",
            scan_truncated=int(payload.scan_truncated),
            table_count=len(payload.tables),
            row_count_observed=payload.row_count_observed,
            error_message=None,
            finished_at=now_utc_iso(),
        )
        self.structured_repo.upsert_catalog(
            resource_id=resource_id,
            latest_run_id=run_id,
            data_format=payload.data_format,
            table_count=len(payload.tables),
            row_count_observed=payload.row_count_observed,
            scan_truncated=int(payload.scan_truncated),
            updated_at=now_utc_iso(),
        )
        return StructuredProfileSummary(
            run_id=run_id,
            resource_id=resource_id,
            status="success",
            data_format=payload.data_format,
            table_count=len(payload.tables),
            row_count_observed=payload.row_count_observed,
            scan_truncated=payload.scan_truncated,
            error=None,
        )

    def get_catalog(self, resource_id: str) -> dict[str, Any] | None:
        row = self.structured_repo.get_catalog_row(resource_id)
        if row is None:
            return None
        return {
            "resource_id": row["resource_id"],
            "latest_run_id": row["latest_run_id"],
            "data_format": row["data_format"],
            "table_count": int(row["table_count"] or 0),
            "row_count_observed": int(row["row_count_observed"] or 0),
            "scan_truncated": bool(row["scan_truncated"]),
            "updated_at": row["updated_at"],
        }

    def list_tables(self, resource_id: str, *, limit: int = 500) -> list[dict[str, Any]]:
        tables = self.structured_repo.list_tables_for_resource(resource_id, limit=limit)
        return [
            {
                "id": table.id,
                "structured_run_id": table.structured_run_id,
                "resource_id": table.resource_id,
                "table_name": table.table_name,
                "sheet_name": table.sheet_name,
                "header_row_index": table.header_row_index,
                "row_count_observed": table.row_count_observed,
                "scan_truncated": bool(table.scan_truncated),
                "columns": json.loads(table.columns_json),
                "sample_rows": json.loads(table.sample_rows_json),
                "created_at": table.created_at,
            }
            for table in tables
        ]

    def resolve_data_cell(self, resource_id: str, selector: dict[str, object]) -> StructuredCellMatch:
        resource = self.resource_repo.get_by_id(resource_id)
        if resource is None:
            raise ValueError(f"Resource not found: {resource_id}")
        data_format = self._resolve_data_format(resource.media_type, resource.original_filename)
        archived_path = self.archive_dir / resource.archived_relpath
        if not archived_path.exists():
            raise ValueError(f"Archived resource file missing: {archived_path}")

        if data_format == "csv":
            return self._resolve_csv_cell(resource_id=resource_id, file_path=archived_path, selector=selector)
        if data_format == "xlsx":
            return self._resolve_xlsx_cell(resource_id=resource_id, file_path=archived_path, selector=selector)
        raise ValueError(f"Structured cell lookup is not supported for format: {data_format}")

    def _profile_path(self, *, archived_path: Path, data_format: str) -> _ProfilePayload:
        if data_format == "csv":
            return self._profile_csv(archived_path)
        if data_format == "xlsx":
            return self._profile_xlsx(archived_path)
        raise ValueError(f"Structured data profiling is not supported for format: {data_format}")

    def _profile_csv(self, file_path: Path) -> _ProfilePayload:
        dialect = self._detect_csv_dialect(file_path)
        rows = self._scan_csv_rows(file_path, dialect=dialect, max_rows=self.max_profile_rows)
        header_idx = self._choose_header_index(rows)
        columns = self._header_columns(rows, header_idx)
        row_count_observed = 0
        sample_rows: list[dict[str, str]] = []

        for idx, row in enumerate(rows, start=1):
            if idx <= (header_idx + 1):
                continue
            if not any(str(v).strip() for v in row):
                continue
            row_count_observed += 1
            if len(sample_rows) < self.sample_rows:
                sample_rows.append(self._row_to_dict(row, columns))

        table = _ProfiledTable(
            table_name="csv_main",
            sheet_name=None,
            header_row_index=header_idx,
            row_count_observed=row_count_observed,
            scan_truncated=row_count_observed >= self.max_profile_rows,
            columns=columns,
            sample_rows=sample_rows,
        )
        return _ProfilePayload(
            data_format="csv",
            scan_truncated=table.scan_truncated,
            row_count_observed=row_count_observed,
            tables=[table],
            config={
                "dialect": getattr(dialect, "delimiter", ","),
                "max_profile_rows": self.max_profile_rows,
                "sample_rows": self.sample_rows,
            },
        )

    def _profile_xlsx(self, file_path: Path) -> _ProfilePayload:
        tables: list[_ProfiledTable] = []
        row_total = 0
        scan_truncated = False
        with zipfile.ZipFile(file_path, "r") as archive:
            shared_strings = self._xlsx_shared_strings(archive)
            sheet_entries = self._xlsx_sheet_entries(archive)
            for sheet_name, sheet_path in sheet_entries:
                scanned = self._scan_xlsx_rows(
                    archive=archive,
                    sheet_path=sheet_path,
                    shared_strings=shared_strings,
                    max_rows=self.max_profile_rows,
                )
                header_row_number, columns = self._choose_xlsx_header(scanned)
                row_count_observed = 0
                sample_rows: list[dict[str, str]] = []
                for row_number, row_map in scanned:
                    if header_row_number is not None and row_number <= header_row_number:
                        continue
                    if not any(str(v).strip() for v in row_map.values()):
                        continue
                    row_count_observed += 1
                    if len(sample_rows) < self.sample_rows:
                        sample_rows.append(self._xlsx_row_to_dict(row_map, columns))

                table_truncated = len(scanned) >= self.max_profile_rows
                row_total += row_count_observed
                scan_truncated = scan_truncated or table_truncated
                tables.append(
                    _ProfiledTable(
                        table_name=sheet_name,
                        sheet_name=sheet_name,
                        header_row_index=(header_row_number - 1) if header_row_number else None,
                        row_count_observed=row_count_observed,
                        scan_truncated=table_truncated,
                        columns=columns,
                        sample_rows=sample_rows,
                    )
                )

        return _ProfilePayload(
            data_format="xlsx",
            scan_truncated=scan_truncated,
            row_count_observed=row_total,
            tables=tables,
            config={"max_profile_rows": self.max_profile_rows, "sample_rows": self.sample_rows},
        )

    def _resolve_csv_cell(self, *, resource_id: str, file_path: Path, selector: dict[str, object]) -> StructuredCellMatch:
        value_column = str(selector.get("value_column") or "").strip()
        filters = self._coerce_filter_map(selector.get("filters"))
        if not value_column:
            raise ValueError("DataCellSelector requires 'value_column'.")
        if not filters:
            raise ValueError("DataCellSelector requires non-empty 'filters'.")

        dialect = self._detect_csv_dialect(file_path)
        scan_rows = self._scan_csv_rows(file_path, dialect=dialect, max_rows=self._CSV_HEADER_SCAN_ROWS)
        header_idx = self._choose_header_index(scan_rows)
        columns = self._header_columns(scan_rows, header_idx)
        if not columns:
            raise ValueError("Unable to resolve CSV columns for selector.")
        column_lookup = {self._normalize_column_name(name): i for i, name in enumerate(columns)}

        value_key = self._normalize_column_name(value_column)
        if value_key not in column_lookup:
            raise ValueError(f"CSV column not found: {value_column}")
        filter_indices = {
            key: column_lookup[self._normalize_column_name(key)]
            for key in filters.keys()
            if self._normalize_column_name(key) in column_lookup
        }
        if len(filter_indices) != len(filters):
            missing = sorted(k for k in filters.keys() if k not in filter_indices)
            raise ValueError(f"CSV filter columns not found: {', '.join(missing)}")

        with file_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
            reader = csv.reader(f, dialect=dialect)
            for row_number, row in enumerate(reader, start=1):
                if row_number <= (header_idx + 1):
                    continue
                if not any(str(v).strip() for v in row):
                    continue
                if not self._row_matches(row, filter_indices, filters):
                    continue
                value_col_idx = column_lookup[value_key]
                value = self._row_cell(row, value_col_idx)
                return StructuredCellMatch(
                    resource_id=resource_id,
                    data_format="csv",
                    table_name="csv_main",
                    row_number=row_number,
                    column_name=columns[value_col_idx],
                    column_index=value_col_idx + 1,
                    value_raw=value,
                    filters=filters,
                )
        raise ValueError("No CSV row matched DataCellSelector filters.")

    def _resolve_xlsx_cell(self, *, resource_id: str, file_path: Path, selector: dict[str, object]) -> StructuredCellMatch:
        value_column = str(selector.get("value_column") or "").strip()
        filters = self._coerce_filter_map(selector.get("filters"))
        target_sheet = str(selector.get("sheet_name") or selector.get("table_name") or "").strip()
        if not value_column:
            raise ValueError("DataCellSelector requires 'value_column'.")
        if not filters:
            raise ValueError("DataCellSelector requires non-empty 'filters'.")

        with zipfile.ZipFile(file_path, "r") as archive:
            shared_strings = self._xlsx_shared_strings(archive)
            sheet_entries = self._xlsx_sheet_entries(archive)
            if not sheet_entries:
                raise ValueError("Workbook does not contain worksheets.")
            sheet_name, sheet_path = self._select_sheet_entry(sheet_entries, target_sheet)

            sampled = self._scan_xlsx_rows(
                archive=archive,
                sheet_path=sheet_path,
                shared_strings=shared_strings,
                max_rows=self._CSV_HEADER_SCAN_ROWS,
            )
            header_row_number, columns = self._choose_xlsx_header(sampled)
            if header_row_number is None or not columns:
                raise ValueError(f"Unable to infer header row for sheet: {sheet_name}")
            column_lookup = {self._normalize_column_name(name): i for i, name in enumerate(columns)}
            value_key = self._normalize_column_name(value_column)
            if value_key not in column_lookup:
                raise ValueError(f"Sheet column not found: {value_column}")
            filter_indices = {
                key: column_lookup[self._normalize_column_name(key)]
                for key in filters.keys()
                if self._normalize_column_name(key) in column_lookup
            }
            if len(filter_indices) != len(filters):
                missing = sorted(k for k in filters.keys() if k not in filter_indices)
                raise ValueError(f"Sheet filter columns not found: {', '.join(missing)}")

            for row_number, row_map in self._iter_xlsx_rows(
                archive=archive,
                sheet_path=sheet_path,
                shared_strings=shared_strings,
            ):
                if row_number <= header_row_number:
                    continue
                if not any(str(v).strip() for v in row_map.values()):
                    continue
                if not self._xlsx_row_matches(row_map, filter_indices, filters):
                    continue
                value_col_idx = column_lookup[value_key]
                value = str(row_map.get(value_col_idx + 1, "")).strip()
                return StructuredCellMatch(
                    resource_id=resource_id,
                    data_format="xlsx",
                    table_name=sheet_name,
                    row_number=row_number,
                    column_name=columns[value_col_idx],
                    column_index=value_col_idx + 1,
                    value_raw=value,
                    filters=filters,
                )
        raise ValueError("No XLSX row matched DataCellSelector filters.")

    def _resolve_data_format(self, media_type: str | None, file_name: str | None) -> str:
        suffix = Path(file_name or "").suffix.lower()
        media = (media_type or "").strip().lower()
        if suffix == ".csv" or media == "text/csv":
            return "csv"
        if suffix == ".xlsx" or media == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            return "xlsx"
        if suffix == ".xls" or media == "application/vnd.ms-excel":
            return "xls"
        if suffix == ".ods" or media == "application/vnd.oasis.opendocument.spreadsheet":
            return "ods"
        return suffix.lstrip(".") or "unknown"

    def _detect_csv_dialect(self, file_path: Path) -> csv.Dialect:
        with file_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
            sample = f.read(self._CSV_DIALECT_SCAN_BYTES)
        try:
            return csv.Sniffer().sniff(sample, delimiters=",;\t|")
        except Exception:
            class _DefaultDialect(csv.Dialect):
                delimiter = ","
                quotechar = '"'
                escapechar = None
                doublequote = True
                skipinitialspace = False
                lineterminator = "\n"
                quoting = csv.QUOTE_MINIMAL

            return _DefaultDialect()

    def _scan_csv_rows(self, file_path: Path, *, dialect: csv.Dialect, max_rows: int) -> list[list[str]]:
        rows: list[list[str]] = []
        with file_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
            reader = csv.reader(f, dialect=dialect)
            for row in reader:
                rows.append(list(row))
                if len(rows) >= max_rows:
                    break
        return rows

    def _choose_header_index(self, rows: list[list[str]]) -> int:
        if not rows:
            return 0
        best_idx = 0
        best_score = float("-inf")
        for idx, row in enumerate(rows):
            nonempty = [str(v).strip() for v in row if str(v).strip()]
            if not nonempty:
                continue
            unique_count = len({v.lower() for v in nonempty})
            metadata_penalty = 3 if self._looks_like_metadata_row(row) else 0
            next_nonempty = 0
            if idx + 1 < len(rows):
                next_nonempty = len([str(v).strip() for v in rows[idx + 1] if str(v).strip()])
            score = (len(nonempty) * 2.5) + (unique_count * 1.1) + (0.5 if next_nonempty >= len(nonempty) else 0.0)
            score -= metadata_penalty
            if score > best_score:
                best_score = score
                best_idx = idx
        return max(0, best_idx)

    def _choose_xlsx_header(self, rows: list[tuple[int, dict[int, str]]]) -> tuple[int | None, list[str]]:
        if not rows:
            return None, []
        best_row_number = rows[0][0]
        best_score = float("-inf")
        best_columns: list[str] = []
        for row_number, row_map in rows:
            ordered = [str(row_map[k]).strip() for k in sorted(row_map.keys())]
            nonempty = [v for v in ordered if v]
            if not nonempty:
                continue
            unique_count = len({v.lower() for v in nonempty})
            metadata_penalty = 3 if self._looks_like_metadata_row(ordered) else 0
            score = (len(nonempty) * 2.5) + (unique_count * 1.1) - metadata_penalty
            if score > best_score:
                best_score = score
                best_row_number = row_number
                best_columns = self._xlsx_header_columns(row_map)
        return best_row_number, best_columns

    def _header_columns(self, rows: list[list[str]], header_idx: int) -> list[str]:
        if not rows:
            return []
        header_row = rows[header_idx] if header_idx < len(rows) else rows[0]
        return self._dedupe_columns([str(v).strip() for v in header_row])

    def _xlsx_header_columns(self, header_map: dict[int, str]) -> list[str]:
        if not header_map:
            return []
        ordered = [str(header_map[idx]).strip() for idx in sorted(header_map.keys())]
        return self._dedupe_columns(ordered)

    @staticmethod
    def _dedupe_columns(columns: list[str]) -> list[str]:
        out: list[str] = []
        seen: dict[str, int] = {}
        for idx, value in enumerate(columns, start=1):
            base = value.strip() or f"column_{idx}"
            key = base.lower()
            if key not in seen:
                seen[key] = 1
                out.append(base)
                continue
            seen[key] += 1
            out.append(f"{base}_{seen[key]}")
        return out

    def _row_to_dict(self, row: list[str], columns: list[str]) -> dict[str, str]:
        out: dict[str, str] = {}
        for idx, name in enumerate(columns):
            out[name] = self._row_cell(row, idx)
        return out

    def _xlsx_row_to_dict(self, row_map: dict[int, str], columns: list[str]) -> dict[str, str]:
        out: dict[str, str] = {}
        for idx, name in enumerate(columns, start=1):
            out[name] = str(row_map.get(idx, "")).strip()
        return out

    @staticmethod
    def _row_cell(row: list[str], idx: int) -> str:
        if idx < 0 or idx >= len(row):
            return ""
        return str(row[idx]).strip()

    @classmethod
    def _looks_like_metadata_row(cls, row: list[str] | tuple[str, ...]) -> bool:
        if not row:
            return False
        first = str(row[0] if len(row) >= 1 else "").strip().lower()
        if not first:
            return False
        return any(first.startswith(prefix) for prefix in cls._METADATA_PREFIXES)

    @staticmethod
    def _normalize_column_name(value: str) -> str:
        return " ".join(str(value or "").strip().lower().split())

    @staticmethod
    def _coerce_filter_map(value: object) -> dict[str, str]:
        if not isinstance(value, dict):
            return {}
        out: dict[str, str] = {}
        for key, item in value.items():
            k = str(key or "").strip()
            if not k:
                continue
            out[k] = str(item or "").strip()
        return out

    def _row_matches(self, row: list[str], filter_indices: dict[str, int], filters: dict[str, str]) -> bool:
        for col_name, idx in filter_indices.items():
            expected = filters.get(col_name, "")
            actual = self._row_cell(row, idx)
            if actual != expected:
                return False
        return True

    def _xlsx_row_matches(
        self,
        row_map: dict[int, str],
        filter_indices: dict[str, int],
        filters: dict[str, str],
    ) -> bool:
        for col_name, idx in filter_indices.items():
            expected = filters.get(col_name, "")
            actual = str(row_map.get(idx + 1, "")).strip()
            if actual != expected:
                return False
        return True

    def _xlsx_sheet_entries(self, archive: zipfile.ZipFile) -> list[tuple[str, str]]:
        if "xl/workbook.xml" not in archive.namelist():
            return []
        workbook_root = ET.fromstring(archive.read("xl/workbook.xml"))
        rel_map: dict[str, str] = {}
        rels_path = "xl/_rels/workbook.xml.rels"
        if rels_path in archive.namelist():
            rels_root = ET.fromstring(archive.read(rels_path))
            for rel in rels_root:
                if self._local_tag(rel.tag) != "Relationship":
                    continue
                rel_id = str(rel.attrib.get("Id") or "").strip()
                target = str(rel.attrib.get("Target") or "").strip()
                if rel_id and target:
                    rel_map[rel_id] = target

        entries: list[tuple[str, str]] = []
        for node in workbook_root.iter():
            if self._local_tag(node.tag) != "sheet":
                continue
            name = str(node.attrib.get("name") or "").strip() or "Sheet"
            rel_id = ""
            for key in node.attrib.keys():
                if key.endswith("}id") or key == "r:id":
                    rel_id = str(node.attrib.get(key) or "").strip()
                    break
            target = rel_map.get(rel_id, "")
            if not target:
                continue
            norm = self._resolve_xlsx_target(target)
            entries.append((name, norm))
        return entries

    @staticmethod
    def _resolve_xlsx_target(target: str) -> str:
        normalized = target.replace("\\", "/").strip()
        if normalized.startswith("/"):
            normalized = normalized.lstrip("/")
        if not normalized.startswith("xl/"):
            normalized = f"xl/{normalized}"
        return str(PurePosixPath(normalized))

    def _scan_xlsx_rows(
        self,
        *,
        archive: zipfile.ZipFile,
        sheet_path: str,
        shared_strings: list[str],
        max_rows: int,
    ) -> list[tuple[int, dict[int, str]]]:
        out: list[tuple[int, dict[int, str]]] = []
        for row in self._iter_xlsx_rows(
            archive=archive,
            sheet_path=sheet_path,
            shared_strings=shared_strings,
        ):
            out.append(row)
            if len(out) >= max_rows:
                break
        return out

    def _iter_xlsx_rows(
        self,
        *,
        archive: zipfile.ZipFile,
        sheet_path: str,
        shared_strings: list[str],
    ):
        with archive.open(sheet_path, "r") as stream:
            row_seq = 0
            for _event, elem in ET.iterparse(stream, events=("end",)):
                if self._local_tag(elem.tag) != "row":
                    continue
                row_seq += 1
                row_number = int(elem.attrib.get("r", row_seq))
                row_map: dict[int, str] = {}
                cell_seq = 0
                for child in list(elem):
                    if self._local_tag(child.tag) != "c":
                        continue
                    cell_seq += 1
                    ref = child.attrib.get("r")
                    col_idx = self._cell_ref_to_col_index(ref) if ref else cell_seq
                    col_idx = max(1, col_idx)
                    row_map[col_idx] = self._xlsx_cell_value(child, shared_strings)
                yield row_number, row_map
                elem.clear()

    def _xlsx_shared_strings(self, archive: zipfile.ZipFile) -> list[str]:
        if "xl/sharedStrings.xml" not in archive.namelist():
            return []
        values: list[str] = []
        with archive.open("xl/sharedStrings.xml", "r") as stream:
            for _event, elem in ET.iterparse(stream, events=("end",)):
                if self._local_tag(elem.tag) != "si":
                    continue
                values.append(self._joined_text(elem))
                elem.clear()
        return values

    def _select_sheet_entry(self, entries: list[tuple[str, str]], target_sheet: str) -> tuple[str, str]:
        if not target_sheet:
            return entries[0]
        normalized_target = self._normalize_column_name(target_sheet)
        for name, path in entries:
            if self._normalize_column_name(name) == normalized_target:
                return name, path
        raise ValueError(f"Sheet not found in workbook: {target_sheet}")

    @classmethod
    def _xlsx_cell_value(cls, cell_node: ET.Element, shared_strings: list[str]) -> str:
        cell_type = (cell_node.attrib.get("t") or "").strip().lower()
        if cell_type == "inlineStr":
            for child in list(cell_node):
                if cls._local_tag(child.tag) == "is":
                    return cls._joined_text(child)
        value_text = ""
        for child in list(cell_node):
            if cls._local_tag(child.tag) == "v":
                value_text = (child.text or "").strip()
                break
        if cell_type == "s":
            try:
                idx = int(value_text)
                if 0 <= idx < len(shared_strings):
                    return shared_strings[idx]
            except Exception:
                return value_text
        return value_text

    @staticmethod
    def _cell_ref_to_col_index(cell_ref: str | None) -> int:
        if not cell_ref:
            return 0
        letters = "".join(ch for ch in str(cell_ref) if ch.isalpha()).upper()
        if not letters:
            return 0
        total = 0
        for ch in letters:
            total = total * 26 + (ord(ch) - ord("A") + 1)
        return total

    @staticmethod
    def _local_tag(tag: str) -> str:
        if "}" in tag:
            return tag.split("}", 1)[1]
        return tag

    @classmethod
    def _joined_text(cls, node: ET.Element) -> str:
        parts: list[str] = []
        for chunk in node.itertext():
            clean = " ".join(str(chunk).split())
            if clean:
                parts.append(clean)
        return " ".join(parts).strip()
