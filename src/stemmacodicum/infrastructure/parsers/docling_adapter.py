from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from stemmacodicum.core.hashing import compute_bytes_digest


@dataclass(slots=True)
class ParsedCell:
    row_index: int
    col_index: int
    value: str


@dataclass(slots=True)
class ParsedTable:
    page_index: int
    caption: str | None
    row_headers: list[str]
    col_headers: list[str]
    cells: list[ParsedCell]
    bbox: dict[str, float] | None = None


@dataclass(slots=True)
class ParseResult:
    parser_name: str
    parser_version: str
    config_digest: str
    tables: list[ParsedTable]


class DoclingAdapter:
    """
    Parser adapter with two modes:
    - Markdown/plaintext table extraction (always available).
    - PDF extraction via docling (optional, if installed).
    """

    def __init__(self, profile: str = "default") -> None:
        self.profile = profile

    def parse_resource(self, file_path: Path, media_type: str) -> ParseResult:
        if media_type == "application/pdf":
            try:
                return self._parse_pdf_docling(file_path)
            except ImportError as exc:
                raise RuntimeError(
                    "Docling is not installed. Install docling to extract PDF tables."
                ) from exc

        if media_type in {"text/markdown", "text/plain", "text/csv"}:
            return self._parse_text_tables(file_path, media_type)

        # Fallback to plain-text scanner for unknown text-like types.
        return self._parse_text_tables(file_path, media_type)

    def _parse_pdf_docling(self, file_path: Path) -> ParseResult:
        # Keep import local so non-PDF users don't require docling.
        from docling.document_converter import DocumentConverter  # type: ignore

        converter = DocumentConverter()
        result = converter.convert(str(file_path))

        tables: list[ParsedTable] = []
        for idx, table in enumerate(getattr(result.document, "tables", []) or []):
            col_headers: list[str] = []
            row_headers: list[str] = []
            cells: list[ParsedCell] = []

            for r_idx, row in enumerate(getattr(table, "data", []) or []):
                row_values = [str(cell) if cell is not None else "" for cell in row]
                if r_idx == 0:
                    col_headers = row_values
                    continue
                if row_values:
                    row_headers.append(row_values[0])
                for c_idx, value in enumerate(row_values):
                    cells.append(ParsedCell(row_index=max(r_idx - 1, 0), col_index=c_idx, value=value))

            caption = getattr(table, "caption", None)
            tables.append(
                ParsedTable(
                    page_index=idx,
                    caption=caption,
                    row_headers=row_headers,
                    col_headers=col_headers,
                    cells=cells,
                    bbox=None,
                )
            )

        config = {"profile": self.profile, "mode": "docling_pdf"}
        return ParseResult(
            parser_name="docling",
            parser_version="auto",
            config_digest=compute_bytes_digest(json.dumps(config, sort_keys=True).encode("utf-8")),
            tables=tables,
        )

    def _parse_text_tables(self, file_path: Path, media_type: str) -> ParseResult:
        text = file_path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()

        tables: list[ParsedTable] = []
        i = 0
        while i < len(lines):
            if self._looks_like_table_header(lines, i):
                table, end = self._read_markdown_table(lines, i)
                tables.append(table)
                i = end
                continue
            i += 1

        config = {"profile": self.profile, "mode": "text_table", "media_type": media_type}
        return ParseResult(
            parser_name="text-table-parser",
            parser_version="0.1",
            config_digest=compute_bytes_digest(json.dumps(config, sort_keys=True).encode("utf-8")),
            tables=tables,
        )

    @staticmethod
    def _looks_like_table_header(lines: list[str], idx: int) -> bool:
        if idx + 1 >= len(lines):
            return False
        header = lines[idx].strip()
        divider = lines[idx + 1].strip()
        if "|" not in header or "|" not in divider:
            return False

        divider_cells = [c.strip() for c in divider.strip("|").split("|")]
        if not divider_cells:
            return False
        return all(cell and set(cell) <= {"-", ":", " "} for cell in divider_cells)

    def _read_markdown_table(self, lines: list[str], start_idx: int) -> tuple[ParsedTable, int]:
        caption = self._find_caption(lines, start_idx)

        header_cells = self._split_pipe_row(lines[start_idx])
        body_rows: list[list[str]] = []

        i = start_idx + 2
        while i < len(lines):
            line = lines[i].strip()
            if not line or "|" not in line:
                break
            row = self._split_pipe_row(lines[i])
            if len(row) != len(header_cells):
                break
            body_rows.append(row)
            i += 1

        row_headers: list[str] = []
        cells: list[ParsedCell] = []

        for r_idx, row in enumerate(body_rows):
            row_headers.append(row[0] if row else "")
            for c_idx, value in enumerate(row):
                cells.append(ParsedCell(row_index=r_idx, col_index=c_idx, value=value))

        table = ParsedTable(
            page_index=0,
            caption=caption,
            row_headers=row_headers,
            col_headers=header_cells,
            cells=cells,
            bbox=None,
        )
        return table, i

    @staticmethod
    def _split_pipe_row(line: str) -> list[str]:
        row = line.strip().strip("|")
        return [part.strip() for part in row.split("|")]

    @staticmethod
    def _find_caption(lines: list[str], start_idx: int) -> str | None:
        for offset in range(1, 4):
            idx = start_idx - offset
            if idx < 0:
                break
            probe = lines[idx].strip()
            if probe.lower().startswith("table"):
                return probe
            if probe:
                break
        return None
