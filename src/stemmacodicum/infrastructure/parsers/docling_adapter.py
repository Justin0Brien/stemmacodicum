from __future__ import annotations

import csv
import json
import logging
import os
import platform
import re
import subprocess
import time
import zipfile
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from stemmacodicum.core.hashing import compute_bytes_digest

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class DoclingRuntimeOptions:
    """Runtime controls for docling PDF extraction."""

    auto_tune: bool = True
    use_threaded_pipeline: bool | None = None
    device: str | None = None
    num_threads: int | None = None
    layout_batch_size: int | None = None
    ocr_batch_size: int | None = None
    table_batch_size: int | None = None
    queue_max_size: int | None = None
    log_settings: bool = True


@dataclass(slots=True)
class SystemResources:
    cpu_cores: int
    memory_gb: int
    platform_name: str
    machine: str


@dataclass(slots=True)
class ResolvedDoclingRuntime:
    mode: str
    use_threaded_pipeline: bool
    device: str
    num_threads: int
    layout_batch_size: int
    ocr_batch_size: int
    table_batch_size: int
    queue_max_size: int
    cpu_cores: int
    memory_gb: int
    platform_name: str
    machine: str

    def to_config(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "use_threaded_pipeline": self.use_threaded_pipeline,
            "device": self.device,
            "num_threads": self.num_threads,
            "layout_batch_size": self.layout_batch_size,
            "ocr_batch_size": self.ocr_batch_size,
            "table_batch_size": self.table_batch_size,
            "queue_max_size": self.queue_max_size,
            "cpu_cores": self.cpu_cores,
            "memory_gb": self.memory_gb,
            "platform": self.platform_name,
            "machine": self.machine,
        }


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
class ParsedBlock:
    block_type: str
    start_offset: int
    end_offset: int
    page_index: int | None = None
    bbox: dict[str, float] | None = None
    attrs: dict[str, object] | None = None


@dataclass(slots=True)
class ParseResult:
    parser_name: str
    parser_version: str
    config_digest: str
    tables: list[ParsedTable]
    full_text: str = ""
    blocks: list[ParsedBlock] = field(default_factory=list)
    elapsed_seconds: float | None = None
    page_count: int | None = None
    timings: dict[str, float] | None = None


class _HTMLTableTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.text_chunks: list[str] = []
        self.tables: list[list[list[str]]] = []
        self._in_table = False
        self._current_table: list[list[str]] | None = None
        self._current_row: list[str] | None = None
        self._in_cell = False
        self._cell_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        normalized = tag.lower()
        if normalized == "table":
            self._in_table = True
            self._current_table = []
            return
        if normalized == "tr" and self._in_table:
            self._current_row = []
            return
        if normalized in {"td", "th"} and self._current_row is not None:
            self._in_cell = True
            self._cell_parts = []

    def handle_data(self, data: str) -> None:
        text = " ".join(data.split())
        if not text:
            return
        self.text_chunks.append(text)
        if self._in_cell:
            self._cell_parts.append(text)

    def handle_endtag(self, tag: str) -> None:
        normalized = tag.lower()
        if normalized in {"td", "th"} and self._in_cell:
            value = " ".join(self._cell_parts).strip()
            if self._current_row is not None:
                self._current_row.append(value)
            self._in_cell = False
            self._cell_parts = []
            return
        if normalized == "tr" and self._current_row is not None:
            if any(cell.strip() for cell in self._current_row):
                if self._current_table is None:
                    self._current_table = []
                self._current_table.append(self._current_row)
            self._current_row = None
            return
        if normalized == "table" and self._in_table:
            if self._current_table:
                self.tables.append(self._current_table)
            self._in_table = False
            self._current_table = None


class DoclingAdapter:
    """
    Parser adapter with two modes:
    - Markdown/plaintext table extraction (always available).
    - PDF extraction via docling (optional, if installed).
    """

    TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".rst", ".adoc", ".tex"}
    HTML_EXTENSIONS = {".html", ".htm", ".xhtml"}
    XML_EXTENSIONS = {".xml", ".svg", ".fb2", ".dita", ".dbk"}
    OOXML_EXTENSIONS = {".docx", ".xlsx", ".pptx"}
    ODF_EXTENSIONS = {".odt", ".ods", ".odp", ".odg"}
    ZIP_XML_EXTENSIONS = {".epub", ".oxps", ".3mf"}

    SUPPORTED_EXTENSIONS = {
        ".pdf",
        *TEXT_EXTENSIONS,
        *HTML_EXTENSIONS,
        *XML_EXTENSIONS,
        *OOXML_EXTENSIONS,
        *ODF_EXTENSIONS,
        *ZIP_XML_EXTENSIONS,
    }

    SUPPORTED_MEDIA_TYPES = {
        "application/pdf",
        "text/plain",
        "text/markdown",
        "text/csv",
        "text/html",
        "application/xhtml+xml",
        "application/xml",
        "text/xml",
        "image/svg+xml",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "application/vnd.oasis.opendocument.text",
        "application/vnd.oasis.opendocument.spreadsheet",
        "application/vnd.oasis.opendocument.presentation",
        "application/vnd.oasis.opendocument.graphics",
        "application/epub+zip",
        "application/oxps",
        "model/3mf",
    }

    def __init__(
        self,
        profile: str = "default",
        runtime_options: DoclingRuntimeOptions | None = None,
    ) -> None:
        self.profile = profile
        self.runtime_options = runtime_options or DoclingRuntimeOptions()

    @classmethod
    def supports(cls, media_type: str | None, file_name: str | None = None) -> bool:
        normalized_media = (media_type or "").strip().lower()
        if normalized_media in cls.SUPPORTED_MEDIA_TYPES:
            return True
        if file_name:
            suffix = Path(file_name).suffix.lower()
            if suffix in cls.SUPPORTED_EXTENSIONS:
                return True
        return False

    def parse_resource(self, file_path: Path, media_type: str) -> ParseResult:
        suffix = file_path.suffix.lower()
        if media_type == "application/pdf":
            try:
                return self._parse_pdf_docling(file_path)
            except ImportError as exc:
                raise RuntimeError(
                    "Docling is not installed. Install docling to extract PDF tables."
                ) from exc

        if media_type == "text/csv" or suffix == ".csv":
            return self._parse_csv_text(file_path, media_type)

        if media_type in {"text/markdown", "text/plain"} or suffix in self.TEXT_EXTENSIONS:
            return self._parse_text_tables(file_path, media_type)

        if media_type in {"text/html", "application/xhtml+xml"} or suffix in self.HTML_EXTENSIONS:
            return self._parse_html_markup(file_path, media_type)

        if media_type in {"application/xml", "text/xml", "image/svg+xml"} or suffix in self.XML_EXTENSIONS:
            return self._parse_xml_markup(file_path, media_type)

        if suffix == ".docx":
            return self._parse_docx(file_path)
        if suffix == ".xlsx":
            return self._parse_xlsx(file_path)
        if suffix == ".pptx":
            return self._parse_pptx(file_path)
        if suffix in self.ODF_EXTENSIONS:
            return self._parse_odf(file_path, suffix=suffix)
        if suffix in self.ZIP_XML_EXTENSIONS:
            return self._parse_zip_xml_bundle(file_path, suffix=suffix)

        # Last-resort fallback for unknown but text-like resources.
        return self._parse_text_tables(file_path, media_type)

    def _parse_pdf_docling(self, file_path: Path) -> ParseResult:
        # Keep import local so non-PDF users don't require docling.
        from docling.datamodel.accelerator_options import AcceleratorOptions  # type: ignore
        from docling.datamodel.base_models import InputFormat  # type: ignore
        from docling.datamodel.pipeline_options import PdfPipelineOptions  # type: ignore
        from docling.document_converter import DocumentConverter, PdfFormatOption  # type: ignore

        ThreadedPdfPipelineOptions: type[Any] | None = None
        ThreadedStandardPdfPipeline: type[Any] | None = None
        try:
            from docling.datamodel.pipeline_options import ThreadedPdfPipelineOptions as _ThreadedOpts  # type: ignore

            ThreadedPdfPipelineOptions = _ThreadedOpts
        except Exception:
            ThreadedPdfPipelineOptions = None
        try:
            from docling.pipeline.threaded_standard_pdf_pipeline import ThreadedStandardPdfPipeline as _ThreadedPipeline  # type: ignore

            ThreadedStandardPdfPipeline = _ThreadedPipeline
        except Exception:
            ThreadedStandardPdfPipeline = None

        threaded_supported = (
            ThreadedPdfPipelineOptions is not None and ThreadedStandardPdfPipeline is not None
        )
        runtime = self._resolve_runtime_settings(threaded_supported=threaded_supported)
        self._log_runtime_settings(file_path=file_path, runtime=runtime, threaded_supported=threaded_supported)

        accelerator = AcceleratorOptions(device=runtime.device, num_threads=runtime.num_threads)
        pipeline_cls: type[Any] = PdfPipelineOptions
        if runtime.use_threaded_pipeline and ThreadedPdfPipelineOptions is not None:
            pipeline_cls = ThreadedPdfPipelineOptions

        pipeline_values = {
            "accelerator_options": accelerator,
            "layout_batch_size": runtime.layout_batch_size,
            "ocr_batch_size": runtime.ocr_batch_size,
            "table_batch_size": runtime.table_batch_size,
            "queue_max_size": runtime.queue_max_size,
        }
        pipeline_options = self._build_model(model_cls=pipeline_cls, values=pipeline_values)

        pdf_option_values: dict[str, object] = {"pipeline_options": pipeline_options}
        if runtime.use_threaded_pipeline and ThreadedStandardPdfPipeline is not None:
            pdf_option_values["pipeline_cls"] = ThreadedStandardPdfPipeline
        pdf_format_option = self._build_model(model_cls=PdfFormatOption, values=pdf_option_values)

        converter = DocumentConverter(format_options={InputFormat.PDF: pdf_format_option})
        print(f"[docling] start file={file_path.name}")
        started = time.perf_counter()
        result = converter.convert(str(file_path))
        elapsed_seconds = time.perf_counter() - started

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
            page_index, bbox = self._extract_table_geometry(table, fallback_index=idx)
            tables.append(
                ParsedTable(
                    page_index=page_index,
                    caption=caption,
                    row_headers=row_headers,
                    col_headers=col_headers,
                    cells=cells,
                    bbox=bbox,
                )
            )

        parser_version = "auto"
        try:
            import docling  # type: ignore

            parser_version = str(getattr(docling, "__version__", "auto"))
        except Exception:
            parser_version = "auto"

        page_count = self._extract_page_count(result)
        timing_totals = self._extract_timing_totals(result)
        pages_per_second = (page_count / elapsed_seconds) if page_count and elapsed_seconds > 0 else None
        timing_summary = self._timing_summary(timing_totals)
        done_bits = [
            f"file={file_path.name}",
            f"elapsed={elapsed_seconds:.2f}s",
            f"pages={page_count}",
            f"tables={len(tables)}",
        ]
        if pages_per_second is not None:
            done_bits.append(f"rate={pages_per_second:.2f}pg/s")
        if timing_summary:
            done_bits.append(f"timings={timing_summary}")
        print("[docling] done " + " ".join(done_bits))

        config = {
            "profile": self.profile,
            "mode": "docling_pdf",
            "runtime": runtime.to_config(),
            "threaded_supported": threaded_supported,
        }
        full_text = self._extract_docling_text(result)
        if not full_text.strip():
            full_text = self._build_fallback_text_from_tables(tables)
        blocks = self._default_blocks_for_text(full_text)
        return ParseResult(
            parser_name="docling",
            parser_version=parser_version,
            config_digest=compute_bytes_digest(json.dumps(config, sort_keys=True).encode("utf-8")),
            tables=tables,
            full_text=full_text,
            blocks=blocks,
            elapsed_seconds=elapsed_seconds,
            page_count=page_count,
            timings=timing_totals,
        )

    def _parse_csv_text(self, file_path: Path, media_type: str) -> ParseResult:
        text = file_path.read_text(encoding="utf-8", errors="replace")
        rows = list(csv.reader(text.splitlines()))
        tables: list[ParsedTable] = []
        if rows:
            table = self._rows_to_parsed_table(rows, page_index=0, caption=file_path.name)
            if table is not None:
                tables.append(table)
        full_text = "\n".join(" | ".join(row) for row in rows) if rows else text
        config = {"profile": self.profile, "mode": "csv", "media_type": media_type}
        blocks = self._default_blocks_for_text(full_text)
        return ParseResult(
            parser_name="csv-parser",
            parser_version="0.1",
            config_digest=compute_bytes_digest(json.dumps(config, sort_keys=True).encode("utf-8")),
            tables=tables,
            full_text=full_text,
            blocks=blocks,
            elapsed_seconds=None,
            page_count=1,
            timings=None,
        )

    def _parse_html_markup(self, file_path: Path, media_type: str) -> ParseResult:
        text = file_path.read_text(encoding="utf-8", errors="replace")
        extractor = _HTMLTableTextExtractor()
        extractor.feed(text)
        tables: list[ParsedTable] = []
        for idx, rows in enumerate(extractor.tables):
            table = self._rows_to_parsed_table(rows, page_index=idx, caption=f"HTML Table {idx + 1}")
            if table is not None:
                tables.append(table)
        full_text = "\n".join(extractor.text_chunks).strip()
        if not full_text:
            full_text = text
        config = {"profile": self.profile, "mode": "html", "media_type": media_type}
        blocks = self._default_blocks_for_text(full_text)
        return ParseResult(
            parser_name="html-parser",
            parser_version="0.1",
            config_digest=compute_bytes_digest(json.dumps(config, sort_keys=True).encode("utf-8")),
            tables=tables,
            full_text=full_text,
            blocks=blocks,
            elapsed_seconds=None,
            page_count=1,
            timings=None,
        )

    def _parse_xml_markup(self, file_path: Path, media_type: str) -> ParseResult:
        root = ET.parse(file_path).getroot()
        text_parts = [" ".join(str(chunk).split()) for chunk in root.itertext() if str(chunk).strip()]
        full_text = "\n".join(text_parts)
        tables = self._extract_xml_tables(root)
        config = {"profile": self.profile, "mode": "xml", "media_type": media_type}
        blocks = self._default_blocks_for_text(full_text)
        return ParseResult(
            parser_name="xml-parser",
            parser_version="0.1",
            config_digest=compute_bytes_digest(json.dumps(config, sort_keys=True).encode("utf-8")),
            tables=tables,
            full_text=full_text,
            blocks=blocks,
            elapsed_seconds=None,
            page_count=1,
            timings=None,
        )

    def _parse_docx(self, file_path: Path) -> ParseResult:
        with zipfile.ZipFile(file_path, "r") as archive:
            doc_xml = archive.read("word/document.xml")
        root = ET.fromstring(doc_xml)
        text_parts = [self._joined_text(node) for node in root.iter() if self._local_tag(node.tag) == "p"]
        full_text = "\n".join(part for part in text_parts if part)
        tables: list[ParsedTable] = []
        table_idx = 0
        for tbl in root.iter():
            if self._local_tag(tbl.tag) != "tbl":
                continue
            rows: list[list[str]] = []
            for tr in tbl:
                if self._local_tag(tr.tag) != "tr":
                    continue
                row: list[str] = []
                for tc in tr:
                    if self._local_tag(tc.tag) != "tc":
                        continue
                    row.append(self._joined_text(tc))
                if any(cell.strip() for cell in row):
                    rows.append(row)
            table = self._rows_to_parsed_table(rows, page_index=table_idx, caption=f"DOCX Table {table_idx + 1}")
            if table is not None:
                tables.append(table)
                table_idx += 1
        config = {"profile": self.profile, "mode": "docx_zipxml"}
        blocks = self._default_blocks_for_text(full_text)
        return ParseResult(
            parser_name="docx-parser",
            parser_version="0.1",
            config_digest=compute_bytes_digest(json.dumps(config, sort_keys=True).encode("utf-8")),
            tables=tables,
            full_text=full_text,
            blocks=blocks,
            elapsed_seconds=None,
            page_count=1,
            timings=None,
        )

    def _parse_xlsx(self, file_path: Path) -> ParseResult:
        with zipfile.ZipFile(file_path, "r") as archive:
            names = archive.namelist()
            shared_strings: list[str] = []
            if "xl/sharedStrings.xml" in names:
                shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
                for si in shared_root.iter():
                    if self._local_tag(si.tag) != "si":
                        continue
                    shared_strings.append(self._joined_text(si))

            sheet_names = sorted(
                name for name in names if name.startswith("xl/worksheets/") and name.endswith(".xml")
            )
            tables: list[ParsedTable] = []
            text_parts: list[str] = []
            for idx, sheet_name in enumerate(sheet_names):
                sheet_root = ET.fromstring(archive.read(sheet_name))
                grid: dict[int, dict[int, str]] = {}
                max_col = 0
                max_row = 0
                row_counter = 0
                for row_node in sheet_root.iter():
                    if self._local_tag(row_node.tag) != "row":
                        continue
                    row_counter += 1
                    logical_row = int(row_node.attrib.get("r", row_counter))
                    row_map = grid.setdefault(logical_row, {})
                    max_row = max(max_row, logical_row)
                    cell_seq = 0
                    for cell in row_node:
                        if self._local_tag(cell.tag) != "c":
                            continue
                        cell_seq += 1
                        ref = cell.attrib.get("r")
                        col_idx = self._cell_ref_to_col_index(ref) if ref else cell_seq
                        col_idx = max(1, col_idx)
                        max_col = max(max_col, col_idx)
                        row_map[col_idx] = self._worksheet_cell_value(cell, shared_strings)
                rows: list[list[str]] = []
                for row_num in range(1, max_row + 1):
                    row_values: list[str] = []
                    row_map = grid.get(row_num, {})
                    for col_num in range(1, max_col + 1):
                        row_values.append(str(row_map.get(col_num, "")))
                    if any(v.strip() for v in row_values):
                        rows.append(row_values)
                        text_parts.append(" | ".join(row_values))
                table = self._rows_to_parsed_table(rows, page_index=idx, caption=sheet_name)
                if table is not None:
                    tables.append(table)
            full_text = "\n".join(text_parts)
        config = {"profile": self.profile, "mode": "xlsx_zipxml"}
        blocks = self._default_blocks_for_text(full_text)
        return ParseResult(
            parser_name="xlsx-parser",
            parser_version="0.1",
            config_digest=compute_bytes_digest(json.dumps(config, sort_keys=True).encode("utf-8")),
            tables=tables,
            full_text=full_text,
            blocks=blocks,
            elapsed_seconds=None,
            page_count=max(1, len(tables)),
            timings=None,
        )

    def _parse_pptx(self, file_path: Path) -> ParseResult:
        with zipfile.ZipFile(file_path, "r") as archive:
            slide_names = sorted(
                name for name in archive.namelist() if name.startswith("ppt/slides/slide") and name.endswith(".xml")
            )
            slides: list[str] = []
            for slide_name in slide_names:
                root = ET.fromstring(archive.read(slide_name))
                parts = [node.text or "" for node in root.iter() if self._local_tag(node.tag) == "t"]
                slide_text = " ".join(part.strip() for part in parts if part and part.strip())
                if slide_text:
                    slides.append(slide_text)
            full_text = "\n\n".join(slides)
        config = {"profile": self.profile, "mode": "pptx_zipxml"}
        blocks = self._default_blocks_for_text(full_text)
        return ParseResult(
            parser_name="pptx-parser",
            parser_version="0.1",
            config_digest=compute_bytes_digest(json.dumps(config, sort_keys=True).encode("utf-8")),
            tables=[],
            full_text=full_text,
            blocks=blocks,
            elapsed_seconds=None,
            page_count=max(1, len(slides)),
            timings=None,
        )

    def _parse_odf(self, file_path: Path, *, suffix: str) -> ParseResult:
        with zipfile.ZipFile(file_path, "r") as archive:
            if "content.xml" not in archive.namelist():
                raise RuntimeError(f"ODF package missing content.xml: {file_path.name}")
            root = ET.fromstring(archive.read("content.xml"))
        text_parts = [" ".join(str(chunk).split()) for chunk in root.itertext() if str(chunk).strip()]
        full_text = "\n".join(text_parts)
        tables = self._extract_xml_tables(root)
        config = {"profile": self.profile, "mode": f"odf_{suffix.lstrip('.')}"}
        blocks = self._default_blocks_for_text(full_text)
        return ParseResult(
            parser_name="odf-parser",
            parser_version="0.1",
            config_digest=compute_bytes_digest(json.dumps(config, sort_keys=True).encode("utf-8")),
            tables=tables,
            full_text=full_text,
            blocks=blocks,
            elapsed_seconds=None,
            page_count=1,
            timings=None,
        )

    def _parse_zip_xml_bundle(self, file_path: Path, *, suffix: str) -> ParseResult:
        with zipfile.ZipFile(file_path, "r") as archive:
            names = sorted(archive.namelist())
            tables: list[ParsedTable] = []
            text_parts: list[str] = []
            page_index = 0
            for name in names:
                lower = name.lower()
                if lower.endswith((".html", ".htm", ".xhtml")):
                    text = archive.read(name).decode("utf-8", errors="replace")
                    extractor = _HTMLTableTextExtractor()
                    extractor.feed(text)
                    if extractor.text_chunks:
                        text_parts.append("\n".join(extractor.text_chunks))
                    for rows in extractor.tables:
                        table = self._rows_to_parsed_table(rows, page_index=page_index, caption=name)
                        if table is not None:
                            tables.append(table)
                            page_index += 1
                    continue
                if not lower.endswith(".xml"):
                    continue
                try:
                    root = ET.fromstring(archive.read(name))
                except Exception:
                    continue
                node_text = self._joined_text(root)
                if node_text:
                    text_parts.append(node_text)
                for table in self._extract_xml_tables(root):
                    table.page_index = page_index
                    if not table.caption:
                        table.caption = name
                    tables.append(table)
                    page_index += 1

        full_text = "\n\n".join(text_parts)
        config = {"profile": self.profile, "mode": f"zip_xml_{suffix.lstrip('.')}"}
        blocks = self._default_blocks_for_text(full_text)
        return ParseResult(
            parser_name="zip-xml-parser",
            parser_version="0.1",
            config_digest=compute_bytes_digest(json.dumps(config, sort_keys=True).encode("utf-8")),
            tables=tables,
            full_text=full_text,
            blocks=blocks,
            elapsed_seconds=None,
            page_count=max(1, len(text_parts)),
            timings=None,
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
        blocks = self._default_blocks_for_text(text)
        return ParseResult(
            parser_name="text-table-parser",
            parser_version="0.1",
            config_digest=compute_bytes_digest(json.dumps(config, sort_keys=True).encode("utf-8")),
            tables=tables,
            full_text=text,
            blocks=blocks,
            elapsed_seconds=None,
            page_count=1,
            timings=None,
        )

    @staticmethod
    def _rows_to_parsed_table(
        rows: list[list[str]],
        *,
        page_index: int,
        caption: str | None,
    ) -> ParsedTable | None:
        normalized_rows = [list(row) for row in rows if any(str(cell).strip() for cell in row)]
        if not normalized_rows:
            return None
        header = [str(cell).strip() for cell in normalized_rows[0]]
        body = [[str(cell).strip() for cell in row] for row in normalized_rows[1:]]
        row_headers = [row[0] if row else "" for row in body]
        cells: list[ParsedCell] = []
        for r_idx, row in enumerate(body):
            for c_idx, value in enumerate(row):
                cells.append(ParsedCell(row_index=r_idx, col_index=c_idx, value=value))
        return ParsedTable(
            page_index=page_index,
            caption=caption,
            row_headers=row_headers,
            col_headers=header,
            cells=cells,
            bbox=None,
        )

    @staticmethod
    def _local_tag(tag: str) -> str:
        if "}" in tag:
            return tag.split("}", 1)[1]
        return tag

    @classmethod
    def _joined_text(cls, node: ET.Element) -> str:
        parts = []
        for chunk in node.itertext():
            compact = " ".join(str(chunk).split())
            if compact:
                parts.append(compact)
        return " ".join(parts).strip()

    @classmethod
    def _extract_xml_tables(cls, root: ET.Element) -> list[ParsedTable]:
        tables: list[ParsedTable] = []
        table_idx = 0
        table_tags = {"table"}
        row_tags = {"tr", "row", "table-row"}
        cell_tags = {"td", "th", "cell", "table-cell", "covered-table-cell", "entry"}
        for candidate in root.iter():
            if cls._local_tag(candidate.tag) not in table_tags:
                continue
            rows: list[list[str]] = []
            for row_node in candidate.iter():
                if cls._local_tag(row_node.tag) not in row_tags:
                    continue
                row: list[str] = []
                for cell_node in row_node:
                    if cls._local_tag(cell_node.tag) not in cell_tags:
                        continue
                    row.append(cls._joined_text(cell_node))
                if any(cell.strip() for cell in row):
                    rows.append(row)
            table = cls._rows_to_parsed_table(rows, page_index=table_idx, caption=f"XML Table {table_idx + 1}")
            if table is not None:
                tables.append(table)
                table_idx += 1
        return tables

    @classmethod
    def _worksheet_cell_value(cls, cell_node: ET.Element, shared_strings: list[str]) -> str:
        cell_type = (cell_node.attrib.get("t") or "").strip().lower()
        if cell_type == "inlineStr":
            for child in cell_node:
                if cls._local_tag(child.tag) == "is":
                    return cls._joined_text(child)
        value_text = ""
        for child in cell_node:
            if cls._local_tag(child.tag) == "v":
                value_text = (child.text or "").strip()
                break
        if cell_type == "s":
            try:
                idx = int(value_text)
                if 0 <= idx < len(shared_strings):
                    return shared_strings[idx]
            except ValueError:
                return value_text
        return value_text

    @staticmethod
    def _cell_ref_to_col_index(cell_ref: str | None) -> int:
        if not cell_ref:
            return 0
        letters = "".join(ch for ch in cell_ref if ch.isalpha()).upper()
        if not letters:
            return 0
        total = 0
        for ch in letters:
            total = total * 26 + (ord(ch) - ord("A") + 1)
        return total

    @staticmethod
    def _extract_docling_text(result: Any) -> str:
        document = getattr(result, "document", None)
        if document is None:
            return ""

        candidates = [
            "export_to_markdown",
            "export_to_text",
            "to_markdown",
            "to_text",
            "export_to_html",
        ]
        for name in candidates:
            fn = getattr(document, name, None)
            if not callable(fn):
                continue
            try:
                value = fn()
            except TypeError:
                continue
            except Exception:
                continue
            if isinstance(value, bytes):
                return value.decode("utf-8", errors="replace")
            if value is not None:
                return str(value)

        direct_text = getattr(document, "text", None)
        if direct_text is not None:
            return str(direct_text)
        return ""

    @staticmethod
    def _build_fallback_text_from_tables(tables: list[ParsedTable]) -> str:
        parts: list[str] = []
        for idx, table in enumerate(tables, start=1):
            parts.append(f"Table {idx}")
            if table.caption:
                parts.append(table.caption)
            if table.col_headers:
                parts.append(" | ".join(table.col_headers))
            by_row: dict[int, list[str]] = {}
            for cell in table.cells:
                by_row.setdefault(cell.row_index, []).append(cell.value)
            for row_idx in sorted(by_row.keys()):
                parts.append(" | ".join(by_row[row_idx]))
            parts.append("")
        return "\n".join(parts).strip()

    @staticmethod
    def _default_blocks_for_text(text: str) -> list[ParsedBlock]:
        if not text:
            return []
        blocks: list[ParsedBlock] = [
            ParsedBlock(
                block_type="document",
                start_offset=0,
                end_offset=len(text),
                page_index=None,
                attrs={"role": "root"},
            )
        ]
        pattern = re.compile(r"\S(?:.*?\S)?(?:(?=\n\s*\n)|\Z)", re.DOTALL)
        for idx, match in enumerate(pattern.finditer(text)):
            start = int(match.start())
            end = int(match.end())
            if end <= start:
                continue
            blocks.append(
                ParsedBlock(
                    block_type="paragraph",
                    start_offset=start,
                    end_offset=end,
                    page_index=None,
                    attrs={"order_index": idx},
                )
            )
        return blocks

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

    @classmethod
    def _extract_table_geometry(cls, table: Any, fallback_index: int) -> tuple[int, dict[str, float] | None]:
        page_index = cls._extract_page_index_hint(table)
        bbox = cls._extract_bbox_hint(table)

        provenance = getattr(table, "prov", None) or getattr(table, "provenance", None) or []
        if isinstance(provenance, (list, tuple)):
            for item in provenance:
                if page_index is None:
                    page_index = cls._extract_page_index_hint(item)
                if bbox is None:
                    bbox = cls._extract_bbox_hint(item)
                    if bbox is None:
                        bbox = cls._extract_bbox_hint(getattr(item, "bbox", None))

        if page_index is None:
            page_index = fallback_index
        return page_index, bbox

    @classmethod
    def _extract_page_index_hint(cls, value: object) -> int | None:
        if value is None:
            return None
        if isinstance(value, dict):
            if "page_index" in value:
                n = cls._as_int(value.get("page_index"))
                return n if n is not None and n >= 0 else None
            if "page_no" in value:
                n = cls._as_int(value.get("page_no"))
                return (n - 1) if n is not None and n >= 1 else None
            if "page" in value:
                n = cls._as_int(value.get("page"))
                return (n - 1) if n is not None and n >= 1 else None
            return None

        direct_idx = cls._as_int(getattr(value, "page_index", None))
        if direct_idx is not None and direct_idx >= 0:
            return direct_idx
        direct_no = cls._as_int(getattr(value, "page_no", None))
        if direct_no is not None and direct_no >= 1:
            return direct_no - 1
        direct_page = cls._as_int(getattr(value, "page", None))
        if direct_page is not None and direct_page >= 1:
            return direct_page - 1
        return None

    @classmethod
    def _extract_bbox_hint(cls, value: object) -> dict[str, float] | None:
        if value is None:
            return None

        if isinstance(value, dict):
            x0 = cls._as_float(value.get("x0", value.get("left", value.get("l"))))
            y0 = cls._as_float(value.get("y0", value.get("top", value.get("t"))))
            x1 = cls._as_float(value.get("x1", value.get("right", value.get("r"))))
            y1 = cls._as_float(value.get("y1", value.get("bottom", value.get("b"))))
            if None not in {x0, y0, x1, y1}:
                if x1 < x0:
                    x0, x1 = x1, x0
                if y1 < y0:
                    y0, y1 = y1, y0
                return {"x0": float(x0), "y0": float(y0), "x1": float(x1), "y1": float(y1)}
            if "bbox" in value:
                return cls._extract_bbox_hint(value.get("bbox"))
            return None

        nested = getattr(value, "bbox", None)
        if nested is not None and nested is not value:
            nested_box = cls._extract_bbox_hint(nested)
            if nested_box is not None:
                return nested_box

        x0 = cls._as_float(getattr(value, "x0", getattr(value, "left", getattr(value, "l", None))))
        y0 = cls._as_float(getattr(value, "y0", getattr(value, "top", getattr(value, "t", None))))
        x1 = cls._as_float(getattr(value, "x1", getattr(value, "right", getattr(value, "r", None))))
        y1 = cls._as_float(getattr(value, "y1", getattr(value, "bottom", getattr(value, "b", None))))
        if None in {x0, y0, x1, y1}:
            return None
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        return {"x0": float(x0), "y0": float(y0), "x1": float(x1), "y1": float(y1)}

    @staticmethod
    def _as_float(value: object) -> float | None:
        try:
            n = float(value)  # type: ignore[arg-type]
        except Exception:
            return None
        return n if n == n else None

    @staticmethod
    def _as_int(value: object) -> int | None:
        try:
            return int(value)  # type: ignore[arg-type]
        except Exception:
            return None

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

    @staticmethod
    def _build_model(model_cls: type[Any], values: dict[str, object]) -> Any:
        model_fields = getattr(model_cls, "model_fields", None)
        if isinstance(model_fields, dict) and model_fields:
            values = {key: value for key, value in values.items() if key in model_fields}
        try:
            return model_cls(**values)
        except Exception:
            model = model_cls()
            for key, value in values.items():
                if hasattr(model, key):
                    setattr(model, key, value)
            return model

    def _resolve_runtime_settings(
        self,
        *,
        threaded_supported: bool,
        system_resources: SystemResources | None = None,
    ) -> ResolvedDoclingRuntime:
        detected = system_resources or self._detect_system_resources()

        if self.runtime_options.auto_tune:
            resolved = self._auto_tuned_runtime(detected=detected, threaded_supported=threaded_supported)
        else:
            resolved = self._conservative_runtime(detected=detected, threaded_supported=threaded_supported)

        if self.runtime_options.use_threaded_pipeline is not None:
            wanted_threaded = bool(self.runtime_options.use_threaded_pipeline)
            if wanted_threaded and not threaded_supported:
                logger.warning(
                    "Docling threaded pipeline requested but not available in this docling version; using standard pipeline."
                )
            resolved.use_threaded_pipeline = wanted_threaded and threaded_supported

        if self.runtime_options.device:
            resolved.device = self.runtime_options.device.strip().lower()
        if self.runtime_options.num_threads is not None:
            resolved.num_threads = self._sanitize_positive_int(
                self.runtime_options.num_threads,
                fallback=resolved.num_threads,
                minimum=1,
            )
        if self.runtime_options.layout_batch_size is not None:
            resolved.layout_batch_size = self._sanitize_positive_int(
                self.runtime_options.layout_batch_size,
                fallback=resolved.layout_batch_size,
                minimum=1,
            )
        if self.runtime_options.ocr_batch_size is not None:
            resolved.ocr_batch_size = self._sanitize_positive_int(
                self.runtime_options.ocr_batch_size,
                fallback=resolved.ocr_batch_size,
                minimum=1,
            )
        if self.runtime_options.table_batch_size is not None:
            resolved.table_batch_size = self._sanitize_positive_int(
                self.runtime_options.table_batch_size,
                fallback=resolved.table_batch_size,
                minimum=1,
            )
        if self.runtime_options.queue_max_size is not None:
            resolved.queue_max_size = self._sanitize_positive_int(
                self.runtime_options.queue_max_size,
                fallback=resolved.queue_max_size,
                minimum=16,
            )

        if self.runtime_options.auto_tune:
            resolved.mode = "auto"
        else:
            resolved.mode = "manual"

        return resolved

    @staticmethod
    def _sanitize_positive_int(value: int, *, fallback: int, minimum: int) -> int:
        if value < minimum:
            return fallback
        return value

    def _auto_tuned_runtime(
        self,
        *,
        detected: SystemResources,
        threaded_supported: bool,
    ) -> ResolvedDoclingRuntime:
        reserve_cores = 1 if detected.cpu_cores <= 4 else 2
        num_threads = max(1, min(32, detected.cpu_cores - reserve_cores))

        is_apple_silicon = (
            detected.platform_name == "darwin" and detected.machine in {"arm64", "aarch64"}
        )
        device = "mps" if is_apple_silicon else "auto"
        gpu_accel = device.startswith(("mps", "cuda", "xpu"))

        if detected.memory_gb >= 48:
            layout_batch_size = 48 if gpu_accel else 16
            ocr_batch_size = 8 if gpu_accel else 4
            table_batch_size = 8 if gpu_accel else 4
        elif detected.memory_gb >= 24:
            layout_batch_size = 32 if gpu_accel else 12
            ocr_batch_size = 6 if gpu_accel else 4
            table_batch_size = 6 if gpu_accel else 4
        elif detected.memory_gb >= 16:
            layout_batch_size = 20 if gpu_accel else 8
            ocr_batch_size = 4
            table_batch_size = 4
        else:
            layout_batch_size = 10
            ocr_batch_size = 2
            table_batch_size = 2

        # Keep layout batches high enough for throughput but proportional to thread budget.
        layout_batch_size = min(layout_batch_size, max(8, num_threads * 4))
        ocr_batch_size = min(ocr_batch_size, max(2, num_threads // 2))
        table_batch_size = min(table_batch_size, max(2, num_threads // 2))
        queue_max_size = max(100, min(400, (layout_batch_size + ocr_batch_size + table_batch_size) * 4))

        use_threaded_pipeline = threaded_supported and detected.cpu_cores >= 8 and detected.memory_gb >= 16

        return ResolvedDoclingRuntime(
            mode="auto",
            use_threaded_pipeline=use_threaded_pipeline,
            device=device,
            num_threads=num_threads,
            layout_batch_size=layout_batch_size,
            ocr_batch_size=ocr_batch_size,
            table_batch_size=table_batch_size,
            queue_max_size=queue_max_size,
            cpu_cores=detected.cpu_cores,
            memory_gb=detected.memory_gb,
            platform_name=detected.platform_name,
            machine=detected.machine,
        )

    def _conservative_runtime(
        self,
        *,
        detected: SystemResources,
        threaded_supported: bool,
    ) -> ResolvedDoclingRuntime:
        return ResolvedDoclingRuntime(
            mode="manual",
            use_threaded_pipeline=False if threaded_supported else False,
            device="auto",
            num_threads=max(1, min(4, detected.cpu_cores)),
            layout_batch_size=4,
            ocr_batch_size=4,
            table_batch_size=4,
            queue_max_size=100,
            cpu_cores=detected.cpu_cores,
            memory_gb=detected.memory_gb,
            platform_name=detected.platform_name,
            machine=detected.machine,
        )

    def _detect_system_resources(self) -> SystemResources:
        cpu_cores = max(1, os.cpu_count() or 1)
        memory_gb = self._detect_total_memory_gb()
        platform_name = platform.system().lower()
        machine = platform.machine().lower()
        return SystemResources(
            cpu_cores=cpu_cores,
            memory_gb=max(memory_gb, 1),
            platform_name=platform_name,
            machine=machine,
        )

    @staticmethod
    def _detect_total_memory_gb() -> int:
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
            pages = os.sysconf("SC_PHYS_PAGES")
            if isinstance(page_size, int) and isinstance(pages, int) and page_size > 0 and pages > 0:
                return int((page_size * pages) / (1024**3))
        except Exception:
            pass

        if platform.system().lower() == "darwin":
            try:
                output = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
                total_bytes = int(output)
                if total_bytes > 0:
                    return int(total_bytes / (1024**3))
            except Exception:
                pass

        return 8

    def _log_runtime_settings(
        self,
        *,
        file_path: Path,
        runtime: ResolvedDoclingRuntime,
        threaded_supported: bool,
    ) -> None:
        if not self.runtime_options.log_settings:
            return
        print(
            "[docling] "
            f"file={file_path.name} "
            f"profile={self.profile} "
            f"mode={runtime.mode} "
            f"threaded={runtime.use_threaded_pipeline} "
            f"threaded_supported={threaded_supported} "
            f"device={runtime.device} "
            f"threads={runtime.num_threads} "
            f"layout_batch={runtime.layout_batch_size} "
            f"ocr_batch={runtime.ocr_batch_size} "
            f"table_batch={runtime.table_batch_size} "
            f"queue_max={runtime.queue_max_size} "
            f"cpu_cores={runtime.cpu_cores} "
            f"memory_gb={runtime.memory_gb}"
        )

    @staticmethod
    def _extract_page_count(result: Any) -> int:
        input_obj = getattr(result, "input", None)
        input_pages = getattr(input_obj, "page_count", None)
        if isinstance(input_pages, int) and input_pages > 0:
            return input_pages
        pages = getattr(result, "pages", None)
        if isinstance(pages, list) and pages:
            return len(pages)
        return 0

    @staticmethod
    def _extract_timing_totals(result: Any) -> dict[str, float]:
        totals: dict[str, float] = {}
        timing_map = getattr(result, "timings", None)
        if not isinstance(timing_map, dict):
            return totals
        for key, item in timing_map.items():
            times = getattr(item, "times", None)
            if isinstance(times, list) and times:
                numeric = [float(x) for x in times if isinstance(x, (int, float))]
                if numeric:
                    totals[str(key)] = sum(numeric)
        return totals

    @staticmethod
    def _timing_summary(totals: dict[str, float], limit: int = 4) -> str:
        if not totals:
            return ""
        ordered = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)
        return ",".join(f"{name}:{seconds:.2f}s" for name, seconds in ordered[:limit])
