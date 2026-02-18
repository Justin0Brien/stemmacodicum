from __future__ import annotations

import json
import logging
import os
import platform
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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


class DoclingAdapter:
    """
    Parser adapter with two modes:
    - Markdown/plaintext table extraction (always available).
    - PDF extraction via docling (optional, if installed).
    """

    def __init__(
        self,
        profile: str = "default",
        runtime_options: DoclingRuntimeOptions | None = None,
    ) -> None:
        self.profile = profile
        self.runtime_options = runtime_options or DoclingRuntimeOptions()

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
