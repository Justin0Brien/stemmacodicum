from pathlib import Path
import zipfile

from stemmacodicum.infrastructure.parsers.docling_adapter import (
    DoclingAdapter,
    DoclingRuntimeOptions,
    SystemResources,
)


def test_text_parser_extracts_markdown_table(tmp_path: Path) -> None:
    source = tmp_path / "table.md"
    source.write_text(
        """
Table 1: Spend by category

| Category | 2024/25 | 2023/24 |
|---|---:|---:|
| Cash at bank | 5631 | 5500 |
| Debt | 120 | 130 |
""",
        encoding="utf-8",
    )

    result = DoclingAdapter().parse_resource(source, "text/markdown")

    assert result.parser_name == "text-table-parser"
    assert len(result.tables) == 1
    assert "Cash at bank" in result.full_text
    assert any(block.block_type == "document" for block in result.blocks)

    table = result.tables[0]
    assert table.caption == "Table 1: Spend by category"
    assert table.col_headers == ["Category", "2024/25", "2023/24"]
    assert table.row_headers == ["Cash at bank", "Debt"]


def test_auto_tune_prefers_mps_on_apple_silicon() -> None:
    adapter = DoclingAdapter()
    resolved = adapter._resolve_runtime_settings(
        threaded_supported=True,
        system_resources=SystemResources(
            cpu_cores=12,
            memory_gb=64,
            platform_name="darwin",
            machine="arm64",
        ),
    )

    assert resolved.mode == "auto"
    assert resolved.use_threaded_pipeline is True
    assert resolved.device == "mps"
    assert resolved.num_threads == 10
    assert resolved.layout_batch_size >= 32
    assert resolved.queue_max_size >= 100


def test_runtime_overrides_take_priority() -> None:
    adapter = DoclingAdapter(
        runtime_options=DoclingRuntimeOptions(
            auto_tune=True,
            use_threaded_pipeline=False,
            device="cpu",
            num_threads=6,
            layout_batch_size=11,
            ocr_batch_size=3,
            table_batch_size=5,
            queue_max_size=77,
        )
    )
    resolved = adapter._resolve_runtime_settings(
        threaded_supported=True,
        system_resources=SystemResources(
            cpu_cores=16,
            memory_gb=64,
            platform_name="linux",
            machine="x86_64",
        ),
    )

    assert resolved.use_threaded_pipeline is False
    assert resolved.device == "cpu"
    assert resolved.num_threads == 6
    assert resolved.layout_batch_size == 11
    assert resolved.ocr_batch_size == 3
    assert resolved.table_batch_size == 5
    assert resolved.queue_max_size == 77


def test_threaded_override_falls_back_when_not_supported() -> None:
    adapter = DoclingAdapter(
        runtime_options=DoclingRuntimeOptions(
            use_threaded_pipeline=True,
        )
    )
    resolved = adapter._resolve_runtime_settings(
        threaded_supported=False,
        system_resources=SystemResources(
            cpu_cores=12,
            memory_gb=32,
            platform_name="linux",
            machine="x86_64",
        ),
    )

    assert resolved.use_threaded_pipeline is False


def test_html_parser_extracts_text_and_table(tmp_path: Path) -> None:
    source = tmp_path / "report.html"
    source.write_text(
        """
<html><body>
  <h1>Annual Summary</h1>
  <p>Liquidity improved during FY2025.</p>
  <table>
    <tr><th>Metric</th><th>Value</th></tr>
    <tr><td>Cash</td><td>5631</td></tr>
  </table>
</body></html>
""",
        encoding="utf-8",
    )

    result = DoclingAdapter().parse_resource(source, "text/html")

    assert result.parser_name == "html-parser"
    assert "Annual Summary" in result.full_text
    assert len(result.tables) == 1
    assert result.tables[0].col_headers == ["Metric", "Value"]


def test_docx_zip_parser_extracts_text_and_table(tmp_path: Path) -> None:
    source = tmp_path / "report.docx"
    doc_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:r><w:t>Statement for FY2025</w:t></w:r></w:p>
    <w:tbl>
      <w:tr>
        <w:tc><w:p><w:r><w:t>Metric</w:t></w:r></w:p></w:tc>
        <w:tc><w:p><w:r><w:t>Value</w:t></w:r></w:p></w:tc>
      </w:tr>
      <w:tr>
        <w:tc><w:p><w:r><w:t>Cash</w:t></w:r></w:p></w:tc>
        <w:tc><w:p><w:r><w:t>5631</w:t></w:r></w:p></w:tc>
      </w:tr>
    </w:tbl>
  </w:body>
</w:document>
"""
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("word/document.xml", doc_xml)

    result = DoclingAdapter().parse_resource(
        source,
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )

    assert result.parser_name == "docx-parser"
    assert "Statement for FY2025" in result.full_text
    assert len(result.tables) == 1
    assert result.tables[0].col_headers == ["Metric", "Value"]


def test_xlsx_zip_parser_extracts_table(tmp_path: Path) -> None:
    source = tmp_path / "report.xlsx"
    shared_strings = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" count="2" uniqueCount="2">
  <si><t>Metric</t></si>
  <si><t>Cash</t></si>
</sst>
"""
    sheet_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <sheetData>
    <row r="1">
      <c r="A1" t="s"><v>0</v></c>
      <c r="B1"><v>Value</v></c>
    </row>
    <row r="2">
      <c r="A2" t="s"><v>1</v></c>
      <c r="B2"><v>5631</v></c>
    </row>
  </sheetData>
</worksheet>
"""
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("xl/sharedStrings.xml", shared_strings)
        archive.writestr("xl/worksheets/sheet1.xml", sheet_xml)

    result = DoclingAdapter().parse_resource(
        source,
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    assert result.parser_name == "xlsx-parser"
    assert len(result.tables) == 1
    assert result.tables[0].col_headers == ["Metric", "Value"]
    assert any(cell.value == "5631" for cell in result.tables[0].cells)
