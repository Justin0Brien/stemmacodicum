from pathlib import Path

from stemmacodicum.infrastructure.parsers.docling_adapter import DoclingAdapter


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

    table = result.tables[0]
    assert table.caption == "Table 1: Spend by category"
    assert table.col_headers == ["Category", "2024/25", "2023/24"]
    assert table.row_headers == ["Cash at bank", "Debt"]
