from pathlib import Path

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
