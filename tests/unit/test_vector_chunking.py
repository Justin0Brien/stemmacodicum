from __future__ import annotations

from stemmacodicum.domain.models.extraction import DocumentText, ExtractedTable, TextSegment
from stemmacodicum.infrastructure.vector.chunking import VectorChunker


def test_chunker_builds_text_and_table_chunks() -> None:
    text = "Cash at bank increased in FY2025. Revenue also improved. Liquidity remained strong."
    document_text = DocumentText(
        id="doc-1",
        extraction_run_id="run-1",
        resource_id="res-1",
        text_content=text,
        text_digest_sha256="d",
        char_count=len(text),
        created_at="2026-01-01T00:00:00Z",
    )
    segments = [
        TextSegment(
            id="s1",
            document_text_id=document_text.id,
            extraction_run_id=document_text.extraction_run_id,
            resource_id=document_text.resource_id,
            segment_type="structure:sentence",
            start_offset=0,
            end_offset=35,
            page_index=None,
            order_index=0,
            bbox_json=None,
            attrs_json=None,
            created_at="2026-01-01T00:00:00Z",
        ),
        TextSegment(
            id="s2",
            document_text_id=document_text.id,
            extraction_run_id=document_text.extraction_run_id,
            resource_id=document_text.resource_id,
            segment_type="structure:sentence",
            start_offset=36,
            end_offset=59,
            page_index=None,
            order_index=1,
            bbox_json=None,
            attrs_json=None,
            created_at="2026-01-01T00:00:00Z",
        ),
    ]
    table = ExtractedTable(
        id="t1",
        extraction_run_id="run-1",
        resource_id="res-1",
        table_id="sha256:table1",
        page_index=2,
        caption="Statement of financial position",
        row_headers_json='["Cash"]',
        col_headers_json='["Item","Value"]',
        cells_json='[{"row_index":0,"col_index":0,"value":"Cash"},{"row_index":0,"col_index":1,"value":"5631"}]',
        bbox_json=None,
        created_at="2026-01-01T00:00:00Z",
    )

    chunker = VectorChunker(target_chars=120, overlap_chars=20)
    chunks = chunker.build_chunks(
        resource_id="res-1",
        extraction_run_id="run-1",
        document_text=document_text,
        segments=segments,
        tables=[table],
    )

    assert any(chunk.source_type == "text_window" for chunk in chunks)
    assert any(chunk.source_type == "table_row" for chunk in chunks)
    assert all(chunk.chunk_id.startswith("sha256:") for chunk in chunks)


def test_chunker_enforces_max_window_size() -> None:
    text = "A" * 5000
    document_text = DocumentText(
        id="doc-2",
        extraction_run_id="run-2",
        resource_id="res-2",
        text_content=text,
        text_digest_sha256="d",
        char_count=len(text),
        created_at="2026-01-01T00:00:00Z",
    )
    segments = [
        TextSegment(
            id="s1",
            document_text_id=document_text.id,
            extraction_run_id=document_text.extraction_run_id,
            resource_id=document_text.resource_id,
            segment_type="structure:sentence",
            start_offset=0,
            end_offset=5000,
            page_index=None,
            order_index=0,
            bbox_json=None,
            attrs_json=None,
            created_at="2026-01-01T00:00:00Z",
        )
    ]
    chunker = VectorChunker(target_chars=900, overlap_chars=100, max_chunk_chars=1200)
    chunks = chunker.build_chunks(
        resource_id="res-2",
        extraction_run_id="run-2",
        document_text=document_text,
        segments=segments,
        tables=[],
    )
    assert len(chunks) > 1
    assert all(len(c.text_content) <= 1200 for c in chunks if c.source_type == "text_window")


def test_chunker_splits_large_table_rows() -> None:
    giant_value = "X" * 5000
    table = ExtractedTable(
        id="t2",
        extraction_run_id="run-3",
        resource_id="res-3",
        table_id="sha256:table2",
        page_index=1,
        caption="Large table",
        row_headers_json='["Row"]',
        col_headers_json='["ColA","ColB"]',
        cells_json=f'[{{"row_index":0,"col_index":0,"value":"Row"}},{{"row_index":0,"col_index":1,"value":"{giant_value}"}}]',
        bbox_json=None,
        created_at="2026-01-01T00:00:00Z",
    )
    chunker = VectorChunker(target_chars=900, overlap_chars=100, max_chunk_chars=1200)
    chunks = chunker.build_chunks(
        resource_id="res-3",
        extraction_run_id="run-3",
        document_text=None,
        segments=[],
        tables=[table],
    )
    table_chunks = [c for c in chunks if c.source_type == "table_row"]
    assert len(table_chunks) > 1
    assert all(len(c.text_content) <= 1200 for c in table_chunks)
