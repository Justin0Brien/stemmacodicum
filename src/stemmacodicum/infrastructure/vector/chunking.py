from __future__ import annotations

import json
from dataclasses import dataclass

from stemmacodicum.core.hashing import compute_bytes_digest
from stemmacodicum.core.ids import deterministic_uuid
from stemmacodicum.domain.models.extraction import DocumentText, ExtractedTable, TextSegment

DEFAULT_CHUNKING_VERSION = "text-sentence-v2"


@dataclass(slots=True)
class ChunkDraft:
    chunk_id: str
    vector_point_id: str
    source_type: str
    source_ref: str | None
    page_index: int | None
    start_offset: int | None
    end_offset: int | None
    token_count_est: int
    text_digest_sha256: str
    text_content: str


class VectorChunker:
    def __init__(
        self,
        *,
        chunking_version: str = DEFAULT_CHUNKING_VERSION,
        target_chars: int = 1100,
        overlap_chars: int = 180,
        max_chunk_chars: int | None = None,
    ) -> None:
        self.chunking_version = chunking_version
        self.target_chars = max(300, target_chars)
        self.overlap_chars = max(0, min(overlap_chars, self.target_chars // 2))
        self.max_chunk_chars = max_chunk_chars or int(self.target_chars * 1.4)

    def build_chunks(
        self,
        *,
        resource_id: str,
        extraction_run_id: str,
        document_text: DocumentText | None,
        segments: list[TextSegment],
        tables: list[ExtractedTable],
    ) -> list[ChunkDraft]:
        out: list[ChunkDraft] = []
        seen_chunk_ids: set[str] = set()

        text_chunks = self._text_chunks(
            resource_id=resource_id,
            extraction_run_id=extraction_run_id,
            document_text=document_text,
            segments=segments,
        )
        for chunk in text_chunks:
            if chunk.chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk.chunk_id)
            out.append(chunk)

        table_chunks = self._table_chunks(
            resource_id=resource_id,
            extraction_run_id=extraction_run_id,
            tables=tables,
        )
        for chunk in table_chunks:
            if chunk.chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk.chunk_id)
            out.append(chunk)

        return out

    def _text_chunks(
        self,
        *,
        resource_id: str,
        extraction_run_id: str,
        document_text: DocumentText | None,
        segments: list[TextSegment],
    ) -> list[ChunkDraft]:
        if document_text is None:
            return []
        text = document_text.text_content or ""
        if not text.strip():
            return []

        sentence_segments = sorted(
            [s for s in segments if s.segment_type == "structure:sentence"],
            key=lambda s: (s.start_offset, s.end_offset),
        )

        windows: list[tuple[int, int]] = []
        if sentence_segments:
            idx = 0
            count = len(sentence_segments)
            while idx < count:
                start = sentence_segments[idx].start_offset
                end = sentence_segments[idx].end_offset
                j = idx + 1
                while j < count:
                    next_end = sentence_segments[j].end_offset
                    if next_end - start > self.target_chars:
                        break
                    end = next_end
                    j += 1
                windows.append((start, end))

                if j >= count:
                    break

                next_idx = j
                while next_idx > idx + 1:
                    candidate = sentence_segments[next_idx - 1]
                    if end - candidate.start_offset >= self.overlap_chars:
                        next_idx -= 1
                        continue
                    break
                if next_idx == idx:
                    next_idx = j
                idx = next_idx
        else:
            step = max(1, self.target_chars - self.overlap_chars)
            start = 0
            while start < len(text):
                end = min(start + self.target_chars, len(text))
                windows.append((start, end))
                if end >= len(text):
                    break
                start += step

        chunks: list[ChunkDraft] = []
        windows = self._enforce_max_window_size(windows, len(text))
        for order, (start, end) in enumerate(windows):
            window_text = text[start:end].strip()
            if not window_text:
                continue
            source_ref = f"sentence-window:{order}"
            chunks.append(
                self._build_chunk(
                    resource_id=resource_id,
                    extraction_run_id=extraction_run_id,
                    source_type="text_window",
                    source_ref=source_ref,
                    page_index=None,
                    start_offset=start,
                    end_offset=end,
                    text_content=window_text,
                )
            )
        return chunks

    def _enforce_max_window_size(
        self,
        windows: list[tuple[int, int]],
        text_len: int,
    ) -> list[tuple[int, int]]:
        if not windows:
            return windows
        out: list[tuple[int, int]] = []
        step = max(1, self.target_chars - self.overlap_chars)
        hard_max = max(self.target_chars, self.max_chunk_chars)
        for start, end in windows:
            s = max(0, min(start, text_len))
            e = max(0, min(end, text_len))
            if e <= s:
                continue
            if (e - s) <= hard_max:
                out.append((s, e))
                continue
            cursor = s
            while cursor < e:
                chunk_end = min(cursor + self.target_chars, e)
                if chunk_end <= cursor:
                    break
                out.append((cursor, chunk_end))
                if chunk_end >= e:
                    break
                cursor += step
        return out

    def _table_chunks(
        self,
        *,
        resource_id: str,
        extraction_run_id: str,
        tables: list[ExtractedTable],
    ) -> list[ChunkDraft]:
        chunks: list[ChunkDraft] = []
        for table in tables:
            try:
                col_headers = json.loads(table.col_headers_json)
            except Exception:
                col_headers = []
            try:
                row_headers = json.loads(table.row_headers_json)
            except Exception:
                row_headers = []
            try:
                cells = json.loads(table.cells_json)
            except Exception:
                cells = []

            grid: dict[int, dict[int, str]] = {}
            max_col = -1
            for cell in cells:
                try:
                    row_idx = int(cell.get("row_index", -1))
                    col_idx = int(cell.get("col_index", -1))
                except Exception:
                    continue
                value = str(cell.get("value") or "").strip()
                if row_idx < 0 or col_idx < 0 or not value:
                    continue
                grid.setdefault(row_idx, {})[col_idx] = value
                max_col = max(max_col, col_idx)

            for row_idx in sorted(grid.keys()):
                row_bits: list[str] = []
                for col_idx in range(max_col + 1):
                    value = grid[row_idx].get(col_idx)
                    if not value:
                        continue
                    header = (
                        str(col_headers[col_idx]).strip()
                        if col_idx < len(col_headers) and str(col_headers[col_idx]).strip()
                        else f"col_{col_idx}"
                    )
                    row_bits.append(f"{header}: {value}")

                row_header = (
                    str(row_headers[row_idx]).strip()
                    if row_idx < len(row_headers) and str(row_headers[row_idx]).strip()
                    else ""
                )
                bits = []
                if table.caption:
                    bits.append(f"Table: {table.caption.strip()}")
                bits.append(f"Table ID: {table.table_id}")
                if row_header:
                    bits.append(f"Row: {row_header}")
                bits.extend(row_bits)
                row_text = "\n".join(bits).strip()
                if not row_text:
                    continue

                parts = self._split_text_with_overlap(row_text)
                for part_idx, part_text in enumerate(parts):
                    source_ref = f"{table.table_id}:row:{row_idx}"
                    if len(parts) > 1:
                        source_ref = f"{source_ref}:part:{part_idx}"
                    chunks.append(
                        self._build_chunk(
                            resource_id=resource_id,
                            extraction_run_id=extraction_run_id,
                            source_type="table_row",
                            source_ref=source_ref,
                            page_index=table.page_index,
                            start_offset=None,
                            end_offset=None,
                            text_content=part_text,
                        )
                    )

            if not grid:
                summary = []
                if table.caption:
                    summary.append(f"Table: {table.caption.strip()}")
                summary.append(f"Table ID: {table.table_id}")
                if col_headers:
                    summary.append("Columns: " + ", ".join(str(x) for x in col_headers if str(x).strip()))
                summary_text = "\n".join(summary).strip()
                if summary_text:
                    parts = self._split_text_with_overlap(summary_text)
                    for part_idx, part_text in enumerate(parts):
                        source_ref = table.table_id
                        if len(parts) > 1:
                            source_ref = f"{source_ref}:part:{part_idx}"
                        chunks.append(
                            self._build_chunk(
                                resource_id=resource_id,
                                extraction_run_id=extraction_run_id,
                                source_type="table_summary",
                                source_ref=source_ref,
                                page_index=table.page_index,
                                start_offset=None,
                                end_offset=None,
                                text_content=part_text,
                            )
                        )
        return chunks

    def _split_text_with_overlap(self, text: str) -> list[str]:
        normalized = text.strip()
        if not normalized:
            return []
        hard_max = max(self.target_chars, self.max_chunk_chars)
        if len(normalized) <= hard_max:
            return [normalized]
        parts: list[str] = []
        step = max(1, self.target_chars - self.overlap_chars)
        cursor = 0
        while cursor < len(normalized):
            end = min(cursor + self.target_chars, len(normalized))
            part = normalized[cursor:end].strip()
            if part:
                parts.append(part)
            if end >= len(normalized):
                break
            cursor += step
        return parts

    def _build_chunk(
        self,
        *,
        resource_id: str,
        extraction_run_id: str,
        source_type: str,
        source_ref: str | None,
        page_index: int | None,
        start_offset: int | None,
        end_offset: int | None,
        text_content: str,
    ) -> ChunkDraft:
        normalized_text = text_content.strip()
        text_digest = compute_bytes_digest(normalized_text.encode("utf-8"))
        identity = {
            "resource_id": resource_id,
            "extraction_run_id": extraction_run_id,
            "chunking_version": self.chunking_version,
            "source_type": source_type,
            "source_ref": source_ref,
            "page_index": page_index,
            "start_offset": start_offset,
            "end_offset": end_offset,
            "text_digest": text_digest,
        }
        raw_id = compute_bytes_digest(json.dumps(identity, sort_keys=True).encode("utf-8"))
        return ChunkDraft(
            chunk_id=f"sha256:{raw_id}",
            vector_point_id=deterministic_uuid(raw_id),
            source_type=source_type,
            source_ref=source_ref,
            page_index=page_index,
            start_offset=start_offset,
            end_offset=end_offset,
            token_count_est=max(1, len(normalized_text) // 4),
            text_digest_sha256=text_digest,
            text_content=normalized_text,
        )
