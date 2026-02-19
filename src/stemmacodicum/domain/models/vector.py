from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class VectorIndexRun:
    id: str
    resource_id: str
    extraction_run_id: str
    vector_backend: str
    collection_name: str
    embedding_model: str
    embedding_dim: int | None
    chunking_version: str
    status: str
    chunks_total: int
    chunks_indexed: int
    error_message: str | None
    created_at: str
    finished_at: str | None


@dataclass(slots=True)
class VectorChunk:
    id: str
    vector_index_run_id: str
    resource_id: str
    extraction_run_id: str
    chunk_id: str
    vector_point_id: str
    source_type: str
    source_ref: str | None
    page_index: int | None
    start_offset: int | None
    end_offset: int | None
    token_count_est: int | None
    embedding_dim: int
    vector_backend: str
    collection_name: str
    text_digest_sha256: str
    text_content: str
    created_at: str
