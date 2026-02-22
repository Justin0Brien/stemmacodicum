from __future__ import annotations

from pathlib import Path

from stemmacodicum.domain.models.vector import VectorChunk, VectorIndexRun
from stemmacodicum.infrastructure.db.sqlite import get_connection


class VectorRepo:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def insert_run(self, run: VectorIndexRun) -> None:
        with get_connection(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO vector_index_runs (
                    id,
                    resource_id,
                    extraction_run_id,
                    vector_backend,
                    collection_name,
                    embedding_model,
                    embedding_dim,
                    chunking_version,
                    status,
                    chunks_total,
                    chunks_indexed,
                    error_message,
                    created_at,
                    finished_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run.id,
                    run.resource_id,
                    run.extraction_run_id,
                    run.vector_backend,
                    run.collection_name,
                    run.embedding_model,
                    run.embedding_dim,
                    run.chunking_version,
                    run.status,
                    run.chunks_total,
                    run.chunks_indexed,
                    run.error_message,
                    run.created_at,
                    run.finished_at,
                ),
            )
            conn.commit()

    def finalize_run(
        self,
        run_id: str,
        *,
        status: str,
        chunks_total: int,
        chunks_indexed: int,
        embedding_dim: int | None,
        error_message: str | None,
        finished_at: str,
    ) -> None:
        with get_connection(self.db_path) as conn:
            conn.execute(
                """
                UPDATE vector_index_runs
                SET status = ?,
                    chunks_total = ?,
                    chunks_indexed = ?,
                    embedding_dim = COALESCE(?, embedding_dim),
                    error_message = ?,
                    finished_at = ?
                WHERE id = ?
                """,
                (
                    status,
                    chunks_total,
                    chunks_indexed,
                    embedding_dim,
                    error_message,
                    finished_at,
                    run_id,
                ),
            )
            conn.commit()

    def insert_chunks(self, chunks: list[VectorChunk]) -> int:
        if not chunks:
            return 0
        with get_connection(self.db_path) as conn:
            conn.executemany(
                """
                INSERT INTO vector_chunks (
                    id,
                    vector_index_run_id,
                    resource_id,
                    extraction_run_id,
                    chunk_id,
                    vector_point_id,
                    source_type,
                    source_ref,
                    page_index,
                    start_offset,
                    end_offset,
                    token_count_est,
                    embedding_dim,
                    vector_backend,
                    collection_name,
                    text_digest_sha256,
                    text_content,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        chunk.id,
                        chunk.vector_index_run_id,
                        chunk.resource_id,
                        chunk.extraction_run_id,
                        chunk.chunk_id,
                        chunk.vector_point_id,
                        chunk.source_type,
                        chunk.source_ref,
                        chunk.page_index,
                        chunk.start_offset,
                        chunk.end_offset,
                        chunk.token_count_est,
                        chunk.embedding_dim,
                        chunk.vector_backend,
                        chunk.collection_name,
                        chunk.text_digest_sha256,
                        chunk.text_content,
                        chunk.created_at,
                    )
                    for chunk in chunks
                ],
            )
            conn.commit()
        return len(chunks)

    def get_latest_run_for_extraction(
        self,
        extraction_run_id: str,
        *,
        vector_backend: str,
        embedding_model: str,
        chunking_version: str,
        collection_name: str,
    ) -> VectorIndexRun | None:
        with get_connection(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT *
                FROM vector_index_runs
                WHERE extraction_run_id = ?
                  AND vector_backend = ?
                  AND embedding_model = ?
                  AND chunking_version = ?
                  AND collection_name = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (
                    extraction_run_id,
                    vector_backend,
                    embedding_model,
                    chunking_version,
                    collection_name,
                ),
            ).fetchone()
        return self._to_run(row) if row else None

    def list_recent_runs(self, limit: int = 100) -> list[VectorIndexRun]:
        with get_connection(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM vector_index_runs
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._to_run(row) for row in rows]

    def list_recent_runs_for_resource(self, resource_id: str, limit: int = 50) -> list[VectorIndexRun]:
        with get_connection(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM vector_index_runs
                WHERE resource_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (resource_id, limit),
            ).fetchall()
        return [self._to_run(row) for row in rows]

    def list_runs_missing_success_for_latest_extraction(
        self,
        *,
        vector_backend: str,
        embedding_model: str,
        chunking_version: str,
        collection_name: str,
        limit: int = 1000,
    ) -> list[dict[str, str]]:
        with get_connection(self.db_path) as conn:
            rows = conn.execute(
                """
                WITH latest_extract AS (
                    SELECT er.resource_id, er.id AS extraction_run_id, er.created_at
                    FROM extraction_runs er
                    INNER JOIN (
                        SELECT resource_id, MAX(created_at) AS max_created
                        FROM extraction_runs
                        GROUP BY resource_id
                    ) latest
                    ON latest.resource_id = er.resource_id
                   AND latest.max_created = er.created_at
                )
                SELECT le.resource_id, le.extraction_run_id
                FROM latest_extract le
                LEFT JOIN vector_index_runs vr
                  ON vr.extraction_run_id = le.extraction_run_id
                 AND vr.status = 'success'
                 AND vr.vector_backend = ?
                 AND vr.embedding_model = ?
                 AND vr.chunking_version = ?
                 AND vr.collection_name = ?
                WHERE vr.id IS NULL
                ORDER BY le.created_at DESC
                LIMIT ?
                """,
                (
                    vector_backend,
                    embedding_model,
                    chunking_version,
                    collection_name,
                    limit,
                ),
            ).fetchall()
        return [
            {
                "resource_id": str(row["resource_id"]),
                "extraction_run_id": str(row["extraction_run_id"]),
            }
            for row in rows
        ]

    def count_chunks(self) -> int:
        with get_connection(self.db_path) as conn:
            row = conn.execute("SELECT COUNT(*) AS c FROM vector_chunks").fetchone()
        return int(row["c"]) if row else 0

    def count_distinct_chunk_ids(self) -> int:
        with get_connection(self.db_path) as conn:
            row = conn.execute("SELECT COUNT(DISTINCT chunk_id) AS c FROM vector_chunks").fetchone()
        return int(row["c"]) if row else 0

    def list_resources_with_vector_chunks(
        self,
        *,
        media_types: list[str],
        min_size_bytes: int = 0,
        limit: int = 100000,
    ) -> list[dict[str, object]]:
        if not media_types:
            return []
        placeholders = ",".join("?" for _ in media_types)
        with get_connection(self.db_path) as conn:
            rows = conn.execute(
                f"""
                SELECT
                    r.id AS resource_id,
                    r.media_type AS media_type,
                    r.original_filename AS original_filename,
                    r.size_bytes AS size_bytes,
                    COUNT(vc.id) AS chunk_rows
                FROM resources r
                JOIN vector_chunks vc ON vc.resource_id = r.id
                WHERE r.media_type IN ({placeholders})
                  AND r.size_bytes >= ?
                GROUP BY r.id, r.media_type, r.original_filename, r.size_bytes
                ORDER BY r.size_bytes DESC
                LIMIT ?
                """,
                (*media_types, max(0, int(min_size_bytes)), max(1, int(limit))),
            ).fetchall()
        return [
            {
                "resource_id": str(row["resource_id"]),
                "media_type": str(row["media_type"]),
                "original_filename": str(row["original_filename"]),
                "size_bytes": int(row["size_bytes"] or 0),
                "chunk_rows": int(row["chunk_rows"] or 0),
            }
            for row in rows
        ]

    def mark_runs_pruned(self, *, resource_ids: list[str], reason: str) -> int:
        if not resource_ids:
            return 0
        placeholders = ",".join("?" for _ in resource_ids)
        with get_connection(self.db_path) as conn:
            cursor = conn.execute(
                f"""
                UPDATE vector_index_runs
                SET status = 'pruned',
                    error_message = ?
                WHERE resource_id IN ({placeholders})
                """,
                (reason, *resource_ids),
            )
            conn.commit()
        return int(cursor.rowcount or 0)

    @staticmethod
    def _to_run(row) -> VectorIndexRun:
        return VectorIndexRun(
            id=row["id"],
            resource_id=row["resource_id"],
            extraction_run_id=row["extraction_run_id"],
            vector_backend=row["vector_backend"],
            collection_name=row["collection_name"],
            embedding_model=row["embedding_model"],
            embedding_dim=row["embedding_dim"],
            chunking_version=row["chunking_version"],
            status=row["status"],
            chunks_total=row["chunks_total"],
            chunks_indexed=row["chunks_indexed"],
            error_message=row["error_message"],
            created_at=row["created_at"],
            finished_at=row["finished_at"],
        )
