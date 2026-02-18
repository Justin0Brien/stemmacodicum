from __future__ import annotations

from pathlib import Path

from stemmacodicum.domain.models.resource import Resource
from stemmacodicum.infrastructure.db.sqlite import get_connection


class ResourceRepo:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def insert(self, resource: Resource) -> None:
        with get_connection(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO resources (
                    id,
                    digest_sha256,
                    media_type,
                    original_filename,
                    source_uri,
                    archived_relpath,
                    size_bytes,
                    ingested_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    resource.id,
                    resource.digest_sha256,
                    resource.media_type,
                    resource.original_filename,
                    resource.source_uri,
                    resource.archived_relpath,
                    resource.size_bytes,
                    resource.ingested_at,
                ),
            )
            conn.execute(
                """
                INSERT INTO resource_digests (resource_id, algorithm, digest_value)
                VALUES (?, 'sha256', ?)
                """,
                (resource.id, resource.digest_sha256),
            )
            conn.commit()

    def get_by_digest(self, digest_sha256: str) -> Resource | None:
        with get_connection(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM resources WHERE digest_sha256 = ?",
                (digest_sha256,),
            ).fetchone()
        return self._to_model(row) if row else None

    def get_by_id(self, resource_id: str) -> Resource | None:
        with get_connection(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM resources WHERE id = ?",
                (resource_id,),
            ).fetchone()
        return self._to_model(row) if row else None

    def list(self, limit: int = 100) -> list[Resource]:
        with get_connection(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT * FROM resources
                ORDER BY ingested_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._to_model(row) for row in rows]

    @staticmethod
    def _to_model(row) -> Resource:
        return Resource(
            id=row["id"],
            digest_sha256=row["digest_sha256"],
            media_type=row["media_type"],
            original_filename=row["original_filename"],
            source_uri=row["source_uri"],
            archived_relpath=row["archived_relpath"],
            size_bytes=row["size_bytes"],
            ingested_at=row["ingested_at"],
        )
