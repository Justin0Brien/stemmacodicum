from __future__ import annotations

from pathlib import Path

from stemmacodicum.domain.models.resource import Resource
from stemmacodicum.infrastructure.db.sqlite import get_connection


class ResourceRepo:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._ensure_resource_metadata_columns()

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
                    download_url,
                    download_urls_json,
                    display_title,
                    title_candidates_json,
                    archived_relpath,
                    size_bytes,
                    ingested_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    resource.id,
                    resource.digest_sha256,
                    resource.media_type,
                    resource.original_filename,
                    resource.source_uri,
                    resource.download_url,
                    resource.download_urls_json,
                    resource.display_title,
                    resource.title_candidates_json,
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

    def update_download_metadata(
        self,
        resource_id: str,
        *,
        download_url: str | None,
        download_urls_json: str | None,
    ) -> None:
        with get_connection(self.db_path) as conn:
            conn.execute(
                """
                UPDATE resources
                SET download_url = ?, download_urls_json = ?
                WHERE id = ?
                """,
                (download_url, download_urls_json, resource_id),
            )
            conn.commit()

    def update_title_metadata(
        self,
        resource_id: str,
        *,
        display_title: str | None,
        title_candidates_json: str | None,
    ) -> None:
        with get_connection(self.db_path) as conn:
            conn.execute(
                """
                UPDATE resources
                SET display_title = ?, title_candidates_json = ?
                WHERE id = ?
                """,
                (display_title, title_candidates_json, resource_id),
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
            download_url=row["download_url"] if "download_url" in row.keys() else None,
            download_urls_json=row["download_urls_json"] if "download_urls_json" in row.keys() else None,
            display_title=row["display_title"] if "display_title" in row.keys() else None,
            title_candidates_json=row["title_candidates_json"] if "title_candidates_json" in row.keys() else None,
        )

    def _ensure_resource_metadata_columns(self) -> None:
        with get_connection(self.db_path) as conn:
            has_resources = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'resources'"
            ).fetchone()
            if not has_resources:
                return
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(resources)").fetchall()}
            if "download_url" not in columns:
                conn.execute("ALTER TABLE resources ADD COLUMN download_url TEXT")
            if "download_urls_json" not in columns:
                conn.execute("ALTER TABLE resources ADD COLUMN download_urls_json TEXT")
            if "display_title" not in columns:
                conn.execute("ALTER TABLE resources ADD COLUMN display_title TEXT")
            if "title_candidates_json" not in columns:
                conn.execute("ALTER TABLE resources ADD COLUMN title_candidates_json TEXT")
            conn.commit()
