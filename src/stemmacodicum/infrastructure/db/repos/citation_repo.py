from __future__ import annotations

from pathlib import Path

from stemmacodicum.domain.models.citation import Citation
from stemmacodicum.infrastructure.db.sqlite import get_connection


class CitationRepo:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def get_by_normalized_key(self, normalized_key: str) -> Citation | None:
        with get_connection(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM citations WHERE normalized_key = ?",
                (normalized_key,),
            ).fetchone()
        return self._to_model(row) if row else None

    def get_by_cite_id(self, cite_id: str) -> Citation | None:
        with get_connection(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM citations WHERE cite_id = ?",
                (cite_id,),
            ).fetchone()
        return self._to_model(row) if row else None

    def exists_cite_id(self, cite_id: str) -> bool:
        with get_connection(self.db_path) as conn:
            row = conn.execute(
                "SELECT 1 FROM citations WHERE cite_id = ?",
                (cite_id,),
            ).fetchone()
        return row is not None

    def insert(self, citation: Citation) -> None:
        with get_connection(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO citations (cite_id, original_key, normalized_key, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    citation.cite_id,
                    citation.original_key,
                    citation.normalized_key,
                    citation.created_at,
                ),
            )
            conn.commit()

    def list(self, limit: int = 100) -> list[Citation]:
        with get_connection(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT * FROM citations
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._to_model(row) for row in rows]

    @staticmethod
    def _to_model(row) -> Citation:
        return Citation(
            cite_id=row["cite_id"],
            original_key=row["original_key"],
            normalized_key=row["normalized_key"],
            created_at=row["created_at"],
        )
