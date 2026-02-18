from __future__ import annotations

from pathlib import Path

from stemmacodicum.domain.models.reference import Reference
from stemmacodicum.infrastructure.db.sqlite import get_connection


class ReferenceRepo:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def upsert(self, reference: Reference) -> str:
        with get_connection(self.db_path) as conn:
            existing = conn.execute(
                "SELECT id FROM reference_entries WHERE cite_id = ?",
                (reference.cite_id,),
            ).fetchone()

            if existing:
                conn.execute(
                    """
                    UPDATE reference_entries
                    SET
                        entry_type = ?,
                        title = ?,
                        author = ?,
                        year = ?,
                        doi = ?,
                        url = ?,
                        raw_bibtex = ?,
                        imported_at = ?
                    WHERE cite_id = ?
                    """,
                    (
                        reference.entry_type,
                        reference.title,
                        reference.author,
                        reference.year,
                        reference.doi,
                        reference.url,
                        reference.raw_bibtex,
                        reference.imported_at,
                        reference.cite_id,
                    ),
                )
                conn.commit()
                return "updated"

            conn.execute(
                """
                INSERT INTO reference_entries (
                    id,
                    cite_id,
                    entry_type,
                    title,
                    author,
                    year,
                    doi,
                    url,
                    raw_bibtex,
                    imported_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    reference.id,
                    reference.cite_id,
                    reference.entry_type,
                    reference.title,
                    reference.author,
                    reference.year,
                    reference.doi,
                    reference.url,
                    reference.raw_bibtex,
                    reference.imported_at,
                ),
            )
            conn.commit()
            return "inserted"

    def get_by_cite_id(self, cite_id: str) -> Reference | None:
        with get_connection(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM reference_entries WHERE cite_id = ?",
                (cite_id,),
            ).fetchone()
        return self._to_model(row) if row else None

    def list(self, limit: int = 100) -> list[Reference]:
        with get_connection(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT * FROM reference_entries
                ORDER BY imported_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._to_model(row) for row in rows]

    def link_to_resource(self, reference_id: str, resource_id: str, linked_at: str) -> None:
        with get_connection(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO reference_resources (
                    reference_id,
                    resource_id,
                    linked_at
                ) VALUES (?, ?, ?)
                """,
                (reference_id, resource_id, linked_at),
            )
            conn.commit()

    @staticmethod
    def _to_model(row) -> Reference:
        return Reference(
            id=row["id"],
            cite_id=row["cite_id"],
            entry_type=row["entry_type"],
            title=row["title"],
            author=row["author"],
            year=row["year"],
            doi=row["doi"],
            url=row["url"],
            raw_bibtex=row["raw_bibtex"],
            imported_at=row["imported_at"],
        )
