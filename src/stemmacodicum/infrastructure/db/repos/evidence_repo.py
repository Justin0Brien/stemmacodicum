from __future__ import annotations

from pathlib import Path

from stemmacodicum.domain.models.evidence import EvidenceItem, EvidenceSelector
from stemmacodicum.infrastructure.db.sqlite import get_connection


class EvidenceRepo:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def insert_evidence_item(self, item: EvidenceItem) -> None:
        with get_connection(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO evidence_items (id, resource_id, role, page_index, note, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (item.id, item.resource_id, item.role, item.page_index, item.note, item.created_at),
            )
            conn.commit()

    def insert_selector(self, selector: EvidenceSelector) -> None:
        with get_connection(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO evidence_selectors (id, evidence_id, selector_type, selector_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    selector.id,
                    selector.evidence_id,
                    selector.selector_type,
                    selector.selector_json,
                    selector.created_at,
                ),
            )
            conn.commit()

    def bind_claim_to_evidence(self, claim_id: str, evidence_id: str, created_at: str) -> None:
        with get_connection(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO claim_evidence_bindings (claim_id, evidence_id, created_at)
                VALUES (?, ?, ?)
                """,
                (claim_id, evidence_id, created_at),
            )
            conn.commit()

    def list_evidence_for_claim(self, claim_id: str) -> list[tuple[EvidenceItem, list[EvidenceSelector]]]:
        with get_connection(self.db_path) as conn:
            item_rows = conn.execute(
                """
                SELECT e.*
                FROM evidence_items e
                INNER JOIN claim_evidence_bindings b ON b.evidence_id = e.id
                WHERE b.claim_id = ?
                ORDER BY e.created_at ASC
                """,
                (claim_id,),
            ).fetchall()

            output: list[tuple[EvidenceItem, list[EvidenceSelector]]] = []
            for row in item_rows:
                evidence = self._to_item(row)
                selector_rows = conn.execute(
                    """
                    SELECT *
                    FROM evidence_selectors
                    WHERE evidence_id = ?
                    ORDER BY created_at ASC
                    """,
                    (evidence.id,),
                ).fetchall()
                selectors = [self._to_selector(r) for r in selector_rows]
                output.append((evidence, selectors))

        return output

    @staticmethod
    def _to_item(row) -> EvidenceItem:
        return EvidenceItem(
            id=row["id"],
            resource_id=row["resource_id"],
            role=row["role"],
            page_index=row["page_index"],
            note=row["note"],
            created_at=row["created_at"],
        )

    @staticmethod
    def _to_selector(row) -> EvidenceSelector:
        return EvidenceSelector(
            id=row["id"],
            evidence_id=row["evidence_id"],
            selector_type=row["selector_type"],
            selector_json=row["selector_json"],
            created_at=row["created_at"],
        )
