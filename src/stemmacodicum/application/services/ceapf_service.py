from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from stemmacodicum.core.errors import CEAPFError
from stemmacodicum.core.ids import new_uuid
from stemmacodicum.core.time import now_utc_iso
from stemmacodicum.infrastructure.db.sqlite import get_connection


@dataclass(slots=True)
class PropositionRecord:
    id: str
    proposition: dict[str, object]


class CEAPFService:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def create_proposition(self, proposition: dict[str, object]) -> str:
        if not proposition:
            raise CEAPFError("Proposition payload cannot be empty")

        proposition_id = new_uuid()
        with get_connection(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO propositions (id, proposition_json, created_at)
                VALUES (?, ?, ?)
                """,
                (proposition_id, json.dumps(proposition, ensure_ascii=True, sort_keys=True), now_utc_iso()),
            )
            conn.commit()
        return proposition_id

    def create_assertion_event(
        self,
        proposition_id: str,
        asserting_agent: str,
        modality: str,
        evidence_id: str | None = None,
    ) -> str:
        with get_connection(self.db_path) as conn:
            p = conn.execute("SELECT id FROM propositions WHERE id = ?", (proposition_id,)).fetchone()
            if p is None:
                raise CEAPFError(f"Proposition not found: {proposition_id}")

            if evidence_id:
                e = conn.execute("SELECT id FROM evidence_items WHERE id = ?", (evidence_id,)).fetchone()
                if e is None:
                    raise CEAPFError(f"Evidence item not found: {evidence_id}")

            assertion_id = new_uuid()
            conn.execute(
                """
                INSERT INTO assertion_events (
                    id,
                    proposition_id,
                    asserting_agent,
                    modality,
                    evidence_id,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (assertion_id, proposition_id, asserting_agent, modality, evidence_id, now_utc_iso()),
            )
            conn.commit()
        return assertion_id

    def add_argument_relation(
        self,
        relation_type: str,
        from_node_type: str,
        from_node_id: str,
        to_node_type: str,
        to_node_id: str,
    ) -> str:
        allowed = {"supports", "rebuts", "undercuts", "qualifies"}
        if relation_type not in allowed:
            raise CEAPFError(f"Unsupported relation_type: {relation_type}")

        rel_id = new_uuid()
        with get_connection(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO argument_relations (
                    id,
                    relation_type,
                    from_node_type,
                    from_node_id,
                    to_node_type,
                    to_node_id,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rel_id,
                    relation_type,
                    from_node_type,
                    from_node_id,
                    to_node_type,
                    to_node_id,
                    now_utc_iso(),
                ),
            )
            conn.commit()
        return rel_id

    def list_propositions(self, limit: int = 100) -> list[PropositionRecord]:
        with get_connection(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT id, proposition_json
                FROM propositions
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        return [
            PropositionRecord(id=row["id"], proposition=json.loads(row["proposition_json"])) for row in rows
        ]
