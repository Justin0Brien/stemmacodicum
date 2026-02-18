from __future__ import annotations

from pathlib import Path

from stemmacodicum.domain.models.claim import Claim, ClaimSet
from stemmacodicum.infrastructure.db.sqlite import get_connection


class ClaimRepo:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def get_claim_set_by_name(self, name: str) -> ClaimSet | None:
        with get_connection(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM claim_sets WHERE name = ?",
                (name,),
            ).fetchone()
        return self._to_claim_set(row) if row else None

    def get_claim_set_by_id(self, claim_set_id: str) -> ClaimSet | None:
        with get_connection(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM claim_sets WHERE id = ?",
                (claim_set_id,),
            ).fetchone()
        return self._to_claim_set(row) if row else None

    def insert_claim_set(self, claim_set: ClaimSet) -> None:
        with get_connection(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO claim_sets (id, name, description, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (claim_set.id, claim_set.name, claim_set.description, claim_set.created_at),
            )
            conn.commit()

    def list_claim_sets(self, limit: int = 100) -> list[ClaimSet]:
        with get_connection(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT * FROM claim_sets
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._to_claim_set(row) for row in rows]

    def get_claim_by_id(self, claim_id: str) -> Claim | None:
        with get_connection(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM claims WHERE id = ?",
                (claim_id,),
            ).fetchone()
        return self._to_claim(row) if row else None

    def insert_claim(self, claim: Claim) -> None:
        with get_connection(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO claims (
                    id,
                    claim_set_id,
                    claim_type,
                    subject,
                    predicate,
                    object_text,
                    narrative_text,
                    value_raw,
                    value_parsed,
                    currency,
                    scale_factor,
                    period_label,
                    source_cite_id,
                    status,
                    created_at,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    claim.id,
                    claim.claim_set_id,
                    claim.claim_type,
                    claim.subject,
                    claim.predicate,
                    claim.object_text,
                    claim.narrative_text,
                    claim.value_raw,
                    claim.value_parsed,
                    claim.currency,
                    claim.scale_factor,
                    claim.period_label,
                    claim.source_cite_id,
                    claim.status,
                    claim.created_at,
                    claim.updated_at,
                ),
            )
            conn.commit()

    def list_claims(self, claim_set_id: str | None = None, limit: int = 200) -> list[Claim]:
        with get_connection(self.db_path) as conn:
            if claim_set_id:
                rows = conn.execute(
                    """
                    SELECT * FROM claims
                    WHERE claim_set_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (claim_set_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM claims
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
        return [self._to_claim(row) for row in rows]

    @staticmethod
    def _to_claim_set(row) -> ClaimSet:
        return ClaimSet(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            created_at=row["created_at"],
        )

    @staticmethod
    def _to_claim(row) -> Claim:
        return Claim(
            id=row["id"],
            claim_set_id=row["claim_set_id"],
            claim_type=row["claim_type"],
            subject=row["subject"],
            predicate=row["predicate"],
            object_text=row["object_text"],
            narrative_text=row["narrative_text"],
            value_raw=row["value_raw"],
            value_parsed=row["value_parsed"],
            currency=row["currency"],
            scale_factor=row["scale_factor"],
            period_label=row["period_label"],
            source_cite_id=row["source_cite_id"],
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
