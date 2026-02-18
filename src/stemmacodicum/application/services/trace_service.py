from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from stemmacodicum.core.errors import TraceError
from stemmacodicum.infrastructure.db.sqlite import get_connection


@dataclass(slots=True)
class ClaimTrace:
    claim_id: str
    claim_type: str
    claim_text: str
    evidence: list[dict[str, object]]


@dataclass(slots=True)
class ResourceTrace:
    resource_id: str
    digest_sha256: str
    references: list[dict[str, object]]
    claims: list[dict[str, object]]


@dataclass(slots=True)
class CitationTrace:
    cite_id: str
    reference: dict[str, object] | None
    resources: list[dict[str, object]]
    claims: list[dict[str, object]]


class TraceService:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def trace_claim(self, claim_id: str) -> ClaimTrace:
        with get_connection(self.db_path) as conn:
            claim = conn.execute("SELECT * FROM claims WHERE id = ?", (claim_id,)).fetchone()
            if claim is None:
                raise TraceError(f"Claim not found: {claim_id}")

            evidence_rows = conn.execute(
                """
                SELECT e.*
                FROM evidence_items e
                INNER JOIN claim_evidence_bindings b ON b.evidence_id = e.id
                WHERE b.claim_id = ?
                ORDER BY e.created_at ASC
                """,
                (claim_id,),
            ).fetchall()

            evidence: list[dict[str, object]] = []
            for row in evidence_rows:
                selectors = conn.execute(
                    "SELECT selector_type, selector_json FROM evidence_selectors WHERE evidence_id = ?",
                    (row["id"],),
                ).fetchall()
                evidence.append(
                    {
                        "evidence_id": row["id"],
                        "resource_id": row["resource_id"],
                        "role": row["role"],
                        "selectors": [
                            {
                                "type": s["selector_type"],
                                "payload": json.loads(s["selector_json"]),
                            }
                            for s in selectors
                        ],
                    }
                )

        text = claim["narrative_text"] or claim["object_text"] or ""
        return ClaimTrace(
            claim_id=claim["id"],
            claim_type=claim["claim_type"],
            claim_text=text,
            evidence=evidence,
        )

    def trace_resource(self, *, resource_id: str | None = None, digest_sha256: str | None = None) -> ResourceTrace:
        if not resource_id and not digest_sha256:
            raise TraceError("Provide resource_id or digest_sha256")

        with get_connection(self.db_path) as conn:
            if resource_id:
                resource = conn.execute("SELECT * FROM resources WHERE id = ?", (resource_id,)).fetchone()
            else:
                resource = conn.execute(
                    "SELECT * FROM resources WHERE digest_sha256 = ?",
                    (digest_sha256,),
                ).fetchone()

            if resource is None:
                token = resource_id or digest_sha256
                raise TraceError(f"Resource not found: {token}")

            ref_rows = conn.execute(
                """
                SELECT r.cite_id, r.title, r.year
                FROM reference_entries r
                INNER JOIN reference_resources rr ON rr.reference_id = r.id
                WHERE rr.resource_id = ?
                ORDER BY rr.linked_at DESC
                """,
                (resource["id"],),
            ).fetchall()

            claim_rows = conn.execute(
                """
                SELECT DISTINCT c.id, c.claim_type, c.predicate, c.narrative_text
                FROM claims c
                INNER JOIN claim_evidence_bindings b ON b.claim_id = c.id
                INNER JOIN evidence_items e ON e.id = b.evidence_id
                WHERE e.resource_id = ?
                ORDER BY c.created_at DESC
                """,
                (resource["id"],),
            ).fetchall()

        return ResourceTrace(
            resource_id=resource["id"],
            digest_sha256=resource["digest_sha256"],
            references=[{"cite_id": r["cite_id"], "title": r["title"], "year": r["year"]} for r in ref_rows],
            claims=[
                {
                    "claim_id": c["id"],
                    "claim_type": c["claim_type"],
                    "predicate": c["predicate"],
                    "text": c["narrative_text"],
                }
                for c in claim_rows
            ],
        )

    def trace_citation(self, cite_id: str) -> CitationTrace:
        with get_connection(self.db_path) as conn:
            citation = conn.execute("SELECT * FROM citations WHERE cite_id = ?", (cite_id,)).fetchone()
            if citation is None:
                raise TraceError(f"Citation not found: {cite_id}")

            ref = conn.execute(
                "SELECT id, cite_id, title, year, doi, url FROM reference_entries WHERE cite_id = ?",
                (cite_id,),
            ).fetchone()

            resources: list[dict[str, object]] = []
            if ref is not None:
                resource_rows = conn.execute(
                    """
                    SELECT res.id, res.digest_sha256, res.original_filename
                    FROM resources res
                    INNER JOIN reference_resources rr ON rr.resource_id = res.id
                    WHERE rr.reference_id = ?
                    """,
                    (ref["id"],),
                ).fetchall()
                resources = [
                    {
                        "resource_id": r["id"],
                        "digest_sha256": r["digest_sha256"],
                        "filename": r["original_filename"],
                    }
                    for r in resource_rows
                ]

            claims_rows = conn.execute(
                """
                SELECT id, claim_type, predicate, narrative_text
                FROM claims
                WHERE source_cite_id = ?
                ORDER BY created_at DESC
                """,
                (cite_id,),
            ).fetchall()

        return CitationTrace(
            cite_id=cite_id,
            reference=(
                {
                    "cite_id": ref["cite_id"],
                    "title": ref["title"],
                    "year": ref["year"],
                    "doi": ref["doi"],
                    "url": ref["url"],
                }
                if ref is not None
                else None
            ),
            resources=resources,
            claims=[
                {
                    "claim_id": c["id"],
                    "claim_type": c["claim_type"],
                    "predicate": c["predicate"],
                    "text": c["narrative_text"],
                }
                for c in claims_rows
            ],
        )
