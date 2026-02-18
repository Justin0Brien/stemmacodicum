from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ClaimSet:
    id: str
    name: str
    description: str | None
    created_at: str


@dataclass(slots=True)
class Claim:
    id: str
    claim_set_id: str
    claim_type: str
    subject: str | None
    predicate: str | None
    object_text: str | None
    narrative_text: str | None
    value_raw: str | None
    value_parsed: float | None
    currency: str | None
    scale_factor: int | None
    period_label: str | None
    source_cite_id: str | None
    status: str
    created_at: str
    updated_at: str
