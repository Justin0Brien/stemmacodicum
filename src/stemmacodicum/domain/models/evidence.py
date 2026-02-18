from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class EvidenceItem:
    id: str
    resource_id: str
    role: str
    page_index: int | None
    note: str | None
    created_at: str


@dataclass(slots=True)
class EvidenceSelector:
    id: str
    evidence_id: str
    selector_type: str
    selector_json: str
    created_at: str
