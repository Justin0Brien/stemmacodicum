from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Reference:
    id: str
    cite_id: str
    entry_type: str
    title: str | None
    author: str | None
    year: str | None
    doi: str | None
    url: str | None
    raw_bibtex: str
    imported_at: str
