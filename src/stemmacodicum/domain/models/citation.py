from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Citation:
    cite_id: str
    original_key: str
    normalized_key: str
    created_at: str
