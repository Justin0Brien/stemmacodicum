from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Resource:
    id: str
    digest_sha256: str
    media_type: str
    original_filename: str
    source_uri: str | None
    archived_relpath: str
    size_bytes: int
    ingested_at: str
