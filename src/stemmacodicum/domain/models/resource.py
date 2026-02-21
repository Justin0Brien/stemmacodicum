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
    download_url: str | None = None
    download_urls_json: str | None = None
    display_title: str | None = None
    title_candidates_json: str | None = None
