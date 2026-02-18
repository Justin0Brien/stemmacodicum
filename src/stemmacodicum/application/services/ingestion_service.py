from __future__ import annotations

import mimetypes
from dataclasses import dataclass
from pathlib import Path

from stemmacodicum.core.errors import ResourceIngestError
from stemmacodicum.core.hashing import compute_file_digest
from stemmacodicum.core.ids import new_uuid
from stemmacodicum.core.time import now_utc_iso
from stemmacodicum.domain.models.resource import Resource
from stemmacodicum.infrastructure.archive.store import ArchiveStore
from stemmacodicum.infrastructure.db.repos.resource_repo import ResourceRepo


@dataclass(slots=True)
class IngestResult:
    resource: Resource
    status: str


class IngestionService:
    def __init__(self, resource_repo: ResourceRepo, archive_store: ArchiveStore) -> None:
        self.resource_repo = resource_repo
        self.archive_store = archive_store

    def ingest_file(self, file_path: Path, source_uri: str | None = None) -> IngestResult:
        path = file_path.expanduser().resolve()
        if not path.exists() or not path.is_file():
            raise ResourceIngestError(f"File not found: {path}")

        digest_sha256 = compute_file_digest(path, "sha256")
        existing = self.resource_repo.get_by_digest(digest_sha256)
        if existing:
            return IngestResult(resource=existing, status="duplicate")

        media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        suffix = path.suffix.lower() if path.suffix else ""
        archived_path = self.archive_store.store_file_immutable(path, digest_sha256, suffix)

        resource = Resource(
            id=new_uuid(),
            digest_sha256=digest_sha256,
            media_type=media_type,
            original_filename=path.name,
            source_uri=source_uri or str(path),
            archived_relpath=str(archived_path.relative_to(self.archive_store.base_dir)),
            size_bytes=path.stat().st_size,
            ingested_at=now_utc_iso(),
        )
        self.resource_repo.insert(resource)

        return IngestResult(resource=resource, status="ingested")
