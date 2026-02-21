from __future__ import annotations

import json
import mimetypes
import os
import plistlib
import sys
from dataclasses import dataclass
from pathlib import Path

from stemmacodicum.core.document_titles import derive_human_title
from stemmacodicum.core.errors import ResourceIngestError
from stemmacodicum.core.hashing import compute_file_digest
from stemmacodicum.core.ids import new_uuid
from stemmacodicum.core.time import now_utc_iso
from stemmacodicum.domain.models.resource import Resource
from stemmacodicum.infrastructure.archive.store import ArchiveStore
from stemmacodicum.infrastructure.db.repos.resource_repo import ResourceRepo

_MACOS_WHEREFROMS_ATTR = "com.apple.metadata:kMDItemWhereFroms"
_MACOS_QUARANTINE_ATTR = "com.apple.quarantine"


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

        discovered_urls = self._discover_download_urls(path, source_uri)
        download_url = self._pick_primary_download_url(discovered_urls)
        download_urls_json = json.dumps(discovered_urls, ensure_ascii=True) if discovered_urls else None

        digest_sha256 = compute_file_digest(path, "sha256")
        existing = self.resource_repo.get_by_digest(digest_sha256)
        if existing:
            merge_candidates = list(discovered_urls)
            if existing.download_url:
                merge_candidates.insert(0, existing.download_url)
            merged_urls = self._merge_download_urls(existing.download_urls_json, merge_candidates)
            merged_download_url = existing.download_url or self._pick_primary_download_url(merged_urls)
            merged_urls_json = json.dumps(merged_urls, ensure_ascii=True) if merged_urls else None
            merged_display_title = existing.display_title or derive_human_title(
                original_filename=existing.original_filename or path.name,
                source_uri=source_uri or existing.source_uri,
                fallback_id=existing.id,
            )
            if merged_download_url != existing.download_url or merged_urls_json != existing.download_urls_json:
                self.resource_repo.update_download_metadata(
                    existing.id,
                    download_url=merged_download_url,
                    download_urls_json=merged_urls_json,
                )
            if merged_display_title != existing.display_title:
                self.resource_repo.update_title_metadata(
                    existing.id,
                    display_title=merged_display_title,
                    title_candidates_json=existing.title_candidates_json,
                )
            if (
                merged_download_url != existing.download_url
                or merged_urls_json != existing.download_urls_json
                or merged_display_title != existing.display_title
            ):
                refreshed = self.resource_repo.get_by_id(existing.id)
                if refreshed is not None:
                    existing = refreshed
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
            download_url=download_url,
            download_urls_json=download_urls_json,
            display_title=derive_human_title(
                original_filename=path.name,
                source_uri=source_uri or str(path),
            ),
        )
        self.resource_repo.insert(resource)

        return IngestResult(resource=resource, status="ingested")

    def _discover_download_urls(self, path: Path, source_uri: str | None) -> list[str]:
        urls: list[str] = []
        urls.extend(self._read_macos_wherefroms(path))
        quarantine_url = self._read_macos_quarantine_url(path)
        if quarantine_url:
            urls.append(quarantine_url)
        if self._looks_like_network_url(source_uri):
            urls.append(source_uri.strip())
        return self._dedupe_urls(urls)

    def _merge_download_urls(self, existing_urls_json: str | None, new_urls: list[str]) -> list[str]:
        existing_urls: list[str] = []
        if existing_urls_json:
            try:
                parsed = json.loads(existing_urls_json)
            except json.JSONDecodeError:
                parsed = []
            if isinstance(parsed, list):
                existing_urls = [item.strip() for item in parsed if isinstance(item, str) and item.strip()]
        return self._dedupe_urls(existing_urls + new_urls)

    @staticmethod
    def _is_macos() -> bool:
        return sys.platform == "darwin"

    def _read_macos_wherefroms(self, path: Path) -> list[str]:
        if not self._is_macos() or not hasattr(os, "getxattr"):
            return []
        try:
            raw = os.getxattr(path, _MACOS_WHEREFROMS_ATTR)
        except OSError:
            return []
        if not raw:
            return []
        try:
            parsed = plistlib.loads(raw)
        except Exception:
            return []
        if isinstance(parsed, list):
            return [item.strip() for item in parsed if isinstance(item, str) and item.strip()]
        if isinstance(parsed, str) and parsed.strip():
            return [parsed.strip()]
        return []

    def _read_macos_quarantine_url(self, path: Path) -> str | None:
        if not self._is_macos() or not hasattr(os, "getxattr"):
            return None
        try:
            raw = os.getxattr(path, _MACOS_QUARANTINE_ATTR)
        except OSError:
            return None
        text = raw.decode("utf-8", errors="ignore")
        if not text:
            return None
        for token in reversed(text.split(";")):
            candidate = token.strip()
            if self._looks_like_network_url(candidate):
                return candidate
        return None

    @staticmethod
    def _looks_like_network_url(value: str | None) -> bool:
        if not value:
            return False
        lowered = value.strip().lower()
        return lowered.startswith("http://") or lowered.startswith("https://")

    @staticmethod
    def _pick_primary_download_url(urls: list[str]) -> str | None:
        if not urls:
            return None
        for url in urls:
            if IngestionService._looks_like_network_url(url):
                return url
        return urls[0]

    @staticmethod
    def _dedupe_urls(urls: list[str]) -> list[str]:
        seen: set[str] = set()
        normalized: list[str] = []
        for url in urls:
            clean = url.strip()
            if not clean or clean in seen:
                continue
            seen.add(clean)
            normalized.append(clean)
        return normalized
