from __future__ import annotations

import json
import logging
import mimetypes
import os
import plistlib
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen

from stemmacodicum.core.document_titles import derive_human_title
from stemmacodicum.core.errors import MissingSourceUrlError, ResourceIngestError
from stemmacodicum.core.hashing import compute_file_digest
from stemmacodicum.core.ids import new_uuid
from stemmacodicum.core.time import now_utc_iso
from stemmacodicum.domain.models.resource import Resource
from stemmacodicum.infrastructure.archive.store import ArchiveStore
from stemmacodicum.infrastructure.db.repos.resource_repo import ResourceRepo

_MACOS_WHEREFROMS_ATTR = "com.apple.metadata:kMDItemWhereFroms"
_MACOS_QUARANTINE_ATTR = "com.apple.quarantine"
_LINUX_ORIGIN_URL_ATTRS = (
    "user.xdg.origin.url",
    "user.xdg.referrer.url",
    "user.kde.origin.url",
)

_WAYBACK_SAVE_PREFIX = "https://web.archive.org/save/"
_WAYBACK_DEFAULT_TIMEOUT_SECONDS = 8.0
_WAYBACK_RESERVED_HOST_EXACT = {
    "localhost",
    "example.com",
    "example.net",
    "example.org",
}
_WAYBACK_RESERVED_HOST_SUFFIX = (
    ".localhost",
    ".example.com",
    ".example.net",
    ".example.org",
    ".example",
    ".test",
    ".invalid",
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class IngestResult:
    resource: Resource
    status: str


class IngestionService:
    def __init__(
        self,
        resource_repo: ResourceRepo,
        archive_store: ArchiveStore,
        *,
        wayback_enabled: bool | None = None,
    ) -> None:
        self.resource_repo = resource_repo
        self.archive_store = archive_store
        self.wayback_enabled = (
            self._env_bool("STEMMA_WAYBACK_REGISTER_ENABLED", default=True)
            if wayback_enabled is None
            else bool(wayback_enabled)
        )
        self.wayback_timeout_seconds = self._env_float(
            "STEMMA_WAYBACK_TIMEOUT_SECONDS",
            default=_WAYBACK_DEFAULT_TIMEOUT_SECONDS,
        )

    def ingest_file(self, file_path: Path, source_uri: str | None = None) -> IngestResult:
        path = file_path.expanduser().resolve()
        if not path.exists() or not path.is_file():
            raise ResourceIngestError(f"File not found: {path}")

        discovered_urls = self._discover_download_urls(path, source_uri)
        download_url = self._pick_primary_download_url(discovered_urls)
        if not download_url:
            raise MissingSourceUrlError(
                (
                    f"Missing source URL for {path}. Ingestion skipped until source resolution succeeds. "
                    "Add a real http(s) URL in file xattrs "
                    f"({_MACOS_WHEREFROMS_ATTR}, {_MACOS_QUARANTINE_ATTR}, {', '.join(_LINUX_ORIGIN_URL_ATTRS)}) "
                    "or provide source_uri explicitly."
                )
            )
        download_urls_json = json.dumps(discovered_urls, ensure_ascii=True)

        digest_sha256 = compute_file_digest(path, "sha256")
        existing = self.resource_repo.get_by_digest(digest_sha256)
        if existing:
            merge_candidates = list(discovered_urls)
            if existing.download_url:
                merge_candidates.insert(0, existing.download_url)
            merged_urls = self._merge_download_urls(existing.download_urls_json, merge_candidates)
            merged_download_url = self._pick_primary_download_url(merged_urls)
            if not merged_download_url:
                raise MissingSourceUrlError(
                    f"Missing source URL for duplicate resource {existing.id}. Ingestion skipped."
                )
            merged_urls_json = json.dumps(merged_urls, ensure_ascii=True)
            merged_display_title = existing.display_title or derive_human_title(
                original_filename=existing.original_filename or path.name,
                source_uri=merged_download_url or existing.source_uri,
                fallback_id=existing.id,
            )
            metadata_changed = (
                merged_download_url != existing.download_url
                or merged_urls_json != existing.download_urls_json
            )
            if metadata_changed:
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
            if metadata_changed or merged_display_title != existing.display_title:
                refreshed = self.resource_repo.get_by_id(existing.id)
                if refreshed is not None:
                    existing = refreshed
            if metadata_changed and merged_download_url:
                self._register_with_wayback(merged_download_url)
            return IngestResult(resource=existing, status="duplicate")

        media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        suffix = path.suffix.lower() if path.suffix else ""
        archived_path = self.archive_store.store_file_immutable(path, digest_sha256, suffix)

        resource = Resource(
            id=new_uuid(),
            digest_sha256=digest_sha256,
            media_type=media_type,
            original_filename=path.name,
            source_uri=download_url,
            archived_relpath=str(archived_path.relative_to(self.archive_store.base_dir)),
            size_bytes=path.stat().st_size,
            ingested_at=now_utc_iso(),
            download_url=download_url,
            download_urls_json=download_urls_json,
            display_title=derive_human_title(
                original_filename=path.name,
                source_uri=download_url,
            ),
        )
        self.resource_repo.insert(resource)
        self._register_with_wayback(download_url)

        return IngestResult(resource=resource, status="ingested")

    def _discover_download_urls(self, path: Path, source_uri: str | None) -> list[str]:
        urls: list[str] = []
        urls.extend(self._read_macos_wherefroms(path))
        quarantine_url = self._read_macos_quarantine_url(path)
        if quarantine_url:
            urls.append(quarantine_url)
        urls.extend(self._read_linux_origin_urls(path))
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
        return self._parse_raw_xattr_urls(raw)

    def _read_linux_origin_urls(self, path: Path) -> list[str]:
        if not hasattr(os, "getxattr"):
            return []
        urls: list[str] = []
        for attr in _LINUX_ORIGIN_URL_ATTRS:
            try:
                raw = os.getxattr(path, attr)
            except OSError:
                continue
            urls.extend(self._parse_raw_xattr_urls(raw))
        return urls

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

    def _register_with_wayback(self, source_url: str) -> None:
        if not self.wayback_enabled:
            return
        if not self._looks_like_network_url(source_url):
            return
        if self._is_reserved_host(source_url):
            return

        try:
            encoded = quote(source_url, safe="")
            request = Request(
                f"{_WAYBACK_SAVE_PREFIX}{encoded}",
                method="GET",
                headers={
                    "User-Agent": "stemmacodicum/0.1",
                    "Accept": "text/html,application/xhtml+xml,application/xml",
                },
            )
            with urlopen(request, timeout=self.wayback_timeout_seconds) as response:
                if int(getattr(response, "status", 200) or 200) >= 400:
                    logger.warning(
                        "Wayback snapshot registration returned HTTP %s for %s",
                        getattr(response, "status", "unknown"),
                        source_url,
                    )
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            logger.warning("Wayback snapshot registration failed for %s: %s", source_url, exc)

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
        return urls[0]

    @staticmethod
    def _dedupe_urls(urls: list[str]) -> list[str]:
        seen: set[str] = set()
        normalized: list[str] = []
        for url in urls:
            clean = url.strip()
            if not clean or not IngestionService._looks_like_network_url(clean) or clean in seen:
                continue
            seen.add(clean)
            normalized.append(clean)
        return normalized

    @staticmethod
    def _parse_raw_xattr_urls(raw: bytes) -> list[str]:
        if not raw:
            return []
        try:
            parsed = plistlib.loads(raw)
        except Exception:
            parsed = None
        if isinstance(parsed, list):
            return [item.strip() for item in parsed if isinstance(item, str) and item.strip()]
        if isinstance(parsed, str) and parsed.strip():
            return [parsed.strip()]

        decoded = raw.decode("utf-8", errors="ignore").strip()
        if not decoded:
            return []

        try:
            parsed_json = json.loads(decoded)
        except Exception:
            parsed_json = None
        if isinstance(parsed_json, str):
            return [parsed_json.strip()]
        if isinstance(parsed_json, list):
            return [item.strip() for item in parsed_json if isinstance(item, str) and item.strip()]

        values: list[str] = []
        for row in decoded.replace("\r", "\n").split("\n"):
            for token in row.split(";"):
                candidate = token.strip()
                if candidate:
                    values.append(candidate)
        return values

    @staticmethod
    def _is_reserved_host(source_url: str) -> bool:
        host = (urlsplit(source_url).hostname or "").strip().lower()
        if not host:
            return True
        if host in _WAYBACK_RESERVED_HOST_EXACT:
            return True
        return any(host.endswith(suffix) for suffix in _WAYBACK_RESERVED_HOST_SUFFIX)

    @staticmethod
    def _env_bool(name: str, default: bool) -> bool:
        raw = os.getenv(name)
        if raw is None:
            return default
        lowered = str(raw).strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
        return default

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        raw = os.getenv(name)
        if raw is None:
            return default
        try:
            parsed = float(raw)
        except (TypeError, ValueError):
            return default
        return parsed if parsed > 0 else default
