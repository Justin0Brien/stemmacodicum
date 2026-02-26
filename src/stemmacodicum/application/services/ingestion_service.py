from __future__ import annotations

import atexit
from concurrent.futures import ThreadPoolExecutor
import json
import logging
import mimetypes
import os
import plistlib
import re
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen

from stemmacodicum.core.document_titles import derive_human_title
from stemmacodicum.core.errors import EmptySourceFileError, MissingSourceUrlError, ResourceIngestError
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
_WAYBACK_DEFAULT_WORKERS = 2
_WAYBACK_PROBE_HOST = "web.archive.org"
_WAYBACK_PROBE_PORT = 443
_WAYBACK_PROBE_TIMEOUT_SECONDS = 3.0
_WAYBACK_PROBE_INTERVAL_SECONDS = 30.0
_WAYBACK_SUSPEND_SECONDS = 120.0
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
_NETWORK_URL_RE = re.compile(r"https?://[^\s\"'<>]+", flags=re.IGNORECASE)

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
        wayback_async: bool | None = None,
    ) -> None:
        self.resource_repo = resource_repo
        self.archive_store = archive_store
        self.wayback_enabled = (
            self._env_bool("STEMMA_WAYBACK_REGISTER_ENABLED", default=True)
            if wayback_enabled is None
            else bool(wayback_enabled)
        )
        self.wayback_async = (
            self._env_bool("STEMMA_WAYBACK_REGISTER_ASYNC", default=True)
            if wayback_async is None
            else bool(wayback_async)
        )
        self.wayback_timeout_seconds = self._env_float(
            "STEMMA_WAYBACK_TIMEOUT_SECONDS",
            default=_WAYBACK_DEFAULT_TIMEOUT_SECONDS,
        )
        self.wayback_probe_timeout_seconds = self._env_float(
            "STEMMA_WAYBACK_PROBE_TIMEOUT_SECONDS",
            default=_WAYBACK_PROBE_TIMEOUT_SECONDS,
        )
        self.wayback_probe_interval_seconds = self._env_float(
            "STEMMA_WAYBACK_PROBE_INTERVAL_SECONDS",
            default=_WAYBACK_PROBE_INTERVAL_SECONDS,
        )
        self.wayback_suspend_seconds = self._env_float(
            "STEMMA_WAYBACK_SUSPEND_SECONDS",
            default=_WAYBACK_SUSPEND_SECONDS,
        )
        self.wayback_workers = self._env_positive_int(
            "STEMMA_WAYBACK_WORKERS",
            default=_WAYBACK_DEFAULT_WORKERS,
        )
        self._wayback_executor: ThreadPoolExecutor | None = None
        self._wayback_lock = threading.Lock()
        self._wayback_probe_checked_epoch = 0.0
        self._wayback_probe_reachable = False
        self._wayback_probe_resolved_ip: str | None = None
        self._wayback_probe_error: str | None = None
        self._wayback_suspend_until_epoch = 0.0
        self._wayback_suspend_reason: str | None = None
        if self.wayback_enabled and self.wayback_async and self.wayback_workers > 0:
            self._wayback_executor = ThreadPoolExecutor(
                max_workers=self.wayback_workers,
                thread_name_prefix="stemma-wayback",
            )
            atexit.register(self._shutdown_wayback_executor)

    def ingest_file(
        self,
        file_path: Path,
        source_uri: str | None = None,
        source_paths: Sequence[Path | str] | None = None,
    ) -> IngestResult:
        path = file_path.expanduser().resolve()
        if not path.exists() or not path.is_file():
            raise ResourceIngestError(f"File not found: {path}")
        try:
            source_size = int(path.stat().st_size)
        except OSError as exc:
            raise ResourceIngestError(f"Unable to inspect file size for {path}: {exc}") from exc
        if source_size <= 0:
            raise EmptySourceFileError(f"Zero-byte source file cannot be ingested: {path}")

        discovered_urls = self._discover_download_urls(path, source_uri, source_paths=source_paths)
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

    def _discover_download_urls(
        self,
        path: Path,
        source_uri: str | None,
        *,
        source_paths: Sequence[Path | str] | None = None,
    ) -> list[str]:
        urls: list[str] = []
        for candidate in self._candidate_source_paths(path, source_paths):
            urls.extend(self._read_macos_wherefroms(candidate))
            quarantine_url = self._read_macos_quarantine_url(candidate)
            if quarantine_url:
                urls.append(quarantine_url)
            urls.extend(self._read_linux_origin_urls(candidate))
        if self._looks_like_network_url(source_uri):
            urls.append(source_uri.strip())
        return self._dedupe_urls(urls)

    @staticmethod
    def _candidate_source_paths(
        primary_path: Path,
        source_paths: Sequence[Path | str] | None,
    ) -> list[Path]:
        candidates: list[Path] = []
        seen: set[str] = set()
        raw_values: list[Path | str] = [primary_path]
        if source_paths:
            raw_values.extend(source_paths)
        for value in raw_values:
            try:
                resolved = Path(value).expanduser().resolve()
            except Exception:
                continue
            if not resolved.exists() or not resolved.is_file():
                continue
            key = str(resolved)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(resolved)
        return candidates

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
        if not self._is_macos():
            return []
        raw = self._read_xattr_bytes(path, _MACOS_WHEREFROMS_ATTR)
        urls = self._parse_raw_xattr_urls(raw) if raw is not None else []
        if urls:
            return urls
        return self._read_macos_mdls_wherefroms(path)

    @staticmethod
    def _read_macos_mdls_wherefroms(path: Path) -> list[str]:
        try:
            proc = subprocess.run(
                ["mdls", "-name", "kMDItemWhereFroms", "-plist", str(path)],
                check=False,
                capture_output=True,
            )
        except Exception:
            return []
        if proc.returncode != 0:
            return []
        stdout = bytes(proc.stdout or b"")
        if not stdout:
            return []
        try:
            parsed = plistlib.loads(stdout)
        except Exception:
            decoded = stdout.decode("utf-8", errors="ignore")
            return IngestionService._extract_network_urls_from_text(decoded)

        values = parsed.get("kMDItemWhereFroms") if isinstance(parsed, dict) else parsed
        if isinstance(values, str):
            return IngestionService._dedupe_urls([values])
        if isinstance(values, list):
            flattened: list[str] = []
            for item in values:
                if isinstance(item, str):
                    flattened.append(item)
                elif isinstance(item, bytes):
                    flattened.extend(IngestionService._parse_raw_xattr_urls(item))
            return IngestionService._dedupe_urls(flattened)
        return []

    def _read_linux_origin_urls(self, path: Path) -> list[str]:
        urls: list[str] = []
        for attr in _LINUX_ORIGIN_URL_ATTRS:
            raw = self._read_xattr_bytes(path, attr)
            if raw is None:
                continue
            urls.extend(self._parse_raw_xattr_urls(raw))
        return urls

    def _read_macos_quarantine_url(self, path: Path) -> str | None:
        if not self._is_macos():
            return None
        raw = self._read_xattr_bytes(path, _MACOS_QUARANTINE_ATTR)
        if raw is None:
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
        if not self._wayback_ready_for_attempt():
            return

        if self._wayback_executor is not None:
            try:
                self._wayback_executor.submit(self._register_with_wayback_request, source_url)
                return
            except RuntimeError:
                # Executor may be shutting down; fall back to inline best-effort call.
                pass
        self._register_with_wayback_request(source_url)

    def _register_with_wayback_request(self, source_url: str) -> None:
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
        except HTTPError as exc:
            if int(getattr(exc, "code", 0) or 0) == 429:
                self._suspend_wayback(
                    reason=f"Wayback returned 429 for {source_url}",
                    warning=f"Wayback rate-limited snapshot registration for {source_url}; pausing attempts temporarily.",
                )
                return
            logger.warning("Wayback snapshot registration failed for %s: %s", source_url, exc)
        except (URLError, TimeoutError, OSError) as exc:
            self._suspend_wayback(
                reason=f"{type(exc).__name__}: {exc}",
                warning=f"Wayback snapshot registration failed for {source_url}: {exc}",
            )
            logger.warning("Wayback snapshot registration failed for %s: %s", source_url, exc)

    def _shutdown_wayback_executor(self) -> None:
        executor = self._wayback_executor
        if executor is None:
            return
        self._wayback_executor = None
        try:
            executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass

    def shutdown(self) -> None:
        self._shutdown_wayback_executor()

    def wayback_diagnostics(self, *, force_probe: bool = False) -> dict[str, object]:
        self._probe_wayback_reachability(force=force_probe)
        with self._wayback_lock:
            checked_epoch = float(self._wayback_probe_checked_epoch or 0.0)
            suspend_until = float(self._wayback_suspend_until_epoch or 0.0)
            now_epoch = time.time()
            return {
                "enabled": bool(self.wayback_enabled),
                "async_enabled": bool(self.wayback_async),
                "save_prefix": _WAYBACK_SAVE_PREFIX,
                "probe_host": _WAYBACK_PROBE_HOST,
                "probe_port": _WAYBACK_PROBE_PORT,
                "probe_timeout_seconds": float(self.wayback_probe_timeout_seconds),
                "probe_interval_seconds": float(self.wayback_probe_interval_seconds),
                "request_timeout_seconds": float(self.wayback_timeout_seconds),
                "workers": int(self.wayback_workers),
                "reachable": bool(self._wayback_probe_reachable),
                "resolved_ip": self._wayback_probe_resolved_ip,
                "last_error": self._wayback_probe_error,
                "last_checked_at": now_utc_iso() if checked_epoch <= 0 else self._epoch_to_iso(checked_epoch),
                "suspended": suspend_until > now_epoch,
                "suspended_until": self._epoch_to_iso(suspend_until) if suspend_until > 0 else None,
                "suspend_reason": self._wayback_suspend_reason,
            }

    def _wayback_ready_for_attempt(self) -> bool:
        now_epoch = time.time()
        with self._wayback_lock:
            if now_epoch < float(self._wayback_suspend_until_epoch or 0.0):
                return False
        probe = self._probe_wayback_reachability(force=False)
        if not probe["reachable"]:
            reason = str(probe.get("error") or "reachability probe failed")
            self._suspend_wayback(
                reason=reason,
                warning=(
                    "Wayback reachability probe failed; pausing snapshot registration temporarily. "
                    f"reason={reason}"
                ),
            )
            return False
        return True

    def _probe_wayback_reachability(self, *, force: bool) -> dict[str, object]:
        now_epoch = time.time()
        with self._wayback_lock:
            checked = float(self._wayback_probe_checked_epoch or 0.0)
            interval = max(0.0, float(self.wayback_probe_interval_seconds or 0.0))
            if not force and checked > 0 and (now_epoch - checked) < interval:
                return {
                    "checked_epoch": checked,
                    "reachable": bool(self._wayback_probe_reachable),
                    "resolved_ip": self._wayback_probe_resolved_ip,
                    "error": self._wayback_probe_error,
                }
        resolved_ip: str | None = None
        error_text: str | None = None
        reachable = False
        timeout = max(0.1, float(self.wayback_probe_timeout_seconds or _WAYBACK_PROBE_TIMEOUT_SECONDS))
        try:
            addr_infos = socket.getaddrinfo(_WAYBACK_PROBE_HOST, _WAYBACK_PROBE_PORT, type=socket.SOCK_STREAM)
            if addr_infos:
                resolved_ip = str(addr_infos[0][4][0])
            with socket.create_connection((_WAYBACK_PROBE_HOST, _WAYBACK_PROBE_PORT), timeout=timeout):
                reachable = True
        except Exception as exc:
            error_text = f"{type(exc).__name__}: {exc}"
            reachable = False
        with self._wayback_lock:
            self._wayback_probe_checked_epoch = now_epoch
            self._wayback_probe_reachable = reachable
            self._wayback_probe_resolved_ip = resolved_ip
            self._wayback_probe_error = error_text
            if reachable:
                self._wayback_suspend_until_epoch = 0.0
                self._wayback_suspend_reason = None
            return {
                "checked_epoch": now_epoch,
                "reachable": reachable,
                "resolved_ip": resolved_ip,
                "error": error_text,
            }

    def _suspend_wayback(self, *, reason: str, warning: str) -> None:
        now_epoch = time.time()
        with self._wayback_lock:
            prior_suspend_until = float(self._wayback_suspend_until_epoch or 0.0)
            already_suspended = now_epoch < prior_suspend_until
            suspend_for = max(1.0, float(self.wayback_suspend_seconds or _WAYBACK_SUSPEND_SECONDS))
            self._wayback_suspend_until_epoch = max(prior_suspend_until, now_epoch + suspend_for)
            self._wayback_suspend_reason = reason
            if already_suspended:
                return
        logger.warning(
            "%s (suspended for %.0fs)",
            warning,
            max(1.0, float(self.wayback_suspend_seconds or _WAYBACK_SUSPEND_SECONDS)),
        )

    @staticmethod
    def _epoch_to_iso(epoch_seconds: float) -> str:
        if epoch_seconds <= 0:
            return now_utc_iso()
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(epoch_seconds))

    @staticmethod
    def _looks_like_network_url(value: str | None) -> bool:
        lowered = IngestionService._normalize_url_candidate(value).lower()
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
            clean = IngestionService._normalize_url_candidate(url)
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
                candidate = IngestionService._normalize_url_candidate(token)
                if candidate:
                    values.append(candidate)
        if values:
            return values
        return IngestionService._extract_network_urls_from_text(decoded)

    @staticmethod
    def _extract_network_urls_from_text(text: str) -> list[str]:
        if not text:
            return []
        values: list[str] = []
        for match in _NETWORK_URL_RE.findall(text):
            clean = IngestionService._normalize_url_candidate(match)
            if clean:
                values.append(clean)
        return values

    @staticmethod
    def _normalize_url_candidate(value: str | None) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        text = text.strip().strip("\"'")
        text = text.strip("<>()[]{}")
        while text and text[-1] in {",", ";"}:
            text = text[:-1].rstrip()
        text = text.strip("\"'")
        return text
        return values

    def _read_xattr_bytes(self, path: Path, attr_name: str) -> bytes | None:
        if hasattr(os, "getxattr"):
            try:
                return os.getxattr(path, attr_name)
            except OSError:
                return None
        if self._is_macos():
            return self._read_macos_xattr_cli(path, attr_name)
        return None

    @staticmethod
    def _read_macos_xattr_cli(path: Path, attr_name: str) -> bytes | None:
        try:
            proc = subprocess.run(
                ["xattr", "-px", attr_name, str(path)],
                check=False,
                capture_output=True,
                text=True,
            )
        except Exception:
            return None
        if proc.returncode != 0:
            return None
        hex_text = "".join(str(proc.stdout or "").split()).strip()
        if not hex_text:
            return b""
        try:
            return bytes.fromhex(hex_text)
        except ValueError:
            return None

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

    @staticmethod
    def _env_positive_int(name: str, default: int) -> int:
        raw = os.getenv(name)
        if raw is None:
            return default
        try:
            parsed = int(raw)
        except (TypeError, ValueError):
            return default
        return parsed if parsed > 0 else default
