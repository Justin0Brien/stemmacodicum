#!/usr/bin/env python3
"""Recover missing download URLs for ingested resources.

Strategy (in order):
  Phase 0c – Catalog database lookup (/Volumes/X10/data/sources/catalog.sqlite)
              Match by SHA-256 digest; follow bib_object_links to bib_entries
              to retrieve url_normalized / doi_normalized.  Skipped when the
              catalog file is absent.  Fully deterministic — no heuristics.
  Phase 0 – Browser download histories (Safari, Arc/Chrome, Brave, Edge,
             Firefox) — matched by target path, filename + size, or filename +
             time proximity.  Highest-confidence source because these are
             verbatim records of what the browser fetched.
  Phase 1 – macOS extended attributes on the archived file
             (kMDItemWhereFroms + quarantine)
  Phase 2 – macOS extended attributes on the original source path
             (if source_uri points to a local file), or use source_uri
             directly if it is already an external URL
  Phase 3 – Scan ~/Downloads (and other common folders) for a file
             whose name matches original_filename, then read xattrs
  Phase 4 – Use the url / doi already recorded in any linked
             BibTeX reference entry (deterministic; no web search)
  Phase 5 – Scan JSON manifests on external drives for URL clues,
             matching only by exact filename (not title keywords)
  Phase 6a – Extract URLs embedded in PDF internal metadata
              (Title, Author, Subject, Keywords, first-page links)
  Phase 6b – Wayback Machine CDX API: SHA1 content-hash lookup
              (finds the exact file regardless of URL/filename)

Additional optional phases:
  Phase 7 – Extract DOI + URL candidates from latest extracted text
  Phase 8 – Lightweight web query using title/author/text hints
            (fallback when local metadata is insufficient)
"""
from __future__ import annotations

import argparse
import base64
import datetime
import hashlib
import html
import html.parser
import json
import os
import plistlib
import re
import shutil
import sqlite3
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError

try:
    import fitz  # PyMuPDF  # type: ignore
    HAVE_FITZ = True
except ImportError:
    HAVE_FITZ = False

from stemmacodicum.core.config import load_paths
from stemmacodicum.infrastructure.db.sqlite import get_connection

MACOS_WHEREFROMS_ATTR = "com.apple.metadata:kMDItemWhereFroms"
MACOS_QUARANTINE_ATTR = "com.apple.quarantine"
URL_RE = re.compile(r"https?://[^\s<>()\"']+", flags=re.IGNORECASE)
DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", flags=re.IGNORECASE)
SEARCH_RESULT_LINK_RE = re.compile(
    r"<a[^>]+class=[\"'][^\"']*result__a[^\"']*[\"'][^>]+href=[\"']([^\"']+)[\"']",
    flags=re.IGNORECASE,
)
SEARCH_BLOCK_HINTS = ("captcha", "cloudflare", "unusual traffic", "robot")

# Directories to search for original filename — shallow (no recursion).
SHALLOW_SEARCH_DIRS: list[Path] = [
    Path.home() / "Downloads",
    Path.home() / "Desktop",
    Path.home() / "Documents",
    Path.home() / "Library" / "Mobile Documents" / "com~apple~CloudDocs",  # iCloud Drive
    Path("/tmp"),
]

# Directories to search recursively (e.g. external drives with sub-folders).
# Overridable via --deep-search-dirs on the CLI.
DEEP_SEARCH_DIRS: list[Path] = [
    Path("/Volumes/X10/data"),
]

# External catalog database — used as Phase 0c for SHA-256-based URL lookup.
# If this path does not exist the phase is silently skipped.
CATALOG_DB: Path = Path("/Volumes/X10/data/sources/catalog.sqlite")


@dataclass(slots=True)
class ResourceTask:
    resource_id: str
    display_title: str | None
    original_filename: str
    source_uri: str | None
    download_url: str | None
    download_urls_json: str | None
    archived_relpath: str
    ref_title: str | None
    ref_author: str | None
    ref_url: str | None
    ref_doi: str | None


@dataclass(slots=True)
class BrowserDownloadRecord:
    """A single download record from a browser's history database."""
    browser: str        # "safari" | "chrome" | "brave" | "firefox" | etc.
    source_url: str     # URL the file was downloaded from
    target_path: str    # Local file path where the browser saved the file
    total_bytes: int    # Reported file size in bytes; -1 if unknown
    download_time: float  # Unix timestamp of download start; -1.0 if unknown
    referrer: str       # HTTP Referer URL (may be empty)


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------

def is_external_url(value: str | None) -> bool:
    text = str(value or "").strip().lower()
    return text.startswith("http://") or text.startswith("https://")


def looks_like_local_path(value: str | None) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    if is_external_url(text):
        return False
    return text.startswith("/") or text.startswith("file://") or text.startswith("~/")


def load_resource_tasks(db_path: Path) -> list[ResourceTask]:
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT
              r.id              AS resource_id,
              r.display_title   AS display_title,
              r.original_filename,
              r.source_uri,
              r.download_url,
              r.download_urls_json,
              r.archived_relpath,
              MIN(re.title)     AS ref_title,
              MIN(re.author)    AS ref_author,
              MIN(re.url)       AS ref_url,
              MIN(re.doi)       AS ref_doi
            FROM resources r
            LEFT JOIN reference_resources rr ON rr.resource_id = r.id
            LEFT JOIN reference_entries re ON re.id = rr.reference_id
            GROUP BY r.id
            ORDER BY r.ingested_at ASC
            """
        ).fetchall()
    out: list[ResourceTask] = []
    for row in rows:
        out.append(
            ResourceTask(
                resource_id=str(row["resource_id"]),
                display_title=row["display_title"],
                original_filename=str(row["original_filename"] or ""),
                source_uri=row["source_uri"],
                download_url=row["download_url"],
                download_urls_json=row["download_urls_json"],
                archived_relpath=str(row["archived_relpath"] or ""),
                ref_title=row["ref_title"],
                ref_author=row["ref_author"],
                ref_url=row["ref_url"],
                ref_doi=row["ref_doi"],
            )
        )
    return out


def load_resource_task_by_id(db_path: Path, resource_id: str) -> ResourceTask | None:
    with get_connection(db_path) as conn:
        row = conn.execute(
            """
            SELECT
              r.id              AS resource_id,
              r.display_title   AS display_title,
              r.original_filename,
              r.source_uri,
              r.download_url,
              r.download_urls_json,
              r.archived_relpath,
              MIN(re.title)     AS ref_title,
              MIN(re.author)    AS ref_author,
              MIN(re.url)       AS ref_url,
              MIN(re.doi)       AS ref_doi
            FROM resources r
            LEFT JOIN reference_resources rr ON rr.resource_id = r.id
            LEFT JOIN reference_entries re ON re.id = rr.reference_id
            WHERE r.id = ?
            GROUP BY r.id
            LIMIT 1
            """,
            (resource_id,),
        ).fetchone()
    if row is None:
        return None
    return ResourceTask(
        resource_id=str(row["resource_id"]),
        display_title=row["display_title"],
        original_filename=str(row["original_filename"] or ""),
        source_uri=row["source_uri"],
        download_url=row["download_url"],
        download_urls_json=row["download_urls_json"],
        archived_relpath=str(row["archived_relpath"] or ""),
        ref_title=row["ref_title"],
        ref_author=row["ref_author"],
        ref_url=row["ref_url"],
        ref_doi=row["ref_doi"],
    )


# ---------------------------------------------------------------------------
# macOS extended-attribute helpers
# ---------------------------------------------------------------------------

def read_macos_wherefroms(path: Path) -> list[str]:
    """Return URLs recorded in kMDItemWhereFroms xattr."""
    if sys.platform != "darwin" or not hasattr(os, "getxattr"):
        return []
    try:
        raw = os.getxattr(path, MACOS_WHEREFROMS_ATTR)
    except OSError:
        return []
    if not raw:
        return []
    try:
        parsed = plistlib.loads(raw)
    except Exception:
        return []
    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if is_external_url(str(item))]
    if isinstance(parsed, str) and is_external_url(parsed):
        return [parsed.strip()]
    return []


def read_all_xattr_urls(path: Path) -> list[str]:
    """Enumerate every extended attribute on *path* and return all URL strings found.

    This catches kMDItemWhereFroms, com.apple.quarantine, and any unknown
    URL-bearing attributes that tools like browsers or download managers may write.
    """
    if sys.platform != "darwin" or not hasattr(os, "listxattr"):
        return xattrs_for_path(path)
    found: list[str] = []
    seen: set[str] = set()

    def _add(url: str) -> None:
        u = url.strip()
        if is_external_url(u) and u not in seen:
            seen.add(u)
            found.append(u)

    # Always try the two well-known attributes first.
    for url in read_macos_wherefroms(path):
        _add(url)
    q = read_macos_quarantine_url(path)
    if q:
        _add(q)

    # Now scan every attribute we haven't already handled.
    known = {MACOS_WHEREFROMS_ATTR, MACOS_QUARANTINE_ATTR}
    try:
        attr_names = os.listxattr(path)
    except OSError:
        return found
    for name in attr_names:
        if name in known:
            continue
        try:
            raw = os.getxattr(path, name)
        except OSError:
            continue
        # Try plist first (some attrs are binary plists).
        try:
            parsed = plistlib.loads(raw)
            if isinstance(parsed, list):
                for item in parsed:
                    _add(str(item))
            elif isinstance(parsed, str):
                _add(parsed)
            continue
        except Exception:
            pass
        # Fall back to raw UTF-8 string scan.
        try:
            text = raw.decode("utf-8", errors="ignore")
        except Exception:
            continue
        for m in URL_RE.findall(text):
            _add(m.rstrip(".,);"))
    return found


def read_macos_quarantine_url(path: Path) -> str | None:
    """Extract the originating URL from the quarantine xattr, if present."""
    if sys.platform != "darwin" or not hasattr(os, "getxattr"):
        return None
    try:
        raw = os.getxattr(path, MACOS_QUARANTINE_ATTR)
    except OSError:
        return None
    text = raw.decode("utf-8", errors="ignore")
    # Format: 0001;xxxxxxxx;<appname>;<url>
    for token in reversed(text.split(";")):
        candidate = token.strip()
        if is_external_url(candidate):
            return candidate
    return None


def xattrs_for_path(path: Path) -> list[str]:
    """Return all download URL candidates from both xattr fields for a file."""
    found: list[str] = []
    found.extend(read_macos_wherefroms(path))
    q = read_macos_quarantine_url(path)
    if q:
        found.append(q)
    return found


def spotlight_find_by_name(filename: str, *, timeout: float = 15.0) -> list[Path]:
    """Use macOS Spotlight (mdfind) to locate files with this exact display name.

    This searches the entire indexed filesystem — including locations outside the
    preset shallow/deep search dirs — and works even if the file has been moved
    or is inside an app bundle or iCloud Drive sync folder.

    Returns a list of absolute Paths for all matches found, deduplicated.
    """
    if sys.platform != "darwin" or not filename:
        return []
    import subprocess
    # -name matches the filename component (kMDItemDisplayName) exactly.
    try:
        result = subprocess.run(
            ["mdfind", "-name", filename],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        lines = [l.strip() for l in result.stdout.splitlines() if l.strip()]
    except Exception:
        return []
    seen: set[Path] = set()
    matches: list[Path] = []
    for line in lines:
        p = Path(line)
        # mdfind -name may return dirs; skip them.
        if not p.is_file():
            continue
        # Only include files whose name actually matches (mdfind can be loose).
        if p.name != filename:
            continue
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            matches.append(p)
    return matches


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------

def merge_urls(existing_json: str | None, candidates: Iterable[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()

    if existing_json:
        try:
            parsed = json.loads(existing_json)
        except json.JSONDecodeError:
            parsed = []
        if isinstance(parsed, list):
            for item in parsed:
                clean = str(item).strip()
                if not is_external_url(clean) or clean in seen:
                    continue
                seen.add(clean)
                merged.append(clean)

    for item in candidates:
        clean = str(item).strip()
        if not is_external_url(clean) or clean in seen:
            continue
        seen.add(clean)
        merged.append(clean)
    return merged


def pick_primary(urls: list[str], fallback: str | None = None) -> str | None:
    for url in urls:
        if is_external_url(url):
            return url
    return fallback if is_external_url(fallback) else None


def save_resource_urls(db_path: Path, task: ResourceTask, urls: list[str]) -> None:
    if not urls:
        return
    primary = pick_primary(urls)
    urls_json = json.dumps(urls, ensure_ascii=True) if urls else None
    with get_connection(db_path) as conn:
        conn.execute(
            """
            UPDATE resources
            SET download_url = ?, download_urls_json = ?
            WHERE id = ?
            """,
            (primary, urls_json, task.resource_id),
        )
        conn.commit()


def load_latest_document_text(db_path: Path, resource_id: str, *, max_chars: int = 20000) -> str:
    with get_connection(db_path) as conn:
        row = conn.execute(
            """
            SELECT dt.text_content
            FROM document_texts dt
            INNER JOIN extraction_runs er ON er.id = dt.extraction_run_id
            WHERE dt.resource_id = ?
            ORDER BY er.created_at DESC
            LIMIT 1
            """,
            (resource_id,),
        ).fetchone()
    if row is None:
        return ""
    text = str(row["text_content"] or "")
    if max_chars > 0 and len(text) > max_chars:
        return text[:max_chars]
    return text


def normalize_doi(value: str | None) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    text = re.sub(r"^https?://(dx\.)?doi\.org/", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^doi:\s*", "", text, flags=re.IGNORECASE)
    match = DOI_RE.search(text)
    if not match:
        return None
    return match.group(0).strip().rstrip(".,);")


def doi_to_url(value: str | None) -> str | None:
    doi = normalize_doi(value)
    if not doi:
        return None
    return f"https://doi.org/{doi}"


def urls_from_text(text: str, *, max_urls: int = 12) -> list[str]:
    if not text:
        return []
    found: list[str] = []
    seen: set[str] = set()
    for raw in URL_RE.findall(text):
        clean = str(raw).strip().rstrip(".,);")
        if not is_external_url(clean):
            continue
        if clean in seen:
            continue
        seen.add(clean)
        found.append(clean)
        if len(found) >= max_urls:
            break
    return found


def doi_urls_from_text(text: str, *, max_dois: int = 8) -> list[str]:
    if not text:
        return []
    found: list[str] = []
    seen: set[str] = set()
    for raw in DOI_RE.findall(text):
        url = doi_to_url(raw)
        if not url or url in seen:
            continue
        seen.add(url)
        found.append(url)
        if len(found) >= max_dois:
            break
    return found


def query_terms_from_text(text: str, *, max_terms: int = 14) -> str:
    if not text:
        return ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    signal = ""
    for line in lines[:20]:
        if len(line) < 8:
            continue
        if line.lower().startswith("page "):
            continue
        signal = line
        break
    if not signal:
        signal = " ".join(lines[:2])
    signal = re.sub(r"[^a-zA-Z0-9\s\-_/]", " ", signal)
    tokens = [tok for tok in signal.split() if len(tok) > 2]
    return " ".join(tokens[:max_terms])


def build_web_search_queries(task: ResourceTask, text_excerpt: str) -> list[str]:
    queries: list[str] = []
    seen: set[str] = set()

    def add(query: str) -> None:
        clean = " ".join(str(query).strip().split())
        if not clean:
            return
        if clean.lower() in seen:
            return
        seen.add(clean.lower())
        queries.append(clean)

    title_hint = str(task.display_title or "").strip() or str(task.ref_title or "").strip()
    author_hint = str(task.ref_author or "").strip()
    filename_hint = Path(task.original_filename or "").stem.replace("_", " ").replace("-", " ").strip()
    text_hint = query_terms_from_text(text_excerpt)
    doi_hint = normalize_doi(task.ref_doi)

    if doi_hint:
        add(doi_hint)
    if title_hint and author_hint:
        add(f"{title_hint} {author_hint}")
    if title_hint:
        add(title_hint)
    if filename_hint and title_hint:
        add(f"{filename_hint} {title_hint}")
    elif filename_hint:
        add(filename_hint)
    if title_hint and text_hint:
        add(f"{title_hint} {text_hint}")
    elif text_hint:
        add(text_hint)

    return queries[:4]


def decode_search_result_url(raw_href: str) -> str | None:
    href = html.unescape(str(raw_href or "").strip())
    if not href:
        return None
    parsed = urllib.parse.urlparse(href)
    if parsed.scheme in {"http", "https"}:
        return href
    if href.startswith("/l/") or href.startswith("l/?"):
        if parsed.query:
            query = parsed.query
        elif "?" in href:
            query = href.split("?", 1)[1]
        else:
            query = ""
        params = urllib.parse.parse_qs(query)
        uddg = params.get("uddg", [])
        if uddg:
            decoded = urllib.parse.unquote(uddg[0])
            if is_external_url(decoded):
                return decoded
    return None


def web_search_urls(query: str, *, timeout: float = 25.0, max_results: int = 8) -> tuple[list[str], str | None]:
    if not query.strip():
        return [], None
    url = "https://duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "stemmacodicum-source-recovery/1.0",
                "Accept-Language": "en-US,en;q=0.9",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="ignore")
    except HTTPError as exc:
        if exc.code in {403, 429, 503}:
            return [], f"http_{exc.code}"
        return [], str(exc)
    except URLError as exc:
        return [], str(exc)
    except Exception as exc:
        return [], str(exc)

    lowered = body.lower()
    if any(marker in lowered for marker in SEARCH_BLOCK_HINTS):
        return [], "blocked"

    found: list[str] = []
    seen: set[str] = set()
    for match in SEARCH_RESULT_LINK_RE.findall(body):
        resolved = decode_search_result_url(match)
        if not resolved or not is_external_url(resolved):
            continue
        if "duckduckgo.com" in resolved:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        found.append(resolved)
        if len(found) >= max_results:
            break
    return found, None

# ---------------------------------------------------------------------------
# Phase 0: Browser download history (Safari, Chromium variants, Firefox)
# ---------------------------------------------------------------------------

# Safari stores Mac Absolute Time: seconds since 2001-01-01 00:00:00 UTC
_MAC_EPOCH_OFFSET: float = 978_307_200.0
# Chrome/Chromium store microseconds since 1601-01-01 00:00:00 UTC
_CHROME_EPOCH_OFFSET: float = 11_644_473_600.0

# Generous matching window — files may have been moved/copied after downloading.
_BROWSER_TIME_WINDOW: float = 14 * 24 * 3600.0  # 14 days

# Chromium-based browsers: (label, base user-data dir)
# Arc stores its profiles under Google/Chrome — intentional.
_CHROMIUM_BROWSER_DIRS: list[tuple[str, Path]] = [
    ("arc/chrome", Path.home() / "Library" / "Application Support" / "Google" / "Chrome"),
    ("chromium",   Path.home() / "Library" / "Application Support" / "Chromium"),
    ("brave",      Path.home() / "Library" / "Application Support" / "BraveSoftware" / "Brave-Browser"),
    ("edge",       Path.home() / "Library" / "Application Support" / "Microsoft Edge"),
    ("edge-beta",  Path.home() / "Library" / "Application Support" / "Microsoft Edge Beta"),
    ("edge-dev",   Path.home() / "Library" / "Application Support" / "Microsoft Edge Dev"),
    ("edge-canary",Path.home() / "Library" / "Application Support" / "Microsoft Edge Canary"),
    ("vivaldi",    Path.home() / "Library" / "Application Support" / "Vivaldi"),
    ("opera",      Path.home() / "Library" / "Application Support" / "com.operasoftware.Opera"),
]


# Maps id(connection) -> tmp file path so we can delete it on close.
# Python 3.14 sqlite3.Connection uses __slots__ and rejects arbitrary attrs.
_sqlite_tmp_paths: dict[int, str] = {}


def _open_sqlite_copy(db_path: Path) -> "sqlite3.Connection | None":
    """Open a SQLite DB by copying it to a temp file first.

    This avoids "database is locked" errors when the browser is running.
    The caller is responsible for calling _close_sqlite_copy() when done.
    """
    if not db_path.exists():
        return None
    tmp_name: str | None = None
    try:
        tmp_fd, tmp_name = tempfile.mkstemp(suffix=".db")
        os.close(tmp_fd)
        shutil.copy2(db_path, tmp_name)
        conn = sqlite3.connect(tmp_name)
        conn.row_factory = sqlite3.Row
        _sqlite_tmp_paths[id(conn)] = tmp_name
        return conn
    except Exception:
        if tmp_name:
            try:
                os.unlink(tmp_name)
            except Exception:
                pass
        return None


def _close_sqlite_copy(conn: "sqlite3.Connection") -> None:
    tmp_path = _sqlite_tmp_paths.pop(id(conn), None)
    try:
        conn.close()
    except Exception:
        pass
    if tmp_path:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass


def load_safari_downloads() -> list[BrowserDownloadRecord]:
    """Load download records from Safari's Downloads.plist."""
    plist_path = Path.home() / "Library" / "Safari" / "Downloads.plist"
    if not plist_path.exists():
        return []
    try:
        with open(plist_path, "rb") as fh:
            data = plistlib.load(fh)
    except Exception:
        return []
    records: list[BrowserDownloadRecord] = []
    for entry in data.get("DownloadHistory", []):
        url = str(entry.get("DownloadEntryURL") or "").strip()
        if not is_external_url(url):
            continue
        target = str(entry.get("DownloadEntryPath") or "").strip()
        size = int(entry.get("DownloadEntryProgressTotalToLoad") or -1)
        # Date is an NSDate stored as a Mac Absolute Time float or a datetime.
        ts_raw = entry.get("DownloadEntryDateAddedKey")
        ts: float = -1.0
        if isinstance(ts_raw, datetime.datetime):
            ts = ts_raw.timestamp()
        elif isinstance(ts_raw, (int, float)):
            ts = float(ts_raw) + _MAC_EPOCH_OFFSET
        records.append(BrowserDownloadRecord(
            browser="safari",
            source_url=url,
            target_path=target,
            total_bytes=size,
            download_time=ts,
            referrer="",
        ))
    return records


# SQL for modern Chromium: original URL lives in downloads_url_chains (chain_index=0)
_CHROMIUM_QUERY_CHAINS = """
    SELECT d.target_path, duc.url AS original_url, d.referrer,
           d.total_bytes, d.start_time, d.tab_url
    FROM downloads d
    LEFT JOIN downloads_url_chains duc ON duc.id = d.id AND duc.chain_index = 0
    WHERE duc.url IS NOT NULL OR d.tab_url IS NOT NULL
"""
# SQL for older Chromium: original_url was a direct column on downloads
_CHROMIUM_QUERY_LEGACY = """
    SELECT target_path, original_url, referrer, total_bytes, start_time, tab_url
    FROM downloads
    WHERE (original_url IS NOT NULL AND original_url != '')
       OR (tab_url IS NOT NULL AND tab_url != '')
"""


def load_chromium_downloads() -> list[BrowserDownloadRecord]:
    """Load download records from all Chromium-based browsers (Arc/Chrome, Brave, Edge, etc.).

    Handles both modern Chromium (original URL in downloads_url_chains) and
    legacy Chromium (original_url column directly on downloads).
    """
    records: list[BrowserDownloadRecord] = []
    seen_profile_paths: set[Path] = set()
    for browser_name, base_dir in _CHROMIUM_BROWSER_DIRS:
        if not base_dir.is_dir():
            continue
        # Each profile subdirectory may contain its own History DB.
        # Exclude snapshot/cache directories to avoid duplicates.
        for history_path in sorted(base_dir.glob("*/History")):
            real = history_path.resolve()
            if real in seen_profile_paths:
                continue
            if "Snapshots" in history_path.parts:
                continue  # skip snapshot copies
            seen_profile_paths.add(real)
            conn = _open_sqlite_copy(history_path)
            if conn is None:
                continue
            try:
                # Try modern query first (downloads_url_chains), fall back to legacy.
                rows = None
                try:
                    rows = conn.execute(_CHROMIUM_QUERY_CHAINS).fetchall()
                except Exception:
                    pass
                if rows is None:
                    try:
                        rows = conn.execute(_CHROMIUM_QUERY_LEGACY).fetchall()
                    except Exception:
                        continue
                for row in rows:
                    url = str(row["original_url"] or "").strip()
                    if not is_external_url(url):
                        tab = str(row["tab_url"] or "").strip()
                        if is_external_url(tab):
                            url = tab
                        else:
                            continue
                    raw_ts = row["start_time"] or 0
                    ts = (int(raw_ts) / 1_000_000.0) - _CHROME_EPOCH_OFFSET if raw_ts else -1.0
                    records.append(BrowserDownloadRecord(
                        browser=browser_name,
                        source_url=url,
                        target_path=str(row["target_path"] or "").strip(),
                        total_bytes=int(row["total_bytes"] or -1),
                        download_time=ts,
                        referrer=str(row["referrer"] or "").strip(),
                    ))
            finally:
                _close_sqlite_copy(conn)
    return records


def load_firefox_downloads() -> list[BrowserDownloadRecord]:
    """Load download records from all Firefox profiles via places.sqlite."""
    records: list[BrowserDownloadRecord] = []
    profiles_dir = (
        Path.home() / "Library" / "Application Support" / "Firefox" / "Profiles"
    )
    if not profiles_dir.is_dir():
        return []
    for places_path in sorted(profiles_dir.glob("*/places.sqlite")):
        conn = _open_sqlite_copy(places_path)
        if conn is None:
            continue
        try:
            # Firefox stores the destination as a file:// annotation on the
            # download source URL entry in moz_places.
            try:
                # Firefox stores two `downloads/metaData` annotations per
                # download: one old-style bare file:// path and one JSON blob
                # with fileSize/endTime.  A plain GROUP BY p.id picks
                # arbitrarily, so we use correlated subqueries to force
                # selection of the correct row for each purpose.
                rows = conn.execute(
                    """
                    SELECT
                        p.url AS source_url,
                        MAX(h.visit_date) AS visit_date,
                        -- Canonical destination URI (downloads/destinationFileURI)
                        (SELECT ma.content
                           FROM moz_annos ma
                           JOIN moz_anno_attributes maa ON maa.id = ma.anno_attribute_id
                          WHERE ma.place_id = p.id
                            AND maa.name = 'downloads/destinationFileURI'
                          LIMIT 1
                        ) AS dest_uri,
                        -- Fallback: old-style file:// path stored in metaData
                        (SELECT ma.content
                           FROM moz_annos ma
                           JOIN moz_anno_attributes maa ON maa.id = ma.anno_attribute_id
                          WHERE ma.place_id = p.id
                            AND maa.name = 'downloads/metaData'
                            AND ma.content LIKE 'file://%'
                          LIMIT 1
                        ) AS dest_uri_fallback,
                        -- JSON metadata (has fileSize) — must start with '{'
                        (SELECT ma.content
                           FROM moz_annos ma
                           JOIN moz_anno_attributes maa ON maa.id = ma.anno_attribute_id
                          WHERE ma.place_id = p.id
                            AND maa.name = 'downloads/metaData'
                            AND ma.content LIKE '{%'
                          LIMIT 1
                        ) AS meta_json
                    FROM moz_places p
                    JOIN moz_historyvisits h ON h.place_id = p.id
                    -- Must have at least one download annotation to count as a download
                    JOIN moz_annos dl_check ON dl_check.place_id = p.id
                    JOIN moz_anno_attributes dl_attr
                         ON dl_attr.id = dl_check.anno_attribute_id
                         AND dl_attr.name IN ('downloads/destinationFileURI',
                                              'downloads/metaData')
                    WHERE p.url LIKE 'http%'
                    GROUP BY p.id
                    """
                ).fetchall()
            except Exception:
                continue
            for row in rows:
                url = str(row["source_url"] or "").strip()
                if not is_external_url(url):
                    continue
                # Prefer the canonical destinationFileURI; fall back to the
                # old-format file:// path stored under downloads/metaData.
                dest_uri = str(row["dest_uri"] or row["dest_uri_fallback"] or "").strip()
                local_path = ""
                if dest_uri.startswith("file://"):
                    local_path = urllib.parse.unquote(dest_uri[7:])
                # visit_date is microseconds since Unix epoch in Firefox
                ts_raw = row["visit_date"] or 0
                ts = float(ts_raw) / 1_000_000.0 if ts_raw else -1.0
                # File size from the JSON metaData blob
                file_size = -1
                meta_raw = row["meta_json"]
                if meta_raw:
                    try:
                        meta = json.loads(meta_raw)
                        file_size = int(meta.get("fileSize", -1))
                    except Exception:
                        pass
                records.append(BrowserDownloadRecord(
                    browser="firefox",
                    source_url=url,
                    target_path=local_path,
                    total_bytes=file_size,
                    download_time=ts,
                    referrer="",
                ))
        finally:
            _close_sqlite_copy(conn)
    return records


# Module-level cache so we only query the browser DBs once per process run.
_browser_download_cache: list[BrowserDownloadRecord] | None = None


def load_all_browser_downloads(*, force_reload: bool = False) -> list[BrowserDownloadRecord]:
    """Return a merged list of download records from all available browsers.

    Results are cached for the lifetime of the process.  Pass *force_reload=True*
    to discard the cache and re-read from disk (useful if browsers are running).
    """
    global _browser_download_cache
    if _browser_download_cache is not None and not force_reload:
        return _browser_download_cache
    records: list[BrowserDownloadRecord] = []
    records.extend(load_safari_downloads())
    records.extend(load_chromium_downloads())
    records.extend(load_firefox_downloads())
    _browser_download_cache = records
    return records


def _browser_match_score(
    record: BrowserDownloadRecord,
    *,
    original_filename: str,
    archive_size: int | None,
    source_uri: str | None,
    archive_ctime: float | None,
) -> int:
    """Score how well a download record matches a resource (0 = no match, 100 = perfect).

    Scoring tiers:
      100 – exact target_path matches source_uri (the original download path)
       90 – basename match + size match + time within window
       80 – basename match + size match
       65 – extension match + exact size match (file was renamed; no name match)
       60 – basename match + time within window
       40 – basename match only (weak; accepted as last resort)
        0 – no filename/extension+size match
    """
    rec_path = Path(record.target_path) if record.target_path else None
    rec_name = rec_path.name if rec_path else ""

    # Exact source_uri path match — strongest possible signal.
    if source_uri and record.target_path:
        src_path = str(source_uri).replace("file://", "").strip()
        if src_path and src_path == record.target_path:
            return 100

    # Filename must match (case-insensitive) for any positive score.
    if not rec_name or rec_name.lower() != original_filename.lower():
        # Fallback: match by file extension + exact byte size.
        # Handles files that were renamed after downloading (or before archiving).
        # Exact byte-level size match is a strong signal even without a name match.
        orig_ext = Path(original_filename).suffix.lower() if original_filename else ""
        rec_ext = rec_path.suffix.lower() if rec_path else ""
        if (
            orig_ext
            and rec_ext == orig_ext
            and archive_size is not None
            and record.total_bytes > 0
            and archive_size == record.total_bytes
        ):
            return 65  # extension + exact size: confident despite name mismatch
        return 0

    size_match = (
        archive_size is not None
        and record.total_bytes > 0
        and archive_size == record.total_bytes
    )
    time_match = (
        archive_ctime is not None
        and record.download_time > 0
        and abs(archive_ctime - record.download_time) <= _BROWSER_TIME_WINDOW
    )

    if size_match and time_match:
        return 90
    if size_match:
        return 80
    if time_match:
        return 60
    return 40  # filename-only weak match


def browser_history_urls_for_file(
    *,
    original_filename: str,
    archive_size: int | None,
    source_uri: str | None,
    archive_file: Path,
    browser_records: list[BrowserDownloadRecord],
    min_score: int = 40,
) -> list[str]:
    """Return source URLs from browser history that match the given resource file.

    Candidates are ranked by match score; only those meeting *min_score* are returned.
    The list is deduplicated and ordered best-first.
    """
    if not original_filename:
        return []

    archive_ctime: float | None = None
    try:
        st = archive_file.stat()
        # st_birthtime is the creation time on macOS (not available on Linux).
        archive_ctime = getattr(st, "st_birthtime", None) or st.st_mtime
    except OSError:
        pass

    scored: list[tuple[int, str]] = []
    seen_urls: set[str] = set()

    for record in browser_records:
        score = _browser_match_score(
            record,
            original_filename=original_filename,
            archive_size=archive_size,
            source_uri=source_uri,
            archive_ctime=archive_ctime,
        )
        if score < min_score:
            continue
        url = record.source_url
        if url in seen_urls:
            continue
        seen_urls.add(url)
        scored.append((score, url))

    # Sort best score first; stable so equal scores preserve insertion order.
    scored.sort(key=lambda t: t[0], reverse=True)
    return [url for _, url in scored]


# ---------------------------------------------------------------------------
# Phase 3: search common download folders
# ---------------------------------------------------------------------------

def find_copies_by_filename(
    original_filename: str,
    *,
    expected_size: int | None = None,
    shallow_dirs: list[Path] | None = None,
    deep_dirs: list[Path] | None = None,
) -> list[Path]:
    """Find files matching original_filename in shallow and/or deep directories.

    Shallow dirs are searched non-recursively (one level only).
    Deep dirs are searched recursively with rglob.

    When expected_size is given, only files whose byte size matches are returned;
    size mismatches are silently skipped (different file with the same name).
    """
    if not original_filename:
        return []
    if shallow_dirs is None:
        shallow_dirs = SHALLOW_SEARCH_DIRS
    if deep_dirs is None:
        deep_dirs = DEEP_SEARCH_DIRS

    seen: set[Path] = set()
    matches: list[Path] = []

    def _accept(p: Path) -> bool:
        if not p.is_file():
            return False
        if expected_size is not None:
            try:
                if p.stat().st_size != expected_size:
                    return False
            except OSError:
                return False
        return True

    def _add(p: Path) -> None:
        rp = p.resolve()
        if rp not in seen and _accept(p):
            seen.add(rp)
            matches.append(p)

    # Shallow search — direct child only.
    for base in shallow_dirs:
        if not base.is_dir():
            continue
        _add(base / original_filename)
        try:
            for item in base.iterdir():
                if item.name == original_filename:
                    _add(item)
        except PermissionError:
            pass

    # Deep (recursive) search.
    for base in deep_dirs:
        if not base.is_dir():
            continue
        try:
            for item in base.rglob(original_filename):
                _add(item)
        except PermissionError:
            pass

    return matches


# ---------------------------------------------------------------------------
# Phase 4: reference entry URL / DOI (deterministic — no web search)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Phase 0c: catalog database lookup (SHA-256 → bib_entries url/doi)
# ---------------------------------------------------------------------------

def catalog_urls_for_sha256(sha256: str | None, catalog_db: Path = CATALOG_DB) -> list[str]:
    """Return URL candidates from the external catalog database.

    Looks up *sha256* in ``objects``, follows ``bib_object_links`` →
    ``bib_entries`` and returns every non-empty ``url_normalized`` and
    ``doi_normalized`` value found.  Returns an empty list when the catalog
    file does not exist or the digest is not present.
    """
    if not sha256 or not catalog_db.exists():
        return []
    results: list[str] = []
    seen: set[str] = set()
    try:
        conn = sqlite3.connect(f"file:{catalog_db}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                """
                SELECT be.url_normalized, be.doi_normalized
                FROM   objects           AS obj
                JOIN   bib_object_links  AS bol USING (object_id)
                JOIN   bib_entries       AS be  USING (bib_key)
                WHERE  obj.sha256 = ?
                """,
                (sha256.lower(),),
            ).fetchall()
        finally:
            conn.close()
    except Exception:
        return []
    for row in rows:
        url = str(row["url_normalized"] or "").strip()
        doi = str(row["doi_normalized"] or "").strip()
        if is_external_url(url) and url not in seen:
            seen.add(url)
            results.append(url)
        if doi and doi not in seen:
            seen.add(doi)
            # Normalise raw DOI to a resolvable URL.
            clean = doi.lstrip("https://doi.org/").lstrip("http://doi.org/").lstrip("doi:")
            doi_url = f"https://doi.org/{clean}"
            if doi_url not in seen:
                seen.add(doi_url)
                results.append(doi_url)
    return results


def url_from_reference(task: ResourceTask) -> list[str]:
    """Return URL(s) directly recorded in a linked reference entry.

    Only uses what is already in the database — never performs a web search.
    A DOI is converted to its canonical doi.org resolver URL.
    """
    results: list[str] = []
    if is_external_url(task.ref_url):
        results.append(str(task.ref_url).strip())
    if task.ref_doi:
        doi = str(task.ref_doi).strip()
        doi = doi.lstrip("https://doi.org/").lstrip("http://doi.org/").lstrip("doi:")
        if doi:
            results.append(f"https://doi.org/{doi}")
    return results


# ---------------------------------------------------------------------------
# Phase 5: manifest JSON scan (filename-exact match only)
# ---------------------------------------------------------------------------

def manifest_urls_for_task(
    manifest_root: Path,
    *,
    original_filename: str,
    resource_id: str,
    max_files: int = 25000,
) -> list[str]:
    """Scan JSON manifests for URLs that appear alongside the exact filename.

    Only matches by exact filename — title/author keywords are deliberately
    excluded to avoid assigning URLs from unrelated documents.
    """
    if not manifest_root.is_dir():
        return []
    if not original_filename:
        return []
    clue = original_filename.lower()

    results: list[str] = []
    seen: set[str] = set()
    scanned = 0
    for json_path in manifest_root.rglob("*.json"):
        scanned += 1
        if scanned > max_files:
            break
        try:
            text = json_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if clue not in text.lower():
            continue
        for match in URL_RE.findall(text):
            clean = match.strip().rstrip(".,);")
            if not is_external_url(clean) or clean in seen:
                continue
            seen.add(clean)
            results.append(clean)
            if len(results) >= 8:
                return results
    return results


# ---------------------------------------------------------------------------
# Phase 6a: HTML canonical/og:url meta tag extraction
# ---------------------------------------------------------------------------

class _CanonicalURLParser(html.parser.HTMLParser):
    """Minimal SAX-style HTML parser that extracts self-referential URL tags."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.urls: list[str] = []
        self._seen: set[str] = set()

    def _add(self, url: str | None) -> None:
        if not url:
            return
        u = url.strip()
        if is_external_url(u) and u not in self._seen:
            self._seen.add(u)
            self.urls.append(u)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        amap = {k.lower(): (v or "") for k, v in attrs}
        tag = tag.lower()
        if tag == "link":
            rel = amap.get("rel", "").lower()
            if "canonical" in rel:
                self._add(amap.get("href"))
        elif tag == "meta":
            prop = amap.get("property", "").lower()
            name = amap.get("name", "").lower()
            content = amap.get("content", "")
            if prop in ("og:url", "twitter:url") or name in ("twitter:url",):
                self._add(content)
            # <meta http-equiv="refresh" content="0; url=https://...">
            if amap.get("http-equiv", "").lower() == "refresh" and "url=" in content.lower():
                idx = content.lower().find("url=")
                self._add(content[idx + 4:].strip().strip("'\""))
        elif tag == "base":
            self._add(amap.get("href"))


def html_canonical_urls(archive_file: Path, *, max_bytes: int = 256 * 1024) -> list[str]:
    """Extract canonical/og:url/twitter:url from a saved HTML file.

    Reads up to *max_bytes* of the file (the relevant tags are always in
    <head>, well within the first 256 KB). Returns [] for non-HTML files.
    Never performs any network request.
    """
    suffix = archive_file.suffix.lower()
    if suffix not in (".html", ".htm", ".xhtml"):
        return []
    try:
        raw = archive_file.read_bytes()[:max_bytes]
        text = raw.decode("utf-8", errors="ignore")
    except OSError:
        return []
    parser = _CanonicalURLParser()
    try:
        parser.feed(text)
    except Exception:
        pass
    return parser.urls


# ---------------------------------------------------------------------------
# Phase 6a (PDF): PDF internal metadata / embedded URLs
# ---------------------------------------------------------------------------

def pdf_metadata_urls(archive_file: Path) -> list[str]:
    """Extract URLs from PDF document-info metadata and first-page links.

    Uses PyMuPDF (fitz) if available; returns [] otherwise.
    Never performs any network request.
    """
    if not HAVE_FITZ:
        return []
    results: list[str] = []
    seen: set[str] = set()
    try:
        doc = fitz.open(str(archive_file))
    except Exception:
        return []
    try:
        # Document info dict: Title, Author, Subject, Keywords, Creator, etc.
        meta = doc.metadata or {}
        for field in ("title", "author", "subject", "keywords", "creator", "producer"):
            val = str(meta.get(field) or "").strip()
            for m in URL_RE.findall(val):
                url = m.rstrip(".,);")  
                if is_external_url(url) and url not in seen:
                    seen.add(url)
                    results.append(url)

        # XMP metadata (may contain originating URL)
        xmp = doc.get_xml_metadata() or ""
        for m in URL_RE.findall(xmp):
            url = m.rstrip(".,);")
            if is_external_url(url) and url not in seen:
                seen.add(url)
                results.append(url)

        # Hyperlinks on the first page (annotations + text URIs)
        if doc.page_count > 0:
            page = doc[0]
            for link in page.get_links():
                uri = str(link.get("uri") or "").strip()
                if is_external_url(uri) and uri not in seen:
                    seen.add(uri)
                    results.append(uri)
    except Exception:
        pass
    finally:
        doc.close()
    return results


# ---------------------------------------------------------------------------
# Phase 6b: Wayback Machine CDX SHA1 hash lookup
# ---------------------------------------------------------------------------

WAYBACK_CDX_ENDPOINT = "https://web.archive.org/cdx/search/cdx"
WAYBACK_CDX_DELAY = 1.0  # seconds between requests — stay polite


def _sha1_b32(path: Path) -> str:
    """Return the Base32-encoded SHA1 digest of a file (Wayback's hash format)."""
    h = hashlib.sha1(usedforsecurity=False)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return base64.b32encode(h.digest()).decode("ascii")


def wayback_cdx_sha1_lookup(
    archive_file: Path,
    *,
    delay: float = WAYBACK_CDX_DELAY,
    verbose: bool = True,
) -> list[str]:
    """Query the Wayback CDX API by SHA1 hash of the file content.

    This finds the exact file regardless of URL path or filename,
    because Wayback records the SHA1 of every captured response body.
    Returns original pre-archival URLs; never returns web.archive.org links.
    """
    try:
        sha1 = _sha1_b32(archive_file)
    except OSError as exc:
        if verbose:
            print(f"  [6b] cannot hash file: {exc}")
        return []

    if verbose:
        print(f"  [6b] Wayback CDX SHA1 lookup: {sha1}")

    # Build query manually: keep `url=*` unencoded — urlencode would turn * into
    # %2A which the CDX API rejects with HTTP 400.
    other_params = urllib.parse.urlencode({
        "filter": f"sha1:{sha1}",
        "output": "json",
        "fl": "original,mimetype,timestamp",
        "collapse": "original",
        "limit": "10",
    })
    api_url = f"{WAYBACK_CDX_ENDPOINT}?url=*&{other_params}"
    try:
        req = urllib.request.Request(
            api_url,
            headers={"User-Agent": "stemmacodicum-url-recovery/1.0"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
    except HTTPError as exc:
        if exc.code == 400:
            # CDX returned 400 — SHA1 not indexed or global wildcard unsupported.
            if verbose:
                print("  [6b] no Wayback captures found for this file hash (CDX 400)")
        else:
            if verbose:
                print(f"  [6b] CDX request failed: {exc}")
        if delay:
            time.sleep(delay)
        return []
    except Exception as exc:
        if verbose:
            print(f"  [6b] CDX request failed: {exc}")
        if delay:
            time.sleep(delay)
        return []

    if delay:
        time.sleep(delay)

    try:
        rows = json.loads(raw)
    except json.JSONDecodeError:
        return []

    if not rows or len(rows) < 2:
        if verbose:
            print("  [6b] no Wayback captures found for this file hash")
        return []

    seen: set[str] = set()
    results: list[str] = []
    for row in rows[1:]:
        if not isinstance(row, list) or not row:
            continue
        original_url = str(row[0]).strip().rstrip(",")
        if not is_external_url(original_url) or original_url in seen:
            continue
        if "web.archive.org" in original_url:
            continue
        seen.add(original_url)
        results.append(original_url)
    return results


# ---------------------------------------------------------------------------
# Already-attempted tracking (avoids re-hammering Wayback on reruns)
# ---------------------------------------------------------------------------

def load_attempted(state_path: Path) -> set[str]:
    """Load the set of resource_ids already attempted for Wayback lookup."""
    if not state_path.exists():
        return set()
    try:
        return set(json.loads(state_path.read_text(encoding="utf-8")))
    except Exception:
        return set()


def save_attempted(state_path: Path, attempted: set[str]) -> None:
    state_path.write_text(
        json.dumps(sorted(attempted), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Reference URL back-fill
# ---------------------------------------------------------------------------

def update_reference_urls_from_resources(db_path: Path) -> int:
    """Copy resource download_url back into linked reference_entries.url."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT re.id AS reference_id, re.url, r.download_url
            FROM reference_entries re
            LEFT JOIN reference_resources rr ON rr.reference_id = re.id
            LEFT JOIN resources r ON r.id = rr.resource_id
            ORDER BY re.imported_at ASC
            """
        ).fetchall()
        updated = 0
        for row in rows:
            current = str(row["url"] or "").strip()
            candidate = str(row["download_url"] or "").strip()
            if is_external_url(current):
                continue
            if not is_external_url(candidate):
                continue
            conn.execute(
                "UPDATE reference_entries SET url = ? WHERE id = ?",
                (candidate, str(row["reference_id"])),
            )
            updated += 1
        conn.commit()
    return updated


def merge_url_lists(existing: list[str], candidates: Iterable[str]) -> list[str]:
    existing_json = json.dumps(existing, ensure_ascii=True) if existing else None
    return merge_urls(existing_json, candidates)


def recover_resource_by_id(
    *,
    db_path: Path,
    archive_dir: Path,
    resource_id: str,
    manifest_root: Path,
    manifest_max_files: int,
    shallow_dirs: list[Path] | None = None,
    deep_dirs: list[Path] | None = None,
    enable_wayback_lookup: bool = True,
    wayback_delay: float = 1.0,
    enable_web_search: bool = True,
    persist: bool = True,
    verbose: bool = False,
) -> dict[str, object]:
    task = load_resource_task_by_id(db_path, resource_id)
    if task is None:
        raise ValueError(f"Resource not found: {resource_id}")

    shallow_search = shallow_dirs if shallow_dirs is not None else SHALLOW_SEARCH_DIRS[:]
    deep_search = deep_dirs if deep_dirs is not None else DEEP_SEARCH_DIRS[:]
    use_manifests = manifest_root.is_dir()
    archive_file = (archive_dir / task.archived_relpath).resolve()
    existing_urls = merge_urls(task.download_urls_json, [task.download_url or ""])

    phase_logs: list[dict[str, object]] = []
    blocked_reason: str | None = None
    selected_phase: str | None = None
    working_urls = existing_urls[:]

    def phase_start(phase: str, detail: str) -> None:
        if verbose:
            print(f"[{phase}] {detail}")

    def phase_done(phase: str, detail: str, *, success: bool, urls: list[str] | None = None) -> None:
        nonlocal selected_phase
        rec: dict[str, object] = {
            "phase": phase,
            "detail": detail,
            "success": success,
        }
        if urls:
            rec["urls"] = urls
        phase_logs.append(rec)
        if success and selected_phase is None:
            selected_phase = phase

    def persist_urls_if_any() -> None:
        if not persist:
            return
        primary = pick_primary(working_urls)
        if not is_external_url(primary):
            return
        save_resource_urls(db_path, task, working_urls)

    if not archive_file.exists():
        phase_done("precheck", "Archive file missing.", success=False)
        return {
            "ok": False,
            "resource_id": task.resource_id,
            "selected_phase": selected_phase,
            "resolved_url": pick_primary(working_urls),
            "urls": working_urls,
            "phase_logs": phase_logs,
            "blocked_reason": blocked_reason,
        }

    # Phase 0c: catalog database (SHA-256 exact match)
    phase_start("phase0c", "Searching catalog database by SHA-256.")
    sha256_val: str | None = None
    with get_connection(db_path) as _conn:
        _row = _conn.execute(
            "SELECT digest_sha256 FROM resources WHERE id = ? LIMIT 1",
            (task.resource_id,),
        ).fetchone()
        if _row:
            sha256_val = _row["digest_sha256"]
    catalog_found = catalog_urls_for_sha256(sha256_val)
    if verbose:
        print(f"  [phase0c] catalog={CATALOG_DB.exists()} sha256={sha256_val} → {len(catalog_found)} URL(s)")
    if catalog_found:
        working_urls = merge_url_lists(working_urls, catalog_found)
        phase_done("phase0c", "Recovered from catalog database (SHA-256 match).", success=True, urls=catalog_found)
        persist_urls_if_any()
        return {
            "ok": True,
            "resource_id": task.resource_id,
            "selected_phase": selected_phase,
            "resolved_url": pick_primary(working_urls),
            "urls": working_urls,
            "phase_logs": phase_logs,
            "blocked_reason": blocked_reason,
        }
    phase_done("phase0c", "No match in catalog database.", success=False)

    # Phase 0: browser download history (Safari / Chromium / Firefox)
    phase_start("phase0", "Searching browser download histories.")
    arch_size: int | None = None
    try:
        arch_size = archive_file.stat().st_size
    except OSError:
        pass
    browser_records = load_all_browser_downloads()
    browser_found = browser_history_urls_for_file(
        original_filename=task.original_filename,
        archive_size=arch_size,
        source_uri=task.source_uri,
        archive_file=archive_file,
        browser_records=browser_records,
    )
    if verbose:
        print(f"  [phase0] searched {len(browser_records)} browser download record(s); "
              f"found {len(browser_found)} candidate URL(s)")
    if browser_found:
        working_urls = merge_url_lists(working_urls, browser_found)
        phase_done("phase0", "Recovered from browser download history.", success=True, urls=browser_found)
        persist_urls_if_any()
        return {
            "ok": True,
            "resource_id": task.resource_id,
            "selected_phase": selected_phase,
            "resolved_url": pick_primary(working_urls),
            "urls": working_urls,
            "phase_logs": phase_logs,
            "blocked_reason": blocked_reason,
        }
    phase_done("phase0", "No match found in browser download histories.", success=False)

    # Phase 1: xattrs on archived file
    phase_start("phase1", "Checking archive file xattrs.")
    if verbose:
        print(f"  [phase1] archive file: {archive_file}")
    found = read_all_xattr_urls(archive_file)
    if verbose:
        try:
            attr_names = os.listxattr(archive_file) if hasattr(os, "listxattr") else []
            print(f"  [phase1] xattr names present: {attr_names or '(none)'}")
        except OSError:
            pass
    if found:
        working_urls = merge_url_lists(working_urls, found)
        phase_done("phase1", "Recovered from archive xattrs.", success=True, urls=found)
        persist_urls_if_any()
        return {
            "ok": True,
            "resource_id": task.resource_id,
            "selected_phase": selected_phase,
            "resolved_url": pick_primary(working_urls),
            "urls": working_urls,
            "phase_logs": phase_logs,
            "blocked_reason": blocked_reason,
        }
    phase_done("phase1", "No URL found in archive xattrs.", success=False)

    # Phase 2: source_uri path xattrs / direct URL
    phase_start("phase2", "Checking source_uri.")
    if looks_like_local_path(task.source_uri):
        src = Path(str(task.source_uri).replace("file://", "")).expanduser()
        if src.exists():
            if verbose:
                print(f"  [phase2] reading xattrs from source path: {src}")
            src_found = read_all_xattr_urls(src)
            if src_found:
                working_urls = merge_url_lists(working_urls, src_found)
                phase_done("phase2", "Recovered from source file xattrs.", success=True, urls=src_found)
                persist_urls_if_any()
                return {
                    "ok": True,
                    "resource_id": task.resource_id,
                    "selected_phase": selected_phase,
                    "resolved_url": pick_primary(working_urls),
                    "urls": working_urls,
                    "phase_logs": phase_logs,
                    "blocked_reason": blocked_reason,
                }
            phase_done("phase2", f"No URL in source xattrs ({src}).", success=False)
        else:
            phase_done("phase2", f"Source path does not exist ({src}).", success=False)
    elif is_external_url(task.source_uri):
        working_urls = merge_url_lists(working_urls, [str(task.source_uri)])
        phase_done("phase2", "source_uri already contains a URL.", success=True, urls=[str(task.source_uri)])
        persist_urls_if_any()
        return {
            "ok": True,
            "resource_id": task.resource_id,
            "selected_phase": selected_phase,
            "resolved_url": pick_primary(working_urls),
            "urls": working_urls,
            "phase_logs": phase_logs,
            "blocked_reason": blocked_reason,
        }
    else:
        phase_done("phase2", "source_uri not usable.", success=False)

    # Phase 3: filename search on local filesystem + xattrs
    phase_start("phase3", "Searching local folders and Spotlight for matching file copies.")
    archive_size: int | None = None
    try:
        archive_size = archive_file.stat().st_size
    except OSError:
        archive_size = None

    # Collect candidates from preset dirs first, then Spotlight.
    copies = find_copies_by_filename(
        task.original_filename,
        expected_size=archive_size,
        shallow_dirs=shallow_search,
        deep_dirs=deep_search,
    )

    # Spotlight search — finds the file anywhere on the indexed filesystem.
    spotlight_hits = spotlight_find_by_name(task.original_filename)
    if verbose and spotlight_hits:
        print(f"  [phase3] Spotlight found {len(spotlight_hits)} hit(s) for '{task.original_filename}'")
    # Merge, deduplicating by resolved path.
    copies_seen: set[Path] = {c.resolve() for c in copies}
    for sp in spotlight_hits:
        rp = sp.resolve()
        if rp not in copies_seen:
            copies_seen.add(rp)
            copies.append(sp)

    phase3_url_found: list[str] | None = None
    if copies:
        for copy_path in copies:
            if verbose:
                print(f"  [phase3] checking xattrs: {copy_path}")
            copy_found = read_all_xattr_urls(copy_path)
            if not copy_found:
                continue
            if verbose:
                print(f"  [phase3] found URLs: {copy_found}")
            phase3_url_found = copy_found
            break

    if phase3_url_found:
        working_urls = merge_url_lists(working_urls, phase3_url_found)
        phase_done("phase3", f"Recovered from matching local copy xattrs.", success=True, urls=phase3_url_found)
        persist_urls_if_any()
        return {
            "ok": True,
            "resource_id": task.resource_id,
            "selected_phase": selected_phase,
            "resolved_url": pick_primary(working_urls),
            "urls": working_urls,
            "phase_logs": phase_logs,
            "blocked_reason": blocked_reason,
        }
    if copies:
        phase_done("phase3", f"Found {len(copies)} local copy/copies but none had xattr URLs.", success=False)
    else:
        phase_done("phase3", "No matching local copies found (searched preset dirs + Spotlight).", success=False)

    # Phase 4: linked reference URL / DOI
    phase_start("phase4", "Checking linked reference entry.")
    ref_candidates = url_from_reference(task)
    if ref_candidates:
        working_urls = merge_url_lists(working_urls, ref_candidates)
        phase_done("phase4", "Recovered from linked reference metadata.", success=True, urls=ref_candidates)
        persist_urls_if_any()
        return {
            "ok": True,
            "resource_id": task.resource_id,
            "selected_phase": selected_phase,
            "resolved_url": pick_primary(working_urls),
            "urls": working_urls,
            "phase_logs": phase_logs,
            "blocked_reason": blocked_reason,
        }
    phase_done("phase4", "No URL/DOI in linked reference metadata.", success=False)

    # Phase 5: manifest JSON scan
    phase_start("phase5", "Scanning JSON manifests for filename clues.")
    if use_manifests:
        manifest_found = manifest_urls_for_task(
            manifest_root,
            original_filename=task.original_filename,
            resource_id=task.resource_id,
            max_files=manifest_max_files,
        )
        if manifest_found:
            working_urls = merge_url_lists(working_urls, manifest_found)
            phase_done("phase5", "Recovered from manifest JSON clues.", success=True, urls=manifest_found)
            persist_urls_if_any()
            return {
                "ok": True,
                "resource_id": task.resource_id,
                "selected_phase": selected_phase,
                "resolved_url": pick_primary(working_urls),
                "urls": working_urls,
                "phase_logs": phase_logs,
                "blocked_reason": blocked_reason,
            }
        phase_done("phase5", "No URL found in manifest JSON scan.", success=False)
    else:
        phase_done("phase5", f"Manifest root unavailable ({manifest_root}).", success=False)

    # Phase 6a: HTML canonical / og:url tags (HTML files)
    phase_start("phase6a", "Extracting canonical/og:url tags from HTML / PDF metadata.")
    _html_ext = archive_file.suffix.lower() in (".html", ".htm", ".xhtml")
    if _html_ext:
        html_urls = html_canonical_urls(archive_file)
        if html_urls:
            working_urls = merge_url_lists(working_urls, html_urls)
            phase_done("phase6a", "Recovered from HTML canonical/og:url tags.", success=True, urls=html_urls)
            persist_urls_if_any()
            return {
                "ok": True,
                "resource_id": task.resource_id,
                "selected_phase": selected_phase,
                "resolved_url": pick_primary(working_urls),
                "urls": working_urls,
                "phase_logs": phase_logs,
                "blocked_reason": blocked_reason,
            }
        phase_done("phase6a", "No canonical/og:url tags found in HTML.", success=False)
    elif HAVE_FITZ:
        pdf_urls = pdf_metadata_urls(archive_file)
        if pdf_urls:
            working_urls = merge_url_lists(working_urls, pdf_urls)
            phase_done("phase6a", "Recovered from PDF metadata and embedded links.", success=True, urls=pdf_urls)
            persist_urls_if_any()
            return {
                "ok": True,
                "resource_id": task.resource_id,
                "selected_phase": selected_phase,
                "resolved_url": pick_primary(working_urls),
                "urls": working_urls,
                "phase_logs": phase_logs,
                "blocked_reason": blocked_reason,
            }
        phase_done("phase6a", "No URL found in PDF metadata.", success=False)
    else:
        phase_done("phase6a", "PyMuPDF not installed; phase skipped.", success=False)

    # Phase 7: extracted text DOI + URL signals
    phase_start("phase7", "Scanning latest extracted text for DOI/URL candidates.")
    text_excerpt = load_latest_document_text(db_path, task.resource_id, max_chars=120000)
    text_candidates = merge_url_lists(
        [],
        [*doi_urls_from_text(text_excerpt, max_dois=8), *urls_from_text(text_excerpt, max_urls=12)],
    )
    if text_candidates:
        working_urls = merge_url_lists(working_urls, text_candidates)
        phase_done("phase7", "Recovered DOI/URL candidates from extracted text.", success=True, urls=text_candidates)
        persist_urls_if_any()
        return {
            "ok": True,
            "resource_id": task.resource_id,
            "selected_phase": selected_phase,
            "resolved_url": pick_primary(working_urls),
            "urls": working_urls,
            "phase_logs": phase_logs,
            "blocked_reason": blocked_reason,
        }
    phase_done("phase7", "No DOI/URL clues in extracted text.", success=False)

    # Phase 6b: Wayback SHA1 lookup
    phase_start("phase6b", "Checking Wayback CDX by SHA1 content hash.")
    if enable_wayback_lookup:
        cdx_found = wayback_cdx_sha1_lookup(
            archive_file,
            delay=wayback_delay,
            verbose=verbose,
        )
        if cdx_found:
            working_urls = merge_url_lists(working_urls, cdx_found)
            phase_done("phase6b", "Recovered from Wayback SHA1 lookup.", success=True, urls=cdx_found)
            persist_urls_if_any()
            return {
                "ok": True,
                "resource_id": task.resource_id,
                "selected_phase": selected_phase,
                "resolved_url": pick_primary(working_urls),
                "urls": working_urls,
                "phase_logs": phase_logs,
                "blocked_reason": blocked_reason,
            }
        phase_done("phase6b", "No Wayback captures matched this file hash.", success=False)
    else:
        phase_done("phase6b", "Wayback lookup disabled.", success=False)

    # Phase 8: lightweight web search fallback
    phase_start("phase8", "Running web query using metadata + extracted text hints.")
    if enable_web_search:
        queries = build_web_search_queries(task, text_excerpt)
        all_hits: list[str] = []
        for query in queries:
            hits, blocked = web_search_urls(query)
            if blocked and blocked_reason is None:
                blocked_reason = blocked
            if hits:
                all_hits = merge_url_lists(all_hits, hits)
            if len(all_hits) >= 10:
                break
        if all_hits:
            working_urls = merge_url_lists(working_urls, all_hits)
            phase_done("phase8", "Recovered from lightweight web query.", success=True, urls=all_hits)
            persist_urls_if_any()
            return {
                "ok": True,
                "resource_id": task.resource_id,
                "selected_phase": selected_phase,
                "resolved_url": pick_primary(working_urls),
                "urls": working_urls,
                "phase_logs": phase_logs,
                "blocked_reason": blocked_reason,
                "search_queries": queries,
            }
        detail = "No suitable URL returned from web query."
        if blocked_reason:
            detail += " Browser-assisted fallback may be required."
        phase_done("phase8", detail, success=False)
    else:
        phase_done("phase8", "Web query phase disabled.", success=False)

    persist_urls_if_any()
    return {
        "ok": is_external_url(pick_primary(working_urls)),
        "resource_id": task.resource_id,
        "selected_phase": selected_phase,
        "resolved_url": pick_primary(working_urls),
        "urls": working_urls,
        "phase_logs": phase_logs,
        "blocked_reason": blocked_reason,
    }


# ---------------------------------------------------------------------------
# Main recovery loop
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> int:  # noqa: C901
    verbose: bool = args.verbose

    # --local-only: disable all network phases for offline/dev runs.
    if args.local_only:
        if args.enable_wayback_lookup:
            args.enable_wayback_lookup = False
        if args.enable_web_search:
            args.enable_web_search = False
        print("[local-only mode] Phases 6b (Wayback) and 8 (web search) disabled.\n")

    paths = load_paths(Path(args.project_root))

    # Resolve search dirs from CLI args.
    args_shallow_dirs: list[Path] = SHALLOW_SEARCH_DIRS[:]
    args_deep_dirs: list[Path] = [
        Path(d) for d in (args.deep_search_dirs or [])
    ] or DEEP_SEARCH_DIRS[:]

    tasks = load_resource_tasks(paths.db_path)
    unresolved = [t for t in tasks if not is_external_url(t.download_url)]
    if args.resource_id:
        unresolved = [t for t in unresolved if t.resource_id == args.resource_id]

    print(f"Loaded {len(tasks)} resources from database.")
    print(f"{len(unresolved)} resources need URL recovery.\n")

    if not HAVE_FITZ and verbose:
        print("Note: PyMuPDF (fitz) not installed — phase 6a (PDF metadata) disabled.\n")

    use_manifests = Path(args.manifest_root).is_dir()
    if not use_manifests and verbose:
        print(f"Note: manifest root '{args.manifest_root}' not found; skipping phase 5.\n")

    # Load already-attempted Wayback lookups so reruns skip them.
    state_path = paths.db_path.parent / "url_recovery_attempted.json"
    if args.reset_wayback_cache and state_path.exists():
        state_path.unlink()
        if verbose:
            print("Wayback attempted-cache cleared.\n")
    attempted: set[str] = load_attempted(state_path)
    if attempted and verbose:
        print(f"Skipping Wayback for {len(attempted)} resource(s) already attempted in a previous run.\n")

    recovered: dict[str, int] = {
        "phase0c": 0,
        "phase0": 0,
        "phase1": 0, "phase2": 0, "phase3": 0,
        "phase4": 0, "phase5": 0, "phase6a": 0, "phase6b": 0,
        "phase7": 0, "phase8": 0,
    }
    failed = 0

    # Load browser download records once for the whole run (cached afterwards).
    print("Loading browser download histories…")
    browser_records = load_all_browser_downloads()
    print(
        f"  Safari: {sum(1 for r in browser_records if r.browser == 'safari')} records  "
        f"Chrome/Chromium/Brave/Edge: {sum(1 for r in browser_records if r.browser not in ('safari', 'firefox'))} records  "
        f"Firefox: {sum(1 for r in browser_records if r.browser == 'firefox')} records\n"
    )

    try:
      for index, task in enumerate(unresolved, start=1):
        archive_file = (paths.archive_dir / task.archived_relpath).resolve()
        print(f"[{index}/{len(unresolved)}] {task.original_filename}  (id={task.resource_id[:8]}…)")

        if not archive_file.exists():
            print("  ✗ archive file missing; skipping.")
            failed += 1
            continue

        # ------------------------------------------------------------------
        # Phase 0c: catalog database (SHA-256 exact match)
        # ------------------------------------------------------------------
        with get_connection(paths.db_path) as _conn:
            _sha_row = _conn.execute(
                "SELECT digest_sha256 FROM resources WHERE id = ? LIMIT 1",
                (task.resource_id,),
            ).fetchone()
            _sha256 = _sha_row["digest_sha256"] if _sha_row else None
        catalog_hits = catalog_urls_for_sha256(_sha256)
        if catalog_hits:
            merged = merge_urls(task.download_urls_json, catalog_hits)
            if is_external_url(pick_primary(merged)):
                save_resource_urls(paths.db_path, task, merged)
                print(f"  ✓ phase0c (catalog db SHA-256): {pick_primary(merged)}")
                recovered["phase0c"] += 1
                continue
        elif verbose:
            print("  [0c] no match in catalog database")

        # ------------------------------------------------------------------
        # Phase 0: browser download history (Safari / Chromium / Firefox)
        # ------------------------------------------------------------------
        archive_size_0: int | None = None
        try:
            archive_size_0 = archive_file.stat().st_size
        except OSError:
            pass
        browser_hits = browser_history_urls_for_file(
            original_filename=task.original_filename,
            archive_size=archive_size_0,
            source_uri=task.source_uri,
            archive_file=archive_file,
            browser_records=browser_records,
        )
        if browser_hits:
            merged = merge_urls(task.download_urls_json, browser_hits)
            if is_external_url(pick_primary(merged)):
                save_resource_urls(paths.db_path, task, merged)
                print(f"  ✓ phase0 (browser history): {pick_primary(merged)}")
                recovered["phase0"] += 1
                continue
        elif verbose:
            print("  [0] no match in browser download histories")

        # ------------------------------------------------------------------
        # Phase 1: xattrs on the archived copy
        # ------------------------------------------------------------------
        if verbose:
            print(f"  [1] checking xattrs on archived file: {archive_file.name}")
        found = xattrs_for_path(archive_file)
        if found:
            merged = merge_urls(task.download_urls_json, found)
            if is_external_url(pick_primary(merged)):
                save_resource_urls(paths.db_path, task, merged)
                print(f"  ✓ phase1 (archive xattr): {pick_primary(merged)}")
                recovered["phase1"] += 1
                continue
        elif verbose:
            print("  [1] no xattr URLs on archive copy")

        # ------------------------------------------------------------------
        # Phase 2: xattrs on the original source file, or source_uri as URL
        # ------------------------------------------------------------------
        if looks_like_local_path(task.source_uri):
            src = Path(str(task.source_uri).replace("file://", "")).expanduser()
            if verbose:
                print(f"  [2] checking xattrs on source path: {src}")
            if src.exists():
                src_found = xattrs_for_path(src)
                if src_found:
                    merged = merge_urls(task.download_urls_json, src_found)
                    if is_external_url(pick_primary(merged)):
                        save_resource_urls(paths.db_path, task, merged)
                        print(f"  ✓ phase2 (source_uri xattr): {pick_primary(merged)}")
                        recovered["phase2"] += 1
                        continue
                elif verbose:
                    print("  [2] no xattr URLs on source path")
            elif verbose:
                print(f"  [2] source path does not exist: {src}")
        elif is_external_url(task.source_uri):
            if verbose:
                print(f"  [2] source_uri is already a URL: {task.source_uri}")
            merged = merge_urls(task.download_urls_json, [str(task.source_uri)])
            save_resource_urls(paths.db_path, task, merged)
            print(f"  ✓ phase2 (source_uri is URL): {task.source_uri}")
            recovered["phase2"] += 1
            continue
        elif verbose:
            print(f"  [2] source_uri not useful: {task.source_uri!r}")

        # ------------------------------------------------------------------
        # Phase 3: search shallow + deep folders for original filename
        # ------------------------------------------------------------------
        archive_size: int | None = None
        try:
            archive_size = archive_file.stat().st_size
        except OSError:
            pass

        if verbose:
            size_str = f", expected size {archive_size:,} bytes" if archive_size else ""
            shallow = ', '.join(str(d) for d in args_shallow_dirs)
            deep = ', '.join(str(d) for d in args_deep_dirs) if args_deep_dirs else "(none)"
            print(f"  [3] searching for '{task.original_filename}'{size_str}")
            print(f"      shallow: {shallow}")
            print(f"      deep (recursive): {deep}")

        phase3_done = False
        copies = find_copies_by_filename(
            task.original_filename,
            expected_size=archive_size,
            shallow_dirs=args_shallow_dirs,
            deep_dirs=args_deep_dirs,
        )
        if verbose and not copies:
            print("  [3] no matching files found")
        for copy_path in copies:
            if verbose:
                print(f"  [3] found copy at {copy_path}; reading xattrs")
            copy_found = xattrs_for_path(copy_path)
            if copy_found:
                merged = merge_urls(task.download_urls_json, copy_found)
                if is_external_url(pick_primary(merged)):
                    save_resource_urls(paths.db_path, task, merged)
                    print(f"  ✓ phase3 (xattr @ {copy_path}): {pick_primary(merged)}")
                    recovered["phase3"] += 1
                    phase3_done = True
                    break
            elif verbose:
                print(f"  [3] no xattr URLs on {copy_path}")
        if phase3_done:
            continue

        # ------------------------------------------------------------------
        # Phase 4: linked reference entry url / doi (deterministic, no search)
        # ------------------------------------------------------------------
        if verbose:
            print(f"  [4] checking linked reference entry (title={task.ref_title!r}, doi={task.ref_doi!r})")
        ref_candidates = url_from_reference(task)
        if ref_candidates:
            merged = merge_urls(task.download_urls_json, ref_candidates)
            if is_external_url(pick_primary(merged)):
                save_resource_urls(paths.db_path, task, merged)
                print(f"  ✓ phase4 (reference entry): {pick_primary(merged)}")
                recovered["phase4"] += 1
                continue
        elif verbose:
            print("  [4] no URL/DOI in linked reference entry")

        # ------------------------------------------------------------------
        # Phase 5: manifest JSON scan (filename-exact match only)
        # ------------------------------------------------------------------
        if use_manifests:
            if verbose:
                print(f"  [5] scanning manifest JSONs under {args.manifest_root} for '{task.original_filename}'")
            manifest_found = manifest_urls_for_task(
                Path(args.manifest_root),
                original_filename=task.original_filename,
                resource_id=task.resource_id,
                max_files=args.manifest_max_files,
            )
            if manifest_found:
                merged = merge_urls(task.download_urls_json, manifest_found)
                if is_external_url(pick_primary(merged)):
                    save_resource_urls(paths.db_path, task, merged)
                    print(f"  ✓ phase5 (manifest scan): {pick_primary(merged)}")
                    recovered["phase5"] += 1
                    continue
            elif verbose:
                print("  [5] no URL found in manifest scan")
        elif verbose:
            print("  [5] manifest root not available; skipping")

        # ------------------------------------------------------------------
        # Phase 6a: HTML canonical/og:url tags (local, no network)
        # ------------------------------------------------------------------
        html_ext = archive_file.suffix.lower() in (".html", ".htm", ".xhtml")
        if html_ext:
            if verbose:
                print(f"  [6a] extracting canonical/og:url tags from {archive_file.name}")
            html_urls = html_canonical_urls(archive_file)
            if html_urls:
                merged = merge_urls(task.download_urls_json, html_urls)
                if is_external_url(pick_primary(merged)):
                    save_resource_urls(paths.db_path, task, merged)
                    print(f"  ✓ phase6a (HTML canonical): {pick_primary(merged)}")
                    recovered["phase6a"] += 1
                    continue
            elif verbose:
                print("  [6a] no canonical/og:url tags found in HTML")

        # Phase 6a (PDF): PDF metadata / embedded hyperlinks (local, no network)
        if not html_ext and HAVE_FITZ:
            if verbose:
                print(f"  [6a] extracting PDF metadata and embedded links from {archive_file.name}")
            meta_urls = pdf_metadata_urls(archive_file)
            if meta_urls:
                merged = merge_urls(task.download_urls_json, meta_urls)
                if is_external_url(pick_primary(merged)):
                    save_resource_urls(paths.db_path, task, merged)
                    print(f"  ✓ phase6a (PDF metadata): {pick_primary(merged)}")
                    recovered["phase6a"] += 1
                    continue
            elif verbose:
                print("  [6a] no URLs found in PDF metadata")
        elif not html_ext and verbose:
            print("  [6a] PyMuPDF not available; skipping")

        # ------------------------------------------------------------------
        # Phase 7: extracted text DOI/URL scan (local DB content)
        # ------------------------------------------------------------------
        if verbose:
            print("  [7] scanning latest extracted text for DOI/URL clues")
        text_excerpt = load_latest_document_text(paths.db_path, task.resource_id, max_chars=120000)
        text_candidates = merge_url_lists(
            [],
            [*doi_urls_from_text(text_excerpt, max_dois=8), *urls_from_text(text_excerpt, max_urls=12)],
        )
        if text_candidates:
            merged = merge_urls(task.download_urls_json, text_candidates)
            if is_external_url(pick_primary(merged)):
                save_resource_urls(paths.db_path, task, merged)
                print(f"  ✓ phase7 (extracted text): {pick_primary(merged)}")
                recovered["phase7"] += 1
                continue
        elif verbose:
            print("  [7] no DOI/URL candidates found in extracted text")

        # ------------------------------------------------------------------
        # Phase 6b: Wayback Machine CDX — SHA1 hash of file content
        # ------------------------------------------------------------------
        if args.enable_wayback_lookup:
            if task.resource_id in attempted:
                if verbose:
                    print("  [6b] Wayback already attempted for this resource; skipping")
            else:
                cdx_found = wayback_cdx_sha1_lookup(
                    archive_file,
                    delay=args.wayback_delay,
                    verbose=verbose,
                )
                attempted.add(task.resource_id)
                save_attempted(state_path, attempted)
                if cdx_found:
                    merged = merge_urls(task.download_urls_json, cdx_found)
                    if is_external_url(pick_primary(merged)):
                        save_resource_urls(paths.db_path, task, merged)
                        print(f"  ✓ phase6b (Wayback SHA1): {pick_primary(merged)}")
                        recovered["phase6b"] += 1
                        continue

        # ------------------------------------------------------------------
        # Phase 8: lightweight web query (metadata + extracted text hints)
        # ------------------------------------------------------------------
        if args.enable_web_search:
            queries = build_web_search_queries(task, text_excerpt)
            if verbose:
                print(f"  [8] web query with {len(queries)} query candidate(s)")
            web_urls: list[str] = []
            blocked_reason: str | None = None
            for query in queries:
                if verbose:
                    print(f"  [8] query: {query}")
                hits, blocked = web_search_urls(query)
                if blocked and blocked_reason is None:
                    blocked_reason = blocked
                if hits:
                    web_urls = merge_url_lists(web_urls, hits)
                if len(web_urls) >= 10:
                    break
            if web_urls:
                merged = merge_urls(task.download_urls_json, web_urls)
                if is_external_url(pick_primary(merged)):
                    save_resource_urls(paths.db_path, task, merged)
                    print(f"  ✓ phase8 (web query): {pick_primary(merged)}")
                    recovered["phase8"] += 1
                    continue
            if verbose:
                if blocked_reason:
                    print(f"  [8] web query blocked ({blocked_reason}); manual browser fallback recommended")
                else:
                    print("  [8] no URL candidates returned from web query")

        failed += 1
        print("  ✗ unresolved — no URL found in any metadata source.")

    finally:
        save_attempted(state_path, attempted)

    reference_updates = update_reference_urls_from_resources(paths.db_path)

    total_recovered = sum(recovered.values())
    print("\nRecovery summary:")
    print(f"  phase0 browser history:   {recovered['phase0']}")
    print(f"  phase1 archive xattr:     {recovered['phase1']}")
    print(f"  phase2 source_uri:        {recovered['phase2']}")
    print(f"  phase3 Downloads xattr:   {recovered['phase3']}")
    print(f"  phase4 reference entry:   {recovered['phase4']}")
    print(f"  phase5 manifest scan:     {recovered['phase5']}")
    print(f"  phase6a HTML canonical/PDF: {recovered['phase6a']}")
    print(f"  phase6b Wayback SHA1:     {recovered['phase6b']}")
    print(f"  phase7 extracted text:    {recovered['phase7']}")
    print(f"  phase8 web query:         {recovered['phase8']}")
    print(f"  ─────────────────────────")
    print(f"  total recovered:          {total_recovered}")
    print(f"  still unresolved:         {failed}")
    print(f"  reference URL backfills:  {reference_updates}")
    return 0 if failed == 0 else 2


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recover missing download URLs for resources using macOS xattr metadata.\n"
            "Does NOT perform title-based web searches (those cause false-positive matches)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--project-root",
        default=str(Path.cwd()),
        help="Project root containing .stemma/ (default: current directory)",
    )
    parser.add_argument(
        "--resource-id",
        default=None,
        help="Only process one resource ID (default: process all unresolved resources)",
    )
    parser.add_argument(
        "--manifest-root",
        default="/Volumes/X10/data",
        help="Root path to scan for JSON manifests (phase 5); skipped if path does not exist",
    )
    parser.add_argument(
        "--manifest-max-files",
        type=int,
        default=25000,
        help="Maximum number of JSON files to inspect per resource in phase 5",
    )
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print detailed per-phase progress for each resource (default: on)",
    )
    parser.add_argument(
        "--deep-search-dirs",
        nargs="*",
        metavar="DIR",
        default=None,
        help=(
            "Directories to search recursively for original filename in phase 3 "
            "(default: /Volumes/X10/data). Pass multiple paths space-separated. "
            "Use --deep-search-dirs with no value to disable deep search."
        ),
    )
    parser.add_argument(
        "--enable-wayback-lookup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Wayback Machine CDX SHA1-hash lookup for phase 6b (default: on; use --no-enable-wayback-lookup to disable)",
    )
    parser.add_argument(
        "--enable-web-search",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable lightweight web search fallback for phase 8 (default: on; use --no-enable-web-search to disable)",
    )
    parser.add_argument(
        "--wayback-delay",
        type=float,
        default=1.0,
        help="Seconds to wait between Wayback CDX requests (default: 1.0)",
    )
    parser.add_argument(
        "--reset-wayback-cache",
        action="store_true",
        default=False,
        help="Delete the already-attempted cache before running, forcing Wayback re-lookup for all resources",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        default=False,
        help=(
            "Disable all network requests (phases 6b Wayback and 8 web search). "
            "Use this during development or when offline. "
            "Equivalent to --no-enable-wayback-lookup --no-enable-web-search."
        ),
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
