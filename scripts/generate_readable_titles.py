#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import queue
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

try:
    from rich.console import Console
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    from rich.table import Table
    from rich.text import Text
    from rich import box
    HAVE_RICH = True
except ImportError:
    HAVE_RICH = False

from stemmacodicum.core.config import load_paths
from stemmacodicum.core.document_titles import derive_human_title
from stemmacodicum.infrastructure.db.sqlite import get_connection, initialize_schema

console = Console(highlight=False) if HAVE_RICH else None


def _print(msg: str, style: str = "") -> None:
    if console:
        console.print(msg, style=style)
    else:
        print(msg)


# Words that are never acceptable as a title on their own or as the opening noun.
_USELESS_WORDS = re.compile(
    r"^(document|file|paper|resource|attachment|unknown|untitled|misc|miscellaneous)\b",
    re.IGNORECASE,
)

# Recognisable benign tokens that contain digits: version strings, year-ranges,
# common units, etc.  Tokens matching these are NOT considered garbage.
# NOTE: deliberately NO re.IGNORECASE — [A-Z][a-z]+ must match only properly
# capitalised words, not all-lowercase consonant-only junk like 'vcnrpxt'.
_BENIGN_MIXED_TOKEN = re.compile(
    r"^(?:"
    r"[vV]\d+(?:\.\d+)*"           # v2, V1.0.3
    r"|\d{4}[/\-]\d{2,4}"          # 2024/25, 2023-24
    r"|\d{4}"                       # plain year
    r"|\d+[gkGK]"                   # 4G, 5G, 4K
    r"|[A-Z]{2,6}"                  # all-caps acronyms: NHS, EU, USA
    r"|[A-Z][a-z]+"                 # properly capitalised word: Aberdeen
    r")$",
)


def _is_garbage_token(token: str) -> bool:
    """Return True if a single space-delimited token looks like a random identifier."""
    t = token.strip("()[]{}\"'.,-").strip()
    if len(t) < 5:
        return False
    # Check tmp-prefix FIRST — before the benign-pattern check, because
    # 'Tmpcfqddjyk' would otherwise match [A-Z][a-z]+ and be wrongly classed benign.
    if re.match(r"^tmp[a-zA-Z0-9_]{3,}$", t, re.IGNORECASE):
        return True
    # Allow benign patterns (years, versions, acronyms, normal words).
    if _BENIGN_MIXED_TOKEN.match(t):
        return False
    has_digit = bool(re.search(r"\d", t))
    alpha_chars = [c for c in t.lower() if c.isalpha()]
    # Tokens that mix letters and digits in any order (Tmph5Yun83Q, ak432nm).
    if has_digit and len(alpha_chars) >= 2:
        return True
    # Pure-alpha tokens that are implausibly consonant-heavy (no real word,
    # e.g. 'tmpxplsxqy', 'vcnrpxt') — vowel ratio < 0.20 for len >= 7.
    if len(alpha_chars) >= 7:
        vowel_ratio = sum(1 for c in alpha_chars if c in "aeiou") / len(alpha_chars)
        if vowel_ratio < 0.20:
            return True
    return False


def _has_garbage_token(title: str) -> bool:
    """Return True if any token in the title looks like a random identifier."""
    for token in re.split(r"[\s\u2013\u2014/|\-]+", title):
        if _is_garbage_token(token):
            return True
    return False


@dataclass(slots=True)
class ResourceContext:
    resource_id: str
    original_filename: str
    source_uri: str | None
    download_url: str | None
    media_type: str | None
    ingested_at: str | None
    existing_title: str | None
    existing_candidates_json: str | None
    linked_reference_title: str | None
    linked_reference_author: str | None
    linked_reference_year: str | None
    linked_reference_entry_type: str | None
    linked_reference_doi: str | None
    heading_candidates: list[str]
    text_excerpt: str
    # Bibliographic metadata sourced from the external catalog database, if a
    # match was found by SHA-256 or (normalised) source URL.
    catalog_title: str | None = None
    catalog_author: str | None = None
    catalog_year: str | None = None
    catalog_url: str | None = None


@dataclass
class LlmAttempt:
    attempt: int
    elapsed_s: float
    prompt_tokens: int
    completion_tokens: int
    # Ollama-reported generation duration in nanoseconds (excludes network overhead).
    eval_duration_ns: int
    prompt_eval_duration_ns: int
    # Tokens consumed by <think>…</think> blocks (qwen3 reasoning overhead).
    think_tokens_estimate: int
    raw_response: str
    parsed_primary: Optional[str]
    parsed_alts: list[str]
    rejection_reason: Optional[str]  # why the best candidate was rejected
    model_name: str = ""          # which model produced this attempt
    timed_out: bool = False       # True if this attempt was aborted by the timeout

    @property
    def generation_tps(self) -> float:
        """Tokens/second for the generation phase (ollama-reported, excludes network)."""
        if self.eval_duration_ns > 0 and self.completion_tokens > 0:
            return self.completion_tokens / (self.eval_duration_ns / 1e9)
        return 0.0

    @property
    def prompt_tps(self) -> float:
        """Tokens/second for the prompt ingestion phase."""
        if self.prompt_eval_duration_ns > 0 and self.prompt_tokens > 0:
            return self.prompt_tokens / (self.prompt_eval_duration_ns / 1e9)
        return 0.0


@dataclass
class LlmResult:
    primary: Optional[str]
    candidates: list[str]
    attempts: list[LlmAttempt] = field(default_factory=list)
    error: Optional[str] = None  # exception message if LLM call failed hard

    @property
    def total_elapsed_s(self) -> float:
        return sum(a.elapsed_s for a in self.attempts)

    @property
    def total_prompt_tokens(self) -> int:
        return sum(a.prompt_tokens for a in self.attempts)

    @property
    def total_completion_tokens(self) -> int:
        return sum(a.completion_tokens for a in self.attempts)

    @property
    def total_think_tokens(self) -> int:
        return sum(a.think_tokens_estimate for a in self.attempts)

    @property
    def mean_generation_tps(self) -> float:
        """Weighted mean tok/s across all attempts that have duration data."""
        total_tok = sum(a.completion_tokens for a in self.attempts if a.eval_duration_ns > 0)
        total_ns = sum(a.eval_duration_ns for a in self.attempts if a.eval_duration_ns > 0)
        if total_ns > 0 and total_tok > 0:
            return total_tok / (total_ns / 1e9)
        return 0.0


# ---------------------------------------------------------------------------
# Multi-model supervisor (UCB1 bandit)
# ---------------------------------------------------------------------------

@dataclass
class ModelStats:
    model_name: str
    attempts: int = 0
    successes: int = 0
    timeouts: int = 0
    errors: int = 0
    total_elapsed_s: float = 0.0

    @property
    def success_rate(self) -> float:
        # Optimistic prior (0.5) for untried models; shrinks towards observed rate.
        if self.attempts == 0:
            return 0.5
        return self.successes / self.attempts

    @property
    def mean_elapsed_s(self) -> float:
        return self.total_elapsed_s / self.attempts if self.attempts else 999.0

    @property
    def score(self) -> float:
        """Higher = better.  Balances success rate vs mean call time."""
        # +0.05 avoids dividing by zero in edge cases; floor of 0.5s on time.
        return (self.success_rate + 0.05) / max(self.mean_elapsed_s, 0.5)


class ModelSupervisor:
    """Routes each resource to the best available model using UCB1 exploration.

    After every attempt the supervisor updates per-model statistics (success rate,
    mean call time) and uses them—plus an exploration bonus for under-sampled
    models—to select the model for the next resource.  This means a fast, reliable
    model will be preferred while the system still periodically re-tests slower
    ones in case they improve with different prompts or contexts.
    """

    _EXPLORE_C = 0.4   # exploration constant; 0 = pure exploitation

    def __init__(self, model_names: list[str]) -> None:
        if not model_names:
            raise ValueError("ModelSupervisor requires at least one model name")
        self.model_names = list(model_names)
        self.stats: dict[str, ModelStats] = {
            name: ModelStats(model_name=name) for name in model_names
        }

    def select(self) -> str:
        """Return the model name with the highest UCB1 score."""
        total_n = sum(s.attempts for s in self.stats.values())
        if total_n == 0:
            return self.model_names[0]  # first call → first in list
        best_model = self.model_names[0]
        best_ucb = -1.0
        for name in self.model_names:
            s = self.stats[name]
            if s.attempts == 0:
                return name  # unexplored → always try immediately
            exploit = s.score
            explore = self._EXPLORE_C * math.sqrt(math.log(total_n + 1) / s.attempts)
            ucb = exploit + explore
            if ucb > best_ucb:
                best_ucb = ucb
                best_model = name
        return best_model

    def record(
        self,
        model_name: str,
        *,
        success: bool,
        elapsed_s: float,
        timed_out: bool = False,
    ) -> None:
        if model_name not in self.stats:
            self.stats[model_name] = ModelStats(model_name=model_name)
        s = self.stats[model_name]
        s.attempts += 1
        s.total_elapsed_s += elapsed_s
        if success:
            s.successes += 1
        elif timed_out:
            s.timeouts += 1
        else:
            s.errors += 1

    def best_model(self) -> str:
        """Return the model with the highest pure exploitation score."""
        explored = [s for s in self.stats.values() if s.attempts > 0]
        if not explored:
            return self.model_names[0]
        return max(explored, key=lambda s: s.score).model_name

    def status_line(self) -> str:
        """One-line summary of all models (for progress bar description)."""
        parts = []
        for name in self.model_names:
            s = self.stats[name]
            short = name.split(":")[0]  # strip tag
            if s.attempts == 0:
                parts.append(f"{short}:?")            
            else:
                parts.append(f"{short}:{s.success_rate*100:.0f}%/{s.mean_elapsed_s:.0f}s")
        return "  ".join(parts)


# ---------------------------------------------------------------------------

JSON_BLOCK_RE = re.compile(r"\{.*\}", flags=re.DOTALL)


def _rejection_reason(title: str) -> str:
    """Return a short human-readable explanation of why a title is rejected."""
    t = title.strip()
    if not t:
        return "empty"
    for token in re.split(r"[\s\u2013\u2014/|\-]+", t):
        tok = token.strip("()[]{}\"'.,-")
        if _is_garbage_token(tok):
            return f"garbage token '{tok}'"
    if _USELESS_WORDS.match(t):
        return "starts with generic word"
    if re.fullmatch(r"[\d\s\-_./,;:]+", t):
        return "only digits/punctuation"
    return "unknown"


def _extract_domain(url: str | None) -> str:
    """Return the bare hostname from a URL, or empty string."""
    if not url:
        return ""
    try:
        import urllib.parse
        host = urllib.parse.urlparse(url).netloc or ""
        # Strip www. prefix
        return re.sub(r"^www\.", "", host).strip()
    except Exception:
        return ""


def _clean_filename_stem(filename: str) -> str:
    """Strip extension and return empty string for machine-generated or hash-like names."""
    stem = re.sub(r"\.[a-zA-Z0-9]{2,6}$", "", filename).strip()
    # Reject pure hex hashes (sha256/md5) or UUIDs.
    if re.fullmatch(r"[0-9a-f]{16,64}", stem, re.IGNORECASE):
        return ""
    if re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", stem, re.IGNORECASE):
        return ""
    # Reject tmp-prefixed temp-file names (Python/OS temp files like tmpxplsxqy4, tmp_ay2og92).
    if re.match(r"^tmp[a-zA-Z0-9_]{2,}$", stem, re.IGNORECASE):
        return ""
    # Reject any stem that would itself be flagged as a garbage token.
    if _is_garbage_token(stem):
        return ""
    return stem


# SQL fragment that matches any stored title that is considered useless.
# Must stay in sync with _is_bad_title() logic.
_USELESS_TITLE_SQL = """
  coalesce(trim(r.display_title), '') = ''
  OR lower(trim(r.display_title)) IN (
    'untitled','untitled document','unknown','document','file',
    'paper','resource','attachment','misc','miscellaneous'
  )
  OR lower(trim(r.display_title)) LIKE 'untitled%'
"""


# ---------------------------------------------------------------------------
# Catalog lookup (external bib database)
# ---------------------------------------------------------------------------

def _normalise_url(u: str | None) -> str | None:
    """Return a normalised URL suitable for fuzzy matching, or None."""
    if not u:
        return None
    import urllib.parse
    u = u.strip().rstrip("/").lower()
    parsed = urllib.parse.urlparse(u)
    host = re.sub(r"^www\.", "", parsed.netloc)
    return f"{parsed.scheme}://{host}{parsed.path}"


@dataclass(slots=True)
class _CatalogBib:
    title: str | None
    author: str | None
    year: str | None
    url: str | None


def load_catalog_lookup(catalog_path: Path | None) -> dict[str, "_CatalogBib"]:
    """Build a mapping from resource keys to catalog bib metadata.

    Keys are either hex SHA-256 digests or normalised URLs.  The calling code
    checks SHA-256 first (strongest), then falls back to URL matching.

    Returns an empty dict if the catalog path is None or does not exist.
    """
    if catalog_path is None or not catalog_path.exists():
        return {}

    try:
        import sqlite3 as _sqlite3
        conn = _sqlite3.connect(str(catalog_path))
        conn.row_factory = _sqlite3.Row
    except Exception as exc:
        _print(f"[dim yellow]  catalog: cannot open {catalog_path}: {exc}[/dim yellow]")
        return {}

    lookup: dict[str, _CatalogBib] = {}

    try:
        # 1. Confirmed bib_object_links (strongest: hash → object → bib_entry)
        rows = conn.execute(
            """
            SELECT o.sha256,
                   be.title, be.author, be.year, be.url_normalized
            FROM bib_object_links bol
            JOIN objects o ON o.object_id = bol.object_id
            JOIN bib_entries be ON be.bib_key = bol.bib_key
            """
        ).fetchall()
        for row in rows:
            sha = (row["sha256"] or "").strip().lower()
            if sha:
                lookup[sha] = _CatalogBib(
                    title=row["title"] or None,
                    author=row["author"] or None,
                    year=row["year"] or None,
                    url=row["url_normalized"] or None,
                )

        # 2. High-confidence match_candidates (sha256 → bib_entry, score ≥ 0.8)
        rows = conn.execute(
            """
            SELECT o.sha256,
                   be.title, be.author, be.year, be.url_normalized,
                   mc.score
            FROM match_candidates mc
            JOIN objects o ON o.object_id = mc.object_id
            JOIN bib_entries be ON be.bib_key = mc.bib_key
            WHERE mc.score >= 0.8
            ORDER BY mc.score DESC
            """
        ).fetchall()
        for row in rows:
            sha = (row["sha256"] or "").strip().lower()
            if sha and sha not in lookup:  # don't overwrite confirmed links
                lookup[sha] = _CatalogBib(
                    title=row["title"] or None,
                    author=row["author"] or None,
                    year=row["year"] or None,
                    url=row["url_normalized"] or None,
                )

        # 3. URL-based lookup from all bib_entries (normalised key → bib_entry)
        rows = conn.execute(
            """
            SELECT bib_key, title, author, year, url_normalized
            FROM bib_entries
            WHERE url_normalized IS NOT NULL
            """
        ).fetchall()
        for row in rows:
            # Exact URL key
            raw_url = (row["url_normalized"] or "").strip()
            if raw_url and raw_url not in lookup:
                lookup[raw_url] = _CatalogBib(
                    title=row["title"] or None,
                    author=row["author"] or None,
                    year=row["year"] or None,
                    url=raw_url,
                )
            # Normalised URL key (covers www./trailing-slash variants)
            norm = _normalise_url(raw_url)
            if norm and norm not in lookup:
                lookup[norm] = _CatalogBib(
                    title=row["title"] or None,
                    author=row["author"] or None,
                    year=row["year"] or None,
                    url=raw_url,
                )

    except Exception as exc:
        _print(f"[dim yellow]  catalog: query error: {exc}[/dim yellow]")
    finally:
        conn.close()

    return lookup


def _catalog_bib_for_resource(
    lookup: dict[str, "_CatalogBib"],
    sha256: str | None,
    source_uri: str | None,
    download_url: str | None,
) -> "_CatalogBib | None":
    """Return the best-matching catalog bib for a resource, or None."""
    if not lookup:
        return None
    # 1. Exact SHA-256 (most reliable)
    if sha256:
        hit = lookup.get(sha256.strip().lower())
        if hit:
            return hit
    # 2. Exact URL match, then normalised URL match
    for url_field in (download_url, source_uri):
        if not url_field:
            continue
        hit = lookup.get(url_field.strip())
        if hit:
            return hit
        norm = _normalise_url(url_field)
        if norm:
            hit = lookup.get(norm)
            if hit:
                return hit
    return None


def load_resource_contexts(
    db_path: Path,
    include_existing: bool,
    catalog_lookup: "dict[str, _CatalogBib] | None" = None,
) -> list[ResourceContext]:
    if include_existing:
        where_clause = ""
    else:
        where_clause = f"WHERE ({_USELESS_TITLE_SQL})"
    with get_connection(db_path) as conn:
        # Report breakdown so user knows why they see the count they see.
        total = conn.execute('SELECT COUNT(*) FROM resources').fetchone()[0]
        missing = conn.execute(
            f'SELECT COUNT(*) FROM resources r WHERE ({_USELESS_TITLE_SQL})'
        ).fetchone()[0]
        has_real = total - missing
        _print(
            f"  [dim]DB: {total} resources total — "
            f"{has_real} already titled, "
            f"[bold]{missing} need titles[/bold][/dim]"
        )
    with get_connection(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT
              r.id                   AS resource_id,
              r.original_filename,
              r.source_uri,
              r.download_url,
              r.media_type,
              r.ingested_at,
              r.display_title,
              r.title_candidates_json,
              r.digest_sha256,
              MIN(re.title)          AS linked_reference_title,
              MIN(re.author)         AS linked_reference_author,
              MIN(re.year)           AS linked_reference_year,
              MIN(re.entry_type)     AS linked_reference_entry_type,
              MIN(re.doi)            AS linked_reference_doi
            FROM resources r
            LEFT JOIN reference_resources rr ON rr.resource_id = r.id
            LEFT JOIN reference_entries re ON re.id = rr.reference_id
            {where_clause}
            GROUP BY r.id
            ORDER BY r.ingested_at ASC
            """
        ).fetchall()

        contexts: list[ResourceContext] = []
        for row in rows:
            resource_id = str(row["resource_id"])

            # Latest extracted text.
            text_row = conn.execute(
                """
                SELECT dt.text_content
                FROM document_texts dt
                WHERE dt.resource_id = ?
                ORDER BY dt.created_at DESC
                LIMIT 1
                """,
                (resource_id,),
            ).fetchone()
            text_content = str(text_row["text_content"] or "") if text_row else ""
            # 3,000 chars is enough context for a title; sending 14KB inflated prompts
            # and wasted LLM tokens without any quality benefit.
            excerpt = text_content[:3000]

            # Heading-level text segments (title, section_header, page_header).
            # Offsets are 0-based Python; SQLite substr is 1-based, so add 1.
            heading_rows = conn.execute(
                """
                SELECT
                  substr(dt.text_content, ts.start_offset + 1,
                         ts.end_offset - ts.start_offset) AS heading_text,
                  ts.page_index,
                  ts.order_index
                FROM text_segments ts
                JOIN document_texts dt ON dt.resource_id = ts.resource_id
                WHERE ts.resource_id = ?
                  AND (
                    ts.segment_type LIKE '%title%'
                    OR ts.segment_type LIKE '%heading%'
                    OR ts.segment_type LIKE '%section%'
                    OR ts.segment_type LIKE '%page_header%'
                  )
                ORDER BY ts.page_index ASC, ts.order_index ASC
                LIMIT 10
                """,
                (resource_id,),
            ).fetchall()
            heading_candidates: list[str] = []
            for hrow in heading_rows:
                ht = str(hrow["heading_text"] or "").strip()
                # Drop very short or very long strings (likely noise).
                if 4 <= len(ht) <= 200 and ht not in heading_candidates:
                    heading_candidates.append(ht)

            # Look up catalog metadata (by SHA-256 first, then URL).
            cat_bib = _catalog_bib_for_resource(
                catalog_lookup or {},
                sha256=row["digest_sha256"],
                source_uri=row["source_uri"],
                download_url=row["download_url"],
            )

            contexts.append(
                ResourceContext(
                    resource_id=resource_id,
                    original_filename=str(row["original_filename"] or ""),
                    source_uri=row["source_uri"],
                    download_url=row["download_url"],
                    media_type=row["media_type"],
                    ingested_at=row["ingested_at"],
                    existing_title=row["display_title"],
                    existing_candidates_json=row["title_candidates_json"],
                    linked_reference_title=row["linked_reference_title"],
                    linked_reference_author=row["linked_reference_author"],
                    linked_reference_year=row["linked_reference_year"],
                    linked_reference_entry_type=row["linked_reference_entry_type"],
                    linked_reference_doi=row["linked_reference_doi"],
                    heading_candidates=heading_candidates,
                    text_excerpt=excerpt,
                    catalog_title=cat_bib.title if cat_bib else None,
                    catalog_author=cat_bib.author if cat_bib else None,
                    catalog_year=cat_bib.year if cat_bib else None,
                    catalog_url=cat_bib.url if cat_bib else None,
                )
            )
    return contexts


def prompt_for_context(ctx: ResourceContext) -> str:
    # Assemble metadata lines, omitting blank/useless values.
    meta_lines: list[str] = []

    filename_stem = _clean_filename_stem(ctx.original_filename)
    filename_warning = ""
    if filename_stem:
        meta_lines.append(f"filename_stem: {filename_stem}")
    else:
        # Filename is a machine-generated temp name — make this explicit to the LLM.
        filename_warning = (
            "\nWARNING: The original filename is a machine-generated temporary name "
            "(e.g. 'tmpxplsxqy4.csv'). It carries zero meaning. "
            "Do NOT use any part of it in the title. "
            "Determine the title entirely from the content and metadata fields below.\n"
        )

    domain = _extract_domain(ctx.download_url or ctx.source_uri)
    if domain:
        meta_lines.append(f"source_domain: {domain}")

    # Use only the bibliographic year — NOT the ingest date, which is always the
    # current year and produces incorrect "2026" titles for old documents.
    year = str(ctx.linked_reference_year or "").strip()
    if year:
        meta_lines.append(f"year: {year}")

    if ctx.linked_reference_entry_type:
        meta_lines.append(f"reference_type: {ctx.linked_reference_entry_type}")
    if ctx.linked_reference_title:
        meta_lines.append(f"reference_title: {ctx.linked_reference_title}")
    if ctx.linked_reference_author:
        meta_lines.append(f"reference_author: {ctx.linked_reference_author}")
    if ctx.linked_reference_doi:
        meta_lines.append(f"reference_doi: {ctx.linked_reference_doi}")

    # Catalog metadata (from external bib database — matched by SHA-256 or URL).
    # Only emit if the SC bib link didn't already provide the same field, to
    # avoid cluttering the prompt with duplicates.
    if ctx.catalog_title and not ctx.linked_reference_title:
        meta_lines.append(f"catalog_linked_title: {ctx.catalog_title}")
    if ctx.catalog_author and not ctx.linked_reference_author:
        meta_lines.append(f"catalog_linked_author: {ctx.catalog_author}")
    # Prefer SC bib year; fall back to catalog year.
    if not year and ctx.catalog_year:
        meta_lines.append(f"year: {ctx.catalog_year}")

    meta_block = "\n".join(meta_lines)

    headings_block = ""
    if ctx.heading_candidates:
        headings_block = "\ndocument_headings (extracted from content, most reliable signal):\n"
        headings_block += "\n".join(f"  - {h}" for h in ctx.heading_candidates[:6])
        headings_block += "\n"

    excerpt = ctx.text_excerpt[:3000].strip()

    return (
        "Task: produce a concise, human-readable archive title for a document based on the\n"
        "evidence below. The title should describe the document's actual subject matter.\n"
        "\n"
        "Output ONLY a JSON object with two keys:\n"
        '  \'best_title\': a single string — the best title\n'
        '  \'alternatives\': an array of 2–4 alternative strings\n'
        "\n"
        "HARD RULES (violating any rule means the title is wrong):\n"
        "1. 3–10 words. Maximum 90 characters.\n"
        "2. Describe the subject, not the file. Never copy the filename verbatim.\n"
        "3. NO alphanumeric reference codes (e.g. 'ak432nm', 'REF-2023-007', 'doc_v3').\n"
        "   If the filename contains such a code, ignore it completely.\n"
        "4. NO generic words as nouns: 'Document', 'File', 'Paper', 'Resource',\n"
        "   'Attachment', 'Report' unless that is the document's actual specific genre\n"
        "   (e.g. 'Annual Report' is fine; just 'Report' is not).\n"
        "5. NO file extensions (.pdf, .docx, etc.).\n"
        "6. NO UUIDs, SHA hashes, or purely numeric identifiers.\n"
        "7. Use the year ONLY if clearly evidenced by the document content or metadata.\n"
        "8. Prefer the organisation or author name + the subject over a generic label.\n"
        "\n"
        "Evidence priority (highest first):\n"
        "  1. document_headings (if present — these are extracted directly from the content)\n"
        "  2. reference_title / catalog_linked_title (bibliographic record if available)\n"
        "  3. source_domain + context from text excerpt\n"
        "  4. filename_stem (lowest priority; may be garbled or renamed — treat with suspicion)\n"
        "\n"
        f"{filename_warning}"
        f"{meta_block}\n"
        f"{headings_block}"
        f"\ntext_excerpt_beginning:\n{excerpt}\n"
    )


def parse_llm_title_payload(raw: str) -> tuple[str | None, list[str]]:
    text = str(raw or "").strip()
    if not text:
        return None, []
    match = JSON_BLOCK_RE.search(text)
    if not match:
        return None, []
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None, []
    if not isinstance(parsed, dict):
        return None, []
    best = str(parsed.get("best_title") or "").strip() or None
    alternatives_raw = parsed.get("alternatives")
    alternatives: list[str] = []
    if isinstance(alternatives_raw, list):
        for item in alternatives_raw:
            candidate = str(item or "").strip()
            if not candidate or candidate in alternatives:
                continue
            alternatives.append(candidate)
    return best, alternatives


def _is_bad_title(title: str) -> bool:
    """Return True if a title should be rejected as useless."""
    t = title.strip()
    if not t:
        return True
    # Reject if any token looks like a random identifier (e.g. Tmph5Yun83Q, tmpxplsxqy4).
    if _has_garbage_token(t):
        return True
    # Reject if it starts with a useless generic word.
    if _USELESS_WORDS.match(t):
        return True
    # Reject if it is only digits / punctuation.
    if re.fullmatch(r"[\d\s\-_./,;:]+", t):
        return True
    return False


def sanitize_titles(primary: str | None, alternatives: list[str]) -> tuple[str | None, list[str]]:
    unique: list[str] = []
    rejected: list[str] = []
    for item in ([primary] if primary else []) + alternatives:
        candidate = str(item or "").strip()
        if not candidate:
            continue
        candidate = re.sub(r"\s+", " ", candidate).strip().rstrip(".")
        candidate = candidate[:90]
        if not candidate:
            continue
        if candidate in unique or candidate in rejected:
            continue
        if _is_bad_title(candidate):
            rejected.append(candidate)
        else:
            unique.append(candidate)
        if len(unique) >= 6:
            break
    if not unique:
        return None, []
    return unique[0], unique


def _has_llm_signal(ctx: ResourceContext) -> bool:
    """Return True if the resource has enough content to be worth an LLM call.

    Resources with only a UUID/tmp filename and no extracted text, headings, or
    bibliographic metadata will always produce a rejected title regardless of how
    many attempts are made — skip the LLM entirely and go straight to fallback.
    """
    if ctx.heading_candidates:
        return True
    if ctx.linked_reference_title or ctx.linked_reference_author or ctx.linked_reference_doi:
        return True
    if ctx.catalog_title or ctx.catalog_author:
        return True
    if _clean_filename_stem(ctx.original_filename):   # non-garbage human-readable stem
        return True
    if len(ctx.text_excerpt.strip()) > 200:
        return True
    domain = _extract_domain(ctx.download_url or ctx.source_uri)
    if domain:
        return True
    return False


def llm_candidates(
    ctx: ResourceContext,
    model_name: str,
    *,
    max_retries: int = 2,
    thinking: bool = False,
    attempt_timeout_s: float = 25.0,
    verbose: bool = False,
    status_fn: Callable[[str], None] | None = None,
) -> LlmResult:
    try:
        import ollama  # type: ignore
    except ImportError:
        return LlmResult(primary=None, candidates=[], error="ollama not installed")

    prompt = prompt_for_context(ctx)
    try:
        client = ollama.Client()
    except Exception as exc:
        return LlmResult(primary=None, candidates=[], error=f"ollama.Client(): {exc}")

    result = LlmResult(primary=None, candidates=[])
    prev_primary: Optional[str] = None

    for attempt in range(max_retries + 1):
        temperature = 0.1 + attempt * 0.25
        if attempt > 0 and prev_primary is not None:
            bad_tokens = [
                tok for tok in re.split(r"[\s\u2013\u2014/|\-]+", prev_primary)
                if _is_garbage_token(tok.strip("()[]{}\"'.,-"))
            ]
            bad_example = bad_tokens[0] if bad_tokens else "<code>"
            retry_note = (
                f"\n\nIMPORTANT CORRECTION: Your previous suggestion contained a "
                f"random-looking identifier: '{bad_example}'. This is NOT a valid title word. "
                f"Ignore the filename entirely and base the title ONLY on the document "
                f"content and metadata fields below. Output corrected JSON only."
            )
            full_prompt = prompt + retry_note
        else:
            full_prompt = prompt

        t0 = time.perf_counter()
        raw_text = ""
        prompt_tokens = 0
        completion_tokens = 0
        eval_duration_ns = 0
        prompt_eval_duration_ns = 0
        timed_out_flag = False
        try:
            generate_kwargs: dict = {
                "model": model_name,
                "prompt": full_prompt,
                "options": {"temperature": temperature},
                "stream": True,
            }
            if not thinking:
                # Suppress the reasoning chain — qwen3's <think> blocks add 3,000-5,000
                # tokens (~70s at 60 tok/s) with no benefit for a title-extraction task.
                # Requires ollama ≥ 0.7; silently ignored on older versions.
                generate_kwargs["think"] = False

            # Run the streaming generator in a background thread so we can impose
            # a hard wall-clock timeout without blocking the main thread forever.
            _chunk_q: queue.Queue = queue.Queue()
            _thread_exc: list[Exception] = []

            def _stream_worker() -> None:
                try:
                    for _c in client.generate(**generate_kwargs):
                        _chunk_q.put(_c)
                except Exception as _e:
                    _thread_exc.append(_e)
                finally:
                    _chunk_q.put(None)  # sentinel — stream finished or failed

            _t = threading.Thread(target=_stream_worker, daemon=True)
            _t.start()

            _deadline = t0 + attempt_timeout_s if attempt_timeout_s > 0 else None
            _last_status_t = t0
            _STATUS_INTERVAL = 0.5

            while True:
                _remaining = (_deadline - time.perf_counter()) if _deadline else 5.0
                if _remaining <= 0:
                    timed_out_flag = True
                    break
                try:
                    chunk = _chunk_q.get(timeout=min(_remaining, _STATUS_INTERVAL))
                except queue.Empty:
                    # Timeout expired during this poll window.
                    if _deadline and time.perf_counter() >= _deadline:
                        timed_out_flag = True
                        break
                    # Still within budget — just a quiet moment; update status.
                    if status_fn:
                        _elapsed_now = time.perf_counter() - t0
                        _in_think = "<think>" in raw_text and "</think>" not in raw_text
                        _phase = "💭" if _in_think else "⏳"
                        status_fn(f"[dim]{_phase} {len(raw_text)}ch {_elapsed_now:.0f}s[/dim]")
                    continue

                if chunk is None:     # sentinel
                    if _thread_exc:
                        raise _thread_exc[0]
                    break

                if isinstance(chunk, dict):
                    piece = chunk.get("response", "") or ""
                    done = bool(chunk.get("done", False))
                    if done:
                        prompt_tokens = int(chunk.get("prompt_eval_count") or 0)
                        completion_tokens = int(chunk.get("eval_count") or 0)
                        eval_duration_ns = int(chunk.get("eval_duration") or 0)
                        prompt_eval_duration_ns = int(chunk.get("prompt_eval_duration") or 0)
                else:
                    piece = getattr(chunk, "response", "") or ""
                    done = bool(getattr(chunk, "done", False))
                    if done:
                        prompt_tokens = int(getattr(chunk, "prompt_eval_count", 0) or 0)
                        completion_tokens = int(getattr(chunk, "eval_count", 0) or 0)
                        eval_duration_ns = int(getattr(chunk, "eval_duration", 0) or 0)
                        prompt_eval_duration_ns = int(getattr(chunk, "prompt_eval_duration", 0) or 0)

                raw_text += piece

                # Throttled live status update.
                if status_fn:
                    _now = time.perf_counter()
                    if _now - _last_status_t >= _STATUS_INTERVAL:
                        _last_status_t = _now
                        _elapsed_now = _now - t0
                        _in_think = "<think>" in raw_text and "</think>" not in raw_text
                        _phase = "💭" if _in_think else "✍"
                        status_fn(f"[dim]{_phase} {len(raw_text)}ch {_elapsed_now:.0f}s[/dim]")

        except Exception as exc:
            elapsed = time.perf_counter() - t0
            err_str = str(exc)
            result.attempts.append(LlmAttempt(
                attempt=attempt,
                elapsed_s=elapsed,
                prompt_tokens=0,
                completion_tokens=0,
                eval_duration_ns=0,
                prompt_eval_duration_ns=0,
                think_tokens_estimate=0,
                raw_response="",
                parsed_primary=None,
                parsed_alts=[],
                rejection_reason=f"LLM error: {err_str}",
                model_name=model_name,
                timed_out=False,
            ))
            result.error = err_str
            return result
        elapsed = time.perf_counter() - t0

        if timed_out_flag:
            result.attempts.append(LlmAttempt(
                attempt=attempt,
                elapsed_s=elapsed,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                eval_duration_ns=eval_duration_ns,
                prompt_eval_duration_ns=prompt_eval_duration_ns,
                think_tokens_estimate=0,
                raw_response=raw_text,
                parsed_primary=None,
                parsed_alts=[],
                rejection_reason=f"timed out after {elapsed:.0f}s",
                model_name=model_name,
                timed_out=True,
            ))
            result.error = f"timed out after {elapsed:.0f}s"
            return result

        # Estimate how many tokens were spent on <think>…</think> reasoning blocks.
        # Rough heuristic: think text length / 4 chars per token.
        think_text = " ".join(re.findall(r"<think>(.*?)</think>", raw_text, re.DOTALL))
        think_tokens_estimate = max(0, len(think_text) // 4)

        # Strip think-block content (qwen3 and other reasoning models).
        raw_text_clean = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()

        parsed_primary, parsed_alts = parse_llm_title_payload(raw_text_clean or raw_text)
        primary, candidates = sanitize_titles(parsed_primary, parsed_alts)

        # Determine rejection reason for the best parsed candidate.
        rejection_reason: Optional[str] = None
        if primary is None:
            if not raw_text_clean:
                rejection_reason = "empty response"
            elif not JSON_BLOCK_RE.search(raw_text_clean or raw_text):
                rejection_reason = "no JSON in response"
            elif parsed_primary is None:
                rejection_reason = "JSON parsed but best_title missing/empty"
            else:
                rejection_reason = _rejection_reason(parsed_primary)

        result.attempts.append(LlmAttempt(
            attempt=attempt,
            elapsed_s=elapsed,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            eval_duration_ns=eval_duration_ns,
            prompt_eval_duration_ns=prompt_eval_duration_ns,
            think_tokens_estimate=think_tokens_estimate,
            raw_response=raw_text,
            parsed_primary=parsed_primary,
            parsed_alts=parsed_alts,
            rejection_reason=rejection_reason,
            model_name=model_name,
            timed_out=False,
        ))

        if primary is not None:
            result.primary = primary
            result.candidates = candidates
            return result

        prev_primary = parsed_primary

    return result


def fallback_candidates(ctx: ResourceContext) -> tuple[str | None, list[str]]:
    candidates: list[str] = []

    # Best non-LLM signal: the first clean extracted heading.
    for h in ctx.heading_candidates:
        if not _is_bad_title(h):
            candidates.append(h)
            break

    # Bibliographic reference title from SC's own linked entries.
    if ctx.linked_reference_title:
        ref = str(ctx.linked_reference_title).strip()
        if not _is_bad_title(ref):
            candidates.append(ref)

    # Catalog-matched bib title (from external catalog database).
    if ctx.catalog_title:
        cat = str(ctx.catalog_title).strip()
        if not _is_bad_title(cat) and cat not in candidates:
            candidates.append(cat)

    # derive_human_title as a further fallback.
    derived = derive_human_title(
        original_filename=ctx.original_filename,
        source_uri=ctx.source_uri,
        text_preview=ctx.text_excerpt,
        fallback_id=ctx.resource_id,
    )
    if derived and not _is_bad_title(derived):
        candidates.append(derived)

    # Filename stem last resort — only if it looks human-readable.
    stem = _clean_filename_stem(ctx.original_filename)
    if stem and not _is_bad_title(stem):
        candidates.append(stem.replace("_", " ").replace("-", " "))

    primary = candidates[0] if candidates else None
    return sanitize_titles(primary, candidates)


def persist_titles(db_path: Path, resource_id: str, primary: str | None, candidates: list[str]) -> None:
    with get_connection(db_path) as conn:
        conn.execute(
            """
            UPDATE resources
            SET display_title = ?, title_candidates_json = ?
            WHERE id = ?
            """,
            (
                primary,
                json.dumps(candidates, ensure_ascii=True) if candidates else None,
                resource_id,
            ),
        )
        conn.commit()


def run(args: argparse.Namespace) -> int:  # noqa: C901
    paths = load_paths(Path(args.project_root))
    schema_path = Path(__file__).parent.parent / "src" / "stemmacodicum" / "infrastructure" / "db" / "schema.sql"
    initialize_schema(paths.db_path, schema_path)

    _print("Loading resources from database...", style="dim")

    # Load external catalog lookup (optional but improves title quality when available).
    _catalog_path = Path(args.catalog) if getattr(args, "catalog", None) else None
    if _catalog_path is None:
        # Check the default location if it exists.
        _default_catalog = Path("/Volumes/X10/data/sources/catalog.sqlite")
        if _default_catalog.exists():
            _catalog_path = _default_catalog
    catalog_lookup: dict = {}
    if _catalog_path and _catalog_path.exists():
        catalog_lookup = load_catalog_lookup(_catalog_path)
        _print(
            f"  [dim]Catalog [cyan]{_catalog_path.name}[/cyan]: "
            f"{len(catalog_lookup)} lookup entries loaded[/dim]"
        )
    else:
        _print("  [dim]No external catalog found — skipping catalog lookup[/dim]")

    contexts = load_resource_contexts(
        paths.db_path,
        include_existing=args.include_existing,
        catalog_lookup=catalog_lookup,
    )
    if catalog_lookup:
        catalog_hits = sum(1 for c in contexts if c.catalog_title)
        if catalog_hits:
            _print(
                f"  [dim]Catalog provided metadata for [cyan]{catalog_hits}[/cyan] "
                f"of {len(contexts)} resources[/dim]"
            )
    if not contexts:
        _print("[bold green]No resources require title generation.[/bold green]")
        return 0

    model_list = [m.strip() for m in args.models.split(",") if m.strip()]
    if not model_list:
        model_list = ["qwen3:4b"]
    supervisor = ModelSupervisor(model_list)

    model_display = ", ".join(f"[yellow]{m}[/yellow]" for m in model_list)
    _print(
        f"[bold]Generating titles for [cyan]{len(contexts)}[/cyan] resources[/bold]  "
        f"models=[{model_display}]  "
        f"timeout=[yellow]{args.attempt_timeout}s[/yellow]  "
        f"llm=[{'[green]on' if args.use_llm else '[red]off'}[/]]"
    )

    total_start = time.perf_counter()
    llm_success = 0
    fallback_used = 0
    no_title = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_think_tokens = 0
    total_llm_s = 0.0
    total_eval_ns = 0
    total_eval_tokens = 0
    failures: list[tuple[str, str, str]] = []  # (resource_id, filename, reason)

    progress_cols = [
        SpinnerColumn(),
        MofNCompleteColumn(),
        BarColumn(bar_width=30),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn("{task.description}"),
    ] if HAVE_RICH else None

    ctx_manager = Progress(*progress_cols, console=console) if HAVE_RICH else None

    def process_all() -> None:
        nonlocal llm_success, fallback_used, no_title
        nonlocal total_prompt_tokens, total_completion_tokens, total_llm_s
        nonlocal total_think_tokens, total_eval_ns, total_eval_tokens

        task_id = ctx_manager.add_task("", total=len(contexts)) if ctx_manager else None

        for idx, ctx in enumerate(contexts, start=1):
            short_name = ctx.original_filename[:40]
            if ctx_manager:
                ctx_manager.update(
                    task_id,
                    description=f"[dim]{short_name}[/dim]  [dim cyan]{supervisor.status_line()}[/dim cyan]",
                    advance=0,
                )

            t_resource_start = time.perf_counter()

            if args.use_llm and _has_llm_signal(ctx):
                chosen_model = supervisor.select()
                if ctx_manager and task_id is not None:
                    _chosen = chosen_model   # capture for closure
                    def _make_status_fn(tid: object, name: str, mdl: str) -> Callable[[str], None]:
                        def _fn(msg: str) -> None:
                            ctx_manager.update(
                                tid,
                                description=f"[dim]{name}[/dim] [cyan]{mdl.split(':')[0]}[/cyan] {msg}",
                            )
                        return _fn
                    _status_fn: Callable[[str], None] | None = _make_status_fn(
                        task_id, short_name, chosen_model
                    )
                else:
                    _status_fn = None
                llm_result = llm_candidates(
                    ctx, chosen_model,
                    max_retries=args.max_retries,
                    thinking=args.thinking,
                    attempt_timeout_s=args.attempt_timeout,
                    verbose=args.verbose,
                    status_fn=_status_fn,
                )
                any_timed_out = any(a.timed_out for a in llm_result.attempts)
                supervisor.record(
                    chosen_model,
                    success=llm_result.primary is not None,
                    elapsed_s=llm_result.total_elapsed_s,
                    timed_out=any_timed_out,
                )
            else:
                chosen_model = ""
                reason = "LLM disabled" if not args.use_llm else "no content signal — skipped LLM"
                llm_result = LlmResult(primary=None, candidates=[], error=reason)

            total_prompt_tokens += llm_result.total_prompt_tokens
            total_completion_tokens += llm_result.total_completion_tokens
            total_think_tokens += llm_result.total_think_tokens
            total_llm_s += llm_result.total_elapsed_s
            for att in llm_result.attempts:
                if att.eval_duration_ns > 0:
                    total_eval_ns += att.eval_duration_ns
                    total_eval_tokens += att.completion_tokens

            primary = llm_result.primary
            candidates = llm_result.candidates
            source_label = ""

            if primary:
                llm_success += 1
                n_attempts = len(llm_result.attempts)
                retry_note = f" [dim](attempt {n_attempts})[/dim]" if n_attempts > 1 else ""
                last = llm_result.attempts[-1] if llm_result.attempts else None
                mdl_short = chosen_model.split(":")[0] if chosen_model else ""
                if last and last.eval_duration_ns > 0:
                    tps = last.completion_tokens / (last.eval_duration_ns / 1e9)
                    think_note = f" [dim magenta]({last.think_tokens_estimate}tk think)[/dim magenta]" if last.think_tokens_estimate > 50 else ""
                    tok_note = (
                        f" [dim]{last.prompt_tokens}in+{last.completion_tokens}out "
                        f"@ [bold]{tps:.0f}[/bold]tok/s "
                        f"{last.elapsed_s:.1f}s[/dim]{think_note}"
                    )
                elif last:
                    tok_note = f" [dim]{last.prompt_tokens}in+{last.completion_tokens}out {last.elapsed_s:.1f}s[/dim]"
                else:
                    tok_note = ""
                source_label = f"[green]{mdl_short or 'LLM'}[/green]{retry_note}{tok_note}"
            else:
                primary, candidates = fallback_candidates(ctx)
                if primary:
                    fallback_used += 1
                    source_label = "[yellow]fallback[/yellow]"
                else:
                    no_title += 1
                    source_label = "[red]none[/red]"
                    reason = llm_result.error or "all candidates rejected"
                    failures.append((ctx.resource_id, ctx.original_filename, reason))

            persist_titles(paths.db_path, ctx.resource_id, primary, candidates)
            elapsed_resource = time.perf_counter() - t_resource_start

            # --- Per-resource output ---
            if ctx_manager:
                ctx_manager.print(
                    f"[dim][{idx}/{len(contexts)}][/dim] {source_label}  "
                    f"[bold]{primary or '(none)'}[/bold]  "
                    f"[dim]{short_name}  {elapsed_resource:.1f}s[/dim]"
                )
            else:
                print(f"[{idx}/{len(contexts)}] {short_name}")
                print(f"  [{source_label}] {primary or '(none)'}  ({elapsed_resource:.1f}s)")

            # --- Verbose detail: show all LLM attempts ---
            if args.verbose and llm_result.attempts:
                for att in llm_result.attempts:
                    tps_str = f"  gen {att.generation_tps:.0f} tok/s  prompt {att.prompt_tps:.0f} tok/s" if att.eval_duration_ns > 0 else ""
                    think_str = f"  ~{att.think_tokens_estimate} think tokens" if att.think_tokens_estimate else ""
                    header = (
                        f"  [dim]attempt {att.attempt + 1}: {att.elapsed_s:.1f}s  "
                        f"{att.prompt_tokens}in+{att.completion_tokens}out{tps_str}{think_str}[/dim]"
                    )
                    if ctx_manager:
                        ctx_manager.print(header)
                    else:
                        print(header)
                    if att.rejection_reason:
                        rej_msg = f"  [bold red]  rejected:[/bold red] {att.rejection_reason}"
                        ctx_manager.print(rej_msg) if ctx_manager else print(rej_msg)
                    if att.raw_response:
                        # Show first 300 chars of raw response, stripped of think blocks.
                        raw_preview = re.sub(r"<think>.*?</think>", "[dim]<think…>[/dim]",
                                             att.raw_response[:600], flags=re.DOTALL)
                        raw_msg = f"  [dim]  raw: {raw_preview.strip()[:300]}[/dim]"
                        ctx_manager.print(raw_msg) if ctx_manager else print(raw_msg)

            if ctx_manager:
                ctx_manager.update(task_id, advance=1)

    if ctx_manager:
        with ctx_manager:
            process_all()
    else:
        process_all()

    total_elapsed = time.perf_counter() - total_start
    avg_per_resource = total_elapsed / len(contexts) if contexts else 0
    overall_tps = total_eval_tokens / (total_eval_ns / 1e9) if total_eval_ns > 0 else 0.0
    think_pct = (total_think_tokens * 100 // total_completion_tokens) if total_completion_tokens > 0 else 0

    # Print per-model supervisor report.
    _print("")
    if HAVE_RICH and console:
        mtable = Table(title="Model Performance", box=box.SIMPLE_HEAVY, show_header=True)
        mtable.add_column("Model", style="cyan")
        mtable.add_column("Tried", justify="right")
        mtable.add_column("✓ Success", justify="right", style="green")
        mtable.add_column("✗ Failed", justify="right", style="red")
        mtable.add_column("⏱ Timeout", justify="right", style="yellow")
        mtable.add_column("Mean time", justify="right")
        mtable.add_column("Score", justify="right", style="bold")
        for name in supervisor.model_names:
            s = supervisor.stats[name]
            if s.attempts == 0:
                mtable.add_row(name, "0", "-", "-", "-", "-", "-")
            else:
                is_best = (name == supervisor.best_model())
                best_mark = " ★" if is_best else ""
                mtable.add_row(
                    f"{name}{best_mark}",
                    str(s.attempts),
                    str(s.successes),
                    str(s.errors),
                    str(s.timeouts),
                    f"{s.mean_elapsed_s:.1f}s",
                    f"{s.score:.3f}",
                )
        console.print(mtable)
    else:
        print("\nModel performance:")
        for name in supervisor.model_names:
            s = supervisor.stats[name]
            if s.attempts:
                print(f"  {name}: {s.successes}/{s.attempts} ok, "
                      f"{s.timeouts} timeout, {s.mean_elapsed_s:.1f}s mean")

    # --- Summary ---
    _print("")
    if HAVE_RICH and console:
        table = Table(title="Title Generation Summary", box=box.SIMPLE_HEAVY, show_header=False)
        table.add_column("metric", style="dim", width=28)
        table.add_column("value", style="bold cyan")
        table.add_row("Resources processed", str(len(contexts)))
        table.add_row("LLM-generated", f"[green]{llm_success}[/green]")
        table.add_row("Fallback-generated", f"[yellow]{fallback_used}[/yellow]")
        table.add_row("No title found", f"[red]{no_title}[/red]" if no_title else "[green]0[/green]")
        table.add_row("Total prompt tokens (in)", f"{total_prompt_tokens:,}")
        table.add_row("Total completion tokens (out)", f"{total_completion_tokens:,}")
        table.add_row("  of which: think-block tokens", f"{total_think_tokens:,}  ({think_pct}%)" if total_think_tokens else "0")
        table.add_row("Generation speed (tok/s)",
                      f"[bold cyan]{overall_tps:.1f}[/bold cyan]" if overall_tps else "[dim]n/a[/dim]")
        table.add_row("Total LLM time", f"{total_llm_s:.1f}s")
        table.add_row("Total wall time", f"{total_elapsed:.1f}s")
        table.add_row("Avg wall time per resource", f"{avg_per_resource:.1f}s")
        console.print(table)
    else:
        print("\nTitle Generation Summary:")
        print(f"  resources processed:      {len(contexts)}")
        print(f"  LLM-generated:            {llm_success}")
        print(f"  fallback-generated:       {fallback_used}")
        print(f"  no title found:           {no_title}")
        print(f"  total prompt tokens (in):     {total_prompt_tokens:,}")
        print(f"  total completion tokens (out):{total_completion_tokens:,}")
        print(f"  think-block tokens:           {total_think_tokens:,} ({think_pct}%)")
        print(f"  generation speed:             {overall_tps:.1f} tok/s")
        print(f"  total LLM time:               {total_llm_s:.1f}s")
        print(f"  total wall time:              {total_elapsed:.1f}s")
        print(f"  avg per resource:             {avg_per_resource:.1f}s")

    if failures:
        _print("")
        _print(f"[bold red]{len(failures)} resource(s) got no title:[/bold red]")
        for rid, fname, reason in failures:
            _print(f"  [dim]{rid}[/dim]  {fname}  [red]{reason}[/red]")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate human-readable resource titles.")
    parser.add_argument("--project-root", default=str(Path.cwd()), help="Project root containing .stemma")
    parser.add_argument(
        "--catalog",
        default=None,
        metavar="PATH",
        help=(
            "Path to an external catalog SQLite database "
            "(e.g. /Volumes/X10/data/sources/catalog.sqlite). "
            "When present the catalog is queried by SHA-256 and source URL to supply "
            "additional bibliographic title/author/year metadata to the LLM. "
            "If omitted the default location /Volumes/X10/data/sources/catalog.sqlite "
            "is tried automatically."
        ),
    )
    parser.add_argument(
        "--models",
        default="phi4-mini:latest,qwen3:4b",
        help=(
            "Comma-separated list of Ollama models to use, in priority order. "
            "The supervisor (UCB1 bandit) will learn which model is fastest and most "
            "reliable and route resources accordingly. "
            "Default: phi4-mini:latest,qwen3:4b"
        ),
    )
    parser.add_argument(
        "--attempt-timeout",
        type=float,
        default=25.0,
        dest="attempt_timeout",
        help="Hard wall-clock timeout in seconds per LLM attempt (default: 25). "
             "Timed-out models are penalised by the supervisor.",
    )
    parser.add_argument(
        "--use-llm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Ollama LLM for title candidates.",
    )
    parser.add_argument(
        "--include-existing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also regenerate resources that already have display_title set.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Max LLM retries per resource when title is rejected (default: 2).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Show raw LLM output, token counts, and rejection reasons for each resource.",
    )
    parser.add_argument(
        "--thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        dest="thinking",
        help=(
            "Enable the model's reasoning/thinking chain (default: OFF). "
            "Disable saves ~70s per resource for qwen3 by suppressing <think> blocks. "
            "Pass --thinking to enable if you need higher-quality results."
        ),
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
