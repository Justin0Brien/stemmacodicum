#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

from stemmacodicum.core.config import load_paths
from stemmacodicum.core.document_titles import derive_human_title
from stemmacodicum.infrastructure.db.sqlite import get_connection, initialize_schema


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
    # Allow benign patterns (years, versions, acronyms, normal words).
    if _BENIGN_MIXED_TOKEN.match(t):
        return False
    # tmp-prefixed names from temp-file generators (tmpxplsxqy4, tmp_ay2og92).
    if re.match(r"^tmp[a-zA-Z0-9_]{3,}$", t, re.IGNORECASE):
        return True
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


JSON_BLOCK_RE = re.compile(r"\{.*\}", flags=re.DOTALL)


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


def load_resource_contexts(db_path: Path, include_existing: bool) -> list[ResourceContext]:
    where_clause = "" if include_existing else "WHERE coalesce(trim(r.display_title), '') = ''"
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
            excerpt = text_content[:14000]

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

    year = str(ctx.linked_reference_year or "").strip()
    if not year and ctx.ingested_at:
        year = ctx.ingested_at[:4]
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

    meta_block = "\n".join(meta_lines)

    headings_block = ""
    if ctx.heading_candidates:
        headings_block = "\ndocument_headings (extracted from content, most reliable signal):\n"
        headings_block += "\n".join(f"  - {h}" for h in ctx.heading_candidates[:6])
        headings_block += "\n"

    # Trim text excerpt; skip the first 200 chars if they look like raw metadata noise.
    excerpt = ctx.text_excerpt[:10000].strip()

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
        "  2. reference_title (bibliographic record if available)\n"
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


def llm_candidates(
    ctx: ResourceContext,
    model_name: str,
    *,
    max_retries: int = 2,
) -> tuple[str | None, list[str]]:
    try:
        import ollama  # type: ignore
    except Exception:
        return None, []

    prompt = prompt_for_context(ctx)
    client = ollama.Client()

    for attempt in range(max_retries + 1):
        temperature = 0.1 + attempt * 0.25  # raise temperature on each retry
        if attempt > 0:
            # Find the first rejected token so we can explain the problem.
            bad_tokens = [
                tok for tok in re.split(r"[\s\u2013\u2014/|\-]+", prev_primary or "")
                if _is_garbage_token(tok.strip("()[]{}\"'.,-"))
            ]
            bad_example = bad_tokens[0] if bad_tokens else "<code>"
            retry_note = (
                f"\n\nIMPORTANT CORRECTION: Your previous suggestion contained a "
                f"random-looking identifier: '{bad_example}'. This is NOT a valid title word. "
                f"It appears to have come from the filename, which is a machine-generated "
                f"temporary name with no meaning whatsoever. "
                f"Ignore the filename entirely and base the title ONLY on the document "
                f"content and metadata fields below. Output corrected JSON only."
            )
            full_prompt = prompt + retry_note
        else:
            full_prompt = prompt
            prev_primary = None

        try:
            response = client.generate(
                model=model_name,
                prompt=full_prompt,
                options={"temperature": temperature},
            )
        except Exception:
            return None, []

        if isinstance(response, dict):
            text = str(response.get("response", "")).strip()
        else:
            text = str(getattr(response, "response", "") or "").strip()

        parsed_primary, parsed_alts = parse_llm_title_payload(text)
        primary, candidates = sanitize_titles(parsed_primary, parsed_alts)

        if primary is not None:
            return primary, candidates

        # All candidates were rejected — remember the raw primary for the retry message.
        prev_primary = parsed_primary

    return None, []


def fallback_candidates(ctx: ResourceContext) -> tuple[str | None, list[str]]:
    candidates: list[str] = []

    # Best non-LLM signal: the first clean extracted heading.
    for h in ctx.heading_candidates:
        if not _is_bad_title(h):
            candidates.append(h)
            break

    # Bibliographic reference title is highly reliable.
    if ctx.linked_reference_title:
        ref = str(ctx.linked_reference_title).strip()
        if not _is_bad_title(ref):
            candidates.append(ref)

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


def run(args: argparse.Namespace) -> int:
    paths = load_paths(Path(args.project_root))
    schema_path = Path(__file__).parent.parent / "src" / "stemmacodicum" / "infrastructure" / "db" / "schema.sql"
    initialize_schema(paths.db_path, schema_path)
    contexts = load_resource_contexts(paths.db_path, include_existing=args.include_existing)
    if not contexts:
        print("No resources require title generation.")
        return 0

    print(f"Generating titles for {len(contexts)} resources...")
    llm_success = 0
    fallback_used = 0
    for idx, ctx in enumerate(contexts, start=1):
        print(f"[{idx}/{len(contexts)}] {ctx.resource_id} {ctx.original_filename}")
        primary, candidates = llm_candidates(ctx, args.model) if args.use_llm else (None, [])
        if primary:
            llm_success += 1
        else:
            primary, candidates = fallback_candidates(ctx)
            fallback_used += 1
        persist_titles(paths.db_path, ctx.resource_id, primary, candidates)
        print(f"  title: {primary or '(none)'}")

    print("")
    print("Title generation summary:")
    print(f"  resources processed: {len(contexts)}")
    print(f"  LLM-generated:       {llm_success}")
    print(f"  fallback-generated:  {fallback_used}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate human-readable resource titles.")
    parser.add_argument("--project-root", default=str(Path.cwd()), help="Project root containing .stemma")
    parser.add_argument("--model", default="qwen3:4b", help="Ollama model for title generation")
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
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
