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


@dataclass(slots=True)
class ResourceContext:
    resource_id: str
    original_filename: str
    source_uri: str | None
    media_type: str | None
    existing_title: str | None
    existing_candidates_json: str | None
    linked_reference_title: str | None
    linked_reference_author: str | None
    linked_reference_year: str | None
    text_excerpt: str


JSON_BLOCK_RE = re.compile(r"\{.*\}", flags=re.DOTALL)


def load_resource_contexts(db_path: Path, include_existing: bool) -> list[ResourceContext]:
    where_clause = "" if include_existing else "WHERE coalesce(trim(r.display_title), '') = ''"
    with get_connection(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT
              r.id AS resource_id,
              r.original_filename,
              r.source_uri,
              r.media_type,
              r.display_title,
              r.title_candidates_json,
              MIN(re.title) AS linked_reference_title,
              MIN(re.author) AS linked_reference_author,
              MIN(re.year) AS linked_reference_year
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
            text_row = conn.execute(
                """
                SELECT dt.text_content
                FROM document_texts dt
                WHERE dt.resource_id = ?
                ORDER BY dt.created_at DESC
                LIMIT 1
                """,
                (str(row["resource_id"]),),
            ).fetchone()
            excerpt = str(text_row["text_content"] or "")[:14000] if text_row else ""
            contexts.append(
                ResourceContext(
                    resource_id=str(row["resource_id"]),
                    original_filename=str(row["original_filename"] or ""),
                    source_uri=row["source_uri"],
                    media_type=row["media_type"],
                    existing_title=row["display_title"],
                    existing_candidates_json=row["title_candidates_json"],
                    linked_reference_title=row["linked_reference_title"],
                    linked_reference_author=row["linked_reference_author"],
                    linked_reference_year=row["linked_reference_year"],
                    text_excerpt=excerpt,
                )
            )
    return contexts


def prompt_for_context(ctx: ResourceContext) -> str:
    return (
        "You generate short, user-friendly archive titles for source documents.\n"
        "Prioritize text near the beginning of the extract.\n"
        "Output JSON only with keys: best_title (string), alternatives (array of strings).\n"
        "Constraints:\n"
        "- Max 90 characters per title.\n"
        "- Avoid UUIDs, hashes, and file extensions.\n"
        "- Prefer pattern: <Institution/Org> - <Document Type> - <Year/Period> when inferable.\n"
        "- If year is uncertain, omit it.\n\n"
        f"filename: {ctx.original_filename}\n"
        f"source_uri: {ctx.source_uri or ''}\n"
        f"media_type: {ctx.media_type or ''}\n"
        f"reference_title: {ctx.linked_reference_title or ''}\n"
        f"reference_author: {ctx.linked_reference_author or ''}\n"
        f"reference_year: {ctx.linked_reference_year or ''}\n\n"
        f"extract_beginning:\n{ctx.text_excerpt[:12000]}\n"
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


def sanitize_titles(primary: str | None, alternatives: list[str]) -> tuple[str | None, list[str]]:
    unique: list[str] = []
    for item in ([primary] if primary else []) + alternatives:
        candidate = str(item or "").strip()
        if not candidate:
            continue
        candidate = re.sub(r"\s+", " ", candidate).strip()
        candidate = candidate[:90]
        if candidate and candidate not in unique:
            unique.append(candidate)
        if len(unique) >= 6:
            break
    if not unique:
        return None, []
    return unique[0], unique


def llm_candidates(ctx: ResourceContext, model_name: str) -> tuple[str | None, list[str]]:
    try:
        import ollama  # type: ignore
    except Exception:
        return None, []
    prompt = prompt_for_context(ctx)
    try:
        client = ollama.Client()
        response = client.generate(
            model=model_name,
            prompt=prompt,
            options={"temperature": 0.1},
        )
    except Exception:
        return None, []
    text = str(response.get("response", "")).strip() if isinstance(response, dict) else ""
    parsed_primary, parsed_alts = parse_llm_title_payload(text)
    return sanitize_titles(parsed_primary, parsed_alts)


def fallback_candidates(ctx: ResourceContext) -> tuple[str | None, list[str]]:
    primary = derive_human_title(
        original_filename=ctx.original_filename,
        source_uri=ctx.source_uri,
        text_preview=ctx.text_excerpt,
        fallback_id=ctx.resource_id,
    )
    alternatives = [primary]
    if ctx.linked_reference_title:
        alternatives.append(str(ctx.linked_reference_title).strip())
    if ctx.original_filename:
        alternatives.append(re.sub(r"\.[a-z0-9]{2,6}$", "", ctx.original_filename, flags=re.IGNORECASE))
    return sanitize_titles(primary, alternatives)


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
