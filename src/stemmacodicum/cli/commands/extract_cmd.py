from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from stemmacodicum.application.services.extraction_service import ExtractionService
from stemmacodicum.application.services.project_service import ProjectService
from stemmacodicum.application.services.vector_service import VectorIndexingService
from stemmacodicum.cli.context import CLIContext
from stemmacodicum.cli.docling_options import add_docling_runtime_args, get_docling_runtime_options
from stemmacodicum.core.errors import ProjectNotInitializedError, ValidationError
from stemmacodicum.infrastructure.db.repos.extraction_repo import ExtractionRepo
from stemmacodicum.infrastructure.db.repos.resource_repo import ResourceRepo
from stemmacodicum.infrastructure.db.repos.vector_repo import VectorRepo
from stemmacodicum.infrastructure.parsers.docling_adapter import DoclingAdapter
from stemmacodicum.infrastructure.vector.chunking import VectorChunker
from stemmacodicum.infrastructure.vector.embeddings import EmbeddingConfig, SentenceTransformerEmbedder
from stemmacodicum.infrastructure.vector.qdrant_store import QdrantLocalStore


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("extract", help="Run and inspect resource extraction")
    extract_subparsers = parser.add_subparsers(dest="extract_command", required=True)

    run_parser = extract_subparsers.add_parser("run", help="Extract structured content from a resource")
    _add_resource_selectors(run_parser)
    run_parser.add_argument("--profile", default="default", help="Parser profile name")
    add_docling_runtime_args(run_parser)
    run_parser.set_defaults(handler=run_extract)

    list_parser = extract_subparsers.add_parser("tables", help="List extracted tables for a resource")
    _add_resource_selectors(list_parser)
    list_parser.add_argument("--limit", type=int, default=50)
    list_parser.set_defaults(handler=run_list_tables)

    text_parser = extract_subparsers.add_parser("text", help="Show extracted canonical text")
    _add_resource_selectors(text_parser)
    text_parser.add_argument("--run-id", help="Optional extraction run ID; defaults to latest run")
    text_parser.add_argument("--max-chars", type=int, default=4000)
    text_parser.add_argument("--full", action="store_true", help="Print full text without truncation")
    text_parser.set_defaults(handler=run_show_text)

    segments_parser = extract_subparsers.add_parser("segments", help="List extracted text segments")
    _add_resource_selectors(segments_parser)
    segments_parser.add_argument("--run-id", help="Optional extraction run ID; defaults to latest run")
    segments_parser.add_argument("--segment-type", help="Optional segment type filter")
    segments_parser.add_argument("--limit", type=int, default=200)
    segments_parser.set_defaults(handler=run_list_segments)

    annotations_parser = extract_subparsers.add_parser("annotations", help="List extracted annotations")
    _add_resource_selectors(annotations_parser)
    annotations_parser.add_argument("--run-id", help="Optional extraction run ID; defaults to latest run")
    annotations_parser.add_argument("--layer", help="Optional annotation layer filter")
    annotations_parser.add_argument("--category", help="Optional annotation category filter")
    annotations_parser.add_argument("--limit", type=int, default=200)
    annotations_parser.set_defaults(handler=run_list_annotations)

    dump_parser = extract_subparsers.add_parser(
        "dump",
        help="Export a full extraction artifact dump (tables + text + segments + annotations)",
    )
    _add_resource_selectors(dump_parser)
    dump_parser.add_argument("--run-id", help="Optional extraction run ID; defaults to latest run")
    dump_parser.add_argument("--segment-limit", type=int, default=5000)
    dump_parser.add_argument("--annotation-limit", type=int, default=5000)
    dump_parser.add_argument("--table-limit", type=int, default=1000)
    dump_parser.add_argument("--out", type=Path, help="Optional JSON output path")
    dump_parser.set_defaults(handler=run_dump)

    backfill_parser = extract_subparsers.add_parser(
        "backfill",
        help="Backfill text/standoff layers by re-running extraction where needed",
    )
    backfill_parser.add_argument("--limit-resources", type=int, default=100000)
    backfill_parser.add_argument("--max-process", type=int, default=None)
    backfill_parser.add_argument("--include-unextracted", action="store_true")
    backfill_parser.add_argument("--force-reextract", action="store_true")
    backfill_parser.add_argument("--profile", default="default", help="Parser profile name")
    add_docling_runtime_args(backfill_parser)
    backfill_parser.set_defaults(handler=run_backfill)


def _add_resource_selectors(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--resource-id")
    group.add_argument("--resource-digest")


def _require_initialized_project(ctx: CLIContext) -> None:
    project_service = ProjectService(ctx.paths)
    if not project_service.is_initialized():
        raise ProjectNotInitializedError(
            f"Project is not initialized. Run 'stemma init' first in {ctx.paths.project_root}"
        )
    project_service.init_project()


def _resolve_resource_id(args: argparse.Namespace, repo: ResourceRepo) -> str:
    if args.resource_id:
        resource = repo.get_by_id(args.resource_id)
        if resource is None:
            raise ValidationError(f"Resource ID not found: {args.resource_id}")
        return resource.id

    if not args.resource_digest:
        raise ValidationError("Provide either --resource-id or --resource-digest")
    resource = repo.get_by_digest(args.resource_digest)
    if resource is None:
        raise ValidationError(f"Resource digest not found: {args.resource_digest}")
    return resource.id


def _make_service(args: argparse.Namespace, ctx: CLIContext) -> tuple[ExtractionService, str]:
    resource_repo = ResourceRepo(ctx.paths.db_path)
    service = _create_service(args, ctx)
    resource_id = _resolve_resource_id(args, resource_repo)
    return service, resource_id


def _create_service(args: argparse.Namespace, ctx: CLIContext) -> ExtractionService:
    resource_repo = ResourceRepo(ctx.paths.db_path)
    extraction_repo = ExtractionRepo(ctx.paths.db_path)
    vector_service = VectorIndexingService(
        resource_repo=resource_repo,
        extraction_repo=extraction_repo,
        vector_repo=VectorRepo(ctx.paths.db_path),
        vector_store=QdrantLocalStore(storage_path=ctx.paths.qdrant_dir),
        embedder=SentenceTransformerEmbedder(config=EmbeddingConfig()),
        chunker=VectorChunker(),
    )
    service = ExtractionService(
        resource_repo=resource_repo,
        extraction_repo=extraction_repo,
        archive_dir=ctx.paths.archive_dir,
        docling_runtime_options=get_docling_runtime_options(args)
        if hasattr(args, "docling_auto_tune")
        else None,
        vector_indexing_service=vector_service,
    )
    return service


def run_extract(args: argparse.Namespace, ctx: CLIContext) -> int:
    _require_initialized_project(ctx)
    service, resource_id = _make_service(args, ctx)

    progress = Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        TimeElapsedColumn(),
        console=ctx.console,
    )

    with progress:
        task = progress.add_task(f"Extracting resource {resource_id}...", total=None)
        summary = service.extract_resource(resource_id=resource_id, parser_profile=args.profile)
        progress.update(task, description=f"Completed extraction for resource {resource_id}")

    timing_summary = ""
    if summary.timings:
        ordered = sorted(summary.timings.items(), key=lambda kv: kv[1], reverse=True)
        timing_summary = ", ".join(f"{name}:{seconds:.2f}s" for name, seconds in ordered[:4])

    ctx.console.print(
        Panel.fit(
            "\n".join(
                [
                    f"Run ID: {summary.run_id}",
                    f"Resource ID: {summary.resource_id}",
                    f"Tables found: {summary.tables_found}",
                    f"Text chars: {summary.text_chars}",
                    f"Segments persisted: {summary.segments_persisted}",
                    f"Annotations persisted: {summary.annotations_persisted}",
                    f"Parser: {summary.parser_name or 'unknown'} ({summary.parser_version or 'unknown'})",
                    (
                        f"Parse time: {summary.elapsed_seconds:.2f}s"
                        if summary.elapsed_seconds is not None
                        else "Parse time: n/a"
                    ),
                    f"Pages: {summary.page_count if summary.page_count is not None else 'n/a'}",
                    (
                        f"Rate: {summary.pages_per_second:.2f} pages/sec"
                        if summary.pages_per_second is not None
                        else "Rate: n/a"
                    ),
                    (f"Top timings: {timing_summary}" if timing_summary else "Top timings: n/a"),
                    (f"Vector index status: {summary.vector_status}" if summary.vector_status else "Vector index status: disabled"),
                    f"Vector chunks indexed: {summary.vector_chunks_indexed}/{summary.vector_chunks_total}",
                    (f"Vector error: {summary.vector_error}" if summary.vector_error else "Vector error: none"),
                ]
            ),
            title="Extraction Summary",
        )
    )
    return 0


def run_list_tables(args: argparse.Namespace, ctx: CLIContext) -> int:
    _require_initialized_project(ctx)
    service, resource_id = _make_service(args, ctx)
    tables = service.list_tables(resource_id=resource_id, limit=args.limit)

    output = Table(title=f"Extracted Tables ({len(tables)})")
    output.add_column("Table ID", overflow="fold")
    output.add_column("Page")
    output.add_column("Caption", overflow="fold")
    output.add_column("Rows")
    output.add_column("Cols")

    for t in tables:
        rows = len(json.loads(t.row_headers_json))
        cols = len(json.loads(t.col_headers_json))
        output.add_row(t.table_id, str(t.page_index), t.caption or "", str(rows), str(cols))

    ctx.console.print(output)
    return 0


def run_show_text(args: argparse.Namespace, ctx: CLIContext) -> int:
    _require_initialized_project(ctx)
    service, resource_id = _make_service(args, ctx)
    doc_text = service.get_document_text(resource_id=resource_id, extraction_run_id=args.run_id)
    if doc_text is None:
        raise ValidationError("No extracted text found. Run `stemma extract run` first.")

    content = doc_text.text_content
    truncated = False
    if not args.full and len(content) > args.max_chars:
        content = content[: args.max_chars]
        truncated = True

    header = [
        f"Document Text ID: {doc_text.id}",
        f"Run ID: {doc_text.extraction_run_id}",
        f"Digest: {doc_text.text_digest_sha256}",
        f"Chars: {doc_text.char_count}",
    ]
    if truncated:
        header.append(f"Showing first {args.max_chars} chars (use --full for all text)")

    body = "\n".join(header) + "\n\n" + content
    ctx.console.print(Panel.fit(body, title="Extracted Text"))
    return 0


def run_list_segments(args: argparse.Namespace, ctx: CLIContext) -> int:
    _require_initialized_project(ctx)
    service, resource_id = _make_service(args, ctx)
    segments = service.list_segments(
        resource_id=resource_id,
        extraction_run_id=args.run_id,
        segment_type=args.segment_type,
        limit=args.limit,
    )

    table = Table(title=f"Text Segments ({len(segments)})")
    table.add_column("ID")
    table.add_column("Type")
    table.add_column("Start")
    table.add_column("End")
    table.add_column("Len")
    table.add_column("Page")
    table.add_column("Order")
    for s in segments:
        table.add_row(
            s.id,
            s.segment_type,
            str(s.start_offset),
            str(s.end_offset),
            str(max(0, s.end_offset - s.start_offset)),
            str(s.page_index) if s.page_index is not None else "",
            str(s.order_index) if s.order_index is not None else "",
        )
    ctx.console.print(table)
    return 0


def run_list_annotations(args: argparse.Namespace, ctx: CLIContext) -> int:
    _require_initialized_project(ctx)
    service, resource_id = _make_service(args, ctx)
    annotations = service.list_annotations(
        resource_id=resource_id,
        extraction_run_id=args.run_id,
        layer=args.layer,
        category=args.category,
        limit=args.limit,
    )

    table = Table(title=f"Annotations ({len(annotations)})")
    table.add_column("ID")
    table.add_column("Layer")
    table.add_column("Category")
    table.add_column("Label", overflow="fold")
    table.add_column("Spans")
    table.add_column("Source")
    for a in annotations:
        table.add_row(
            str(a["id"]),
            str(a["layer"]),
            str(a["category"]),
            str(a["label"] or ""),
            str(len(a["spans"])),
            str(a["source"] or ""),
        )
    ctx.console.print(table)
    return 0


def run_dump(args: argparse.Namespace, ctx: CLIContext) -> int:
    _require_initialized_project(ctx)
    service, resource_id = _make_service(args, ctx)
    payload = service.build_dump(
        resource_id=resource_id,
        extraction_run_id=args.run_id,
        segment_limit=args.segment_limit,
        annotation_limit=args.annotation_limit,
        table_limit=args.table_limit,
    )
    encoded = json.dumps(payload, ensure_ascii=True, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded, encoding="utf-8")
        ctx.console.print(Panel.fit(f"Wrote extraction dump to {args.out}", title="Extract Dump"))
        return 0
    ctx.console.print(encoded)
    return 0


def run_backfill(args: argparse.Namespace, ctx: CLIContext) -> int:
    _require_initialized_project(ctx)
    service = _create_service(args, ctx)
    resource_repo = ResourceRepo(ctx.paths.db_path)
    extraction_repo = ExtractionRepo(ctx.paths.db_path)

    resources = resource_repo.list(limit=args.limit_resources)
    scanned = 0
    processed = 0
    extracted = 0
    skipped = 0
    failed = 0

    progress = Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        TimeElapsedColumn(),
        console=ctx.console,
    )

    with progress:
        task = progress.add_task("Backfilling extraction artifacts...", total=None)
        for resource in resources:
            scanned += 1
            progress.update(task, description=f"Backfilling {resource.original_filename}")

            if not DoclingAdapter.supports(resource.media_type, resource.original_filename):
                skipped += 1
                continue

            latest_run = extraction_repo.get_latest_run(resource.id)
            has_document_text = (
                latest_run is not None
                and extraction_repo.get_document_text_for_run(latest_run.id) is not None
            )

            should_extract = False
            if args.force_reextract:
                should_extract = True
            elif latest_run is not None and not has_document_text:
                should_extract = True
            elif latest_run is None and args.include_unextracted:
                should_extract = True

            if not should_extract:
                skipped += 1
                continue

            if args.max_process is not None and processed >= args.max_process:
                break

            processed += 1
            try:
                service.extract_resource(resource.id, parser_profile=args.profile)
                extracted += 1
            except Exception:
                failed += 1

    ctx.console.print(
        Panel.fit(
            "\n".join(
                [
                    f"Resources scanned: {scanned}",
                    f"Resources considered: {processed}",
                    f"Re-extracted: {extracted}",
                    f"Skipped: {skipped}",
                    f"Failed: {failed}",
                ]
            ),
            title="Backfill Summary",
        )
    )
    return 0 if failed == 0 else 1
