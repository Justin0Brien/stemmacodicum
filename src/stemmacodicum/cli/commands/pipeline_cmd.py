from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from rich.markup import escape
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from stemmacodicum.application.services.extraction_service import ExtractionService
from stemmacodicum.application.services.ingestion_service import IngestionService
from stemmacodicum.application.services.pipeline_service import FinancialPipelineService
from stemmacodicum.application.services.project_service import ProjectService
from stemmacodicum.cli.context import CLIContext
from stemmacodicum.cli.docling_options import add_docling_runtime_args, get_docling_runtime_options
from stemmacodicum.core.errors import ProjectNotInitializedError
from stemmacodicum.infrastructure.archive.store import ArchiveStore
from stemmacodicum.infrastructure.db.repos.extraction_repo import ExtractionRepo
from stemmacodicum.infrastructure.db.repos.resource_repo import ResourceRepo


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("pipeline", help="Batch production pipelines")
    pipeline_subparsers = parser.add_subparsers(dest="pipeline_command", required=True)

    fin = pipeline_subparsers.add_parser(
        "financial-pass",
        help="Ingest and extract financial documents from a root directory",
    )
    fin.add_argument("--root", required=True, help="Root directory to scan")
    fin.add_argument("--max-files", type=int, help="Optional cap for this run")
    fin.add_argument("--skip-extraction", action="store_true")
    fin.add_argument(
        "--state-file",
        default=None,
        help="Optional state file path (default: <project>/.stemma/financial_pass_state.json)",
    )
    fin.add_argument(
        "--log-file",
        default=None,
        help="Optional log file path (default: <project>/.stemma/financial_pass.log.jsonl)",
    )
    fin.add_argument(
        "--extract-timeout-seconds",
        type=int,
        default=300,
        help="Per-file extraction timeout in seconds (default: 300)",
    )
    fin.add_argument(
        "--verbose-docs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print a detailed line for each processed document (default: enabled).",
    )
    add_docling_runtime_args(fin)
    fin.set_defaults(handler=run_financial_pass)


def run_financial_pass(args: argparse.Namespace, ctx: CLIContext) -> int:
    project_service = ProjectService(ctx.paths)
    if not project_service.is_initialized():
        raise ProjectNotInitializedError(
            f"Project is not initialized. Run 'stemma init' first in {ctx.paths.project_root}"
        )
    project_service.init_project()

    resource_repo = ResourceRepo(ctx.paths.db_path)
    extraction_repo = ExtractionRepo(ctx.paths.db_path)

    ingestion_service = IngestionService(resource_repo=resource_repo, archive_store=ArchiveStore(ctx.paths.archive_dir))
    extraction_service = ExtractionService(
        resource_repo=resource_repo,
        extraction_repo=extraction_repo,
        archive_dir=ctx.paths.archive_dir,
        docling_runtime_options=get_docling_runtime_options(args),
    )

    state_path = (
        Path(args.state_file).expanduser().resolve()
        if args.state_file
        else (ctx.paths.stemma_dir / "financial_pass_state.json")
    )
    log_path = (
        Path(args.log_file).expanduser().resolve()
        if args.log_file
        else (ctx.paths.stemma_dir / "financial_pass.log.jsonl")
    )

    service = FinancialPipelineService(
        ingestion_service=ingestion_service,
        extraction_service=extraction_service,
        extraction_repo=extraction_repo,
        state_path=state_path,
        log_path=log_path,
    )

    progress = Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=ctx.console,
    )

    with progress:
        overall_task = progress.add_task("Scanning candidates...", total=1, completed=0)
        current_task = progress.add_task("Waiting for first file...", total=None)

        def _on_progress(event: dict[str, object]) -> None:
            kind = event.get("event")
            if kind == "scan_complete":
                total = int(event.get("planned_total", 0))
                progress.update(
                    overall_task,
                    description=f"Pipeline progress ({total} files)",
                    total=max(total, 1),
                    completed=0,
                )
                if total == 0:
                    progress.update(current_task, description="No files to process.")
                if args.verbose_docs:
                    ctx.console.print(
                        "[pipeline] "
                        f"candidates={event.get('candidates', 0)} "
                        f"already_processed={event.get('already_processed', 0)} "
                        f"planned={total}"
                    )
                return

            if kind == "state_skip":
                if args.verbose_docs:
                    path = str(event.get("path", ""))
                    ctx.console.print(f"[pipeline] state-skip path={escape(path)}")
                return

            if kind == "file_start":
                index = int(event.get("index", 0))
                total = int(event.get("total", 0))
                path = str(event.get("path", ""))
                progress.update(
                    current_task,
                    description=f"Processing {index}/{total}: {escape(Path(path).name)}",
                )
                return

            if kind == "file_done":
                progress.advance(overall_task, 1)
                index = int(event.get("index", 0))
                total = int(event.get("total", 0))
                path = str(event.get("path", ""))
                elapsed = float(event.get("elapsed_seconds", 0.0))
                parse_elapsed = event.get("parse_elapsed_seconds")
                page_count = event.get("page_count")
                rate = event.get("pages_per_second")
                progress.update(
                    current_task,
                    description=f"Completed {index}/{total}: {escape(Path(path).name)}",
                )
                if args.verbose_docs:
                    path_display = escape(path)
                    details = [
                        f"index={index}/{total}",
                        f"path={path_display}",
                        f"ingest={event.get('ingest_status', 'unknown')}",
                        f"extract={event.get('extract_status', 'unknown')}",
                        f"tables={event.get('tables_found', 0)}",
                        f"elapsed={elapsed:.2f}s",
                    ]
                    if isinstance(parse_elapsed, (int, float)):
                        details.append(f"parse={float(parse_elapsed):.2f}s")
                    if isinstance(page_count, int) and page_count > 0:
                        details.append(f"pages={page_count}")
                    if isinstance(rate, (int, float)):
                        details.append(f"rate={float(rate):.2f}pg/s")
                    ctx.console.print("[pipeline] " + " ".join(details))
                return

            if kind == "file_error":
                progress.advance(overall_task, 1)
                index = int(event.get("index", 0))
                total = int(event.get("total", 0))
                path = str(event.get("path", ""))
                elapsed = float(event.get("elapsed_seconds", 0.0))
                err = str(event.get("error", "unknown error"))
                progress.update(
                    current_task,
                    description=f"Failed {index}/{total}: {escape(Path(path).name)}",
                )
                ctx.console.print(
                    "[pipeline] "
                    f"index={index}/{total} path={escape(path)} elapsed={elapsed:.2f}s error={escape(err)}"
                )

        stats = service.run(
            root=Path(args.root),
            max_files=args.max_files,
            run_extraction=not args.skip_extraction,
            extract_timeout_seconds=args.extract_timeout_seconds,
            progress_callback=_on_progress,
        )

    state_last_modified = "n/a"
    if state_path.exists():
        state_last_modified = datetime.fromtimestamp(state_path.stat().st_mtime).isoformat(timespec="seconds")

    no_work_reason = ""
    if stats.processed == 0 and stats.already_processed > 0:
        no_work_reason = (
            f"No new processing: all {stats.already_processed} candidate files were already in state"
        )

    lines = [
        f"Candidates found: {stats.candidates}",
        f"Already processed (state skip): {stats.already_processed}",
        f"Eligible for processing: {stats.candidates - stats.already_processed}",
        f"Processed this run: {stats.processed}",
        f"  ├─ Ingested: {stats.ingested}",
        f"  ├─ Duplicates: {stats.duplicates}",
        f"  ├─ Extracted: {stats.extracted}",
        f"  ├─ Skipped extraction: {stats.skipped_extraction}",
        f"  └─ Failed: {stats.failed}",
        f"Remaining unprocessed candidates: {stats.remaining_unprocessed}",
        (
            f"State entries: {stats.state_entries_before} -> {stats.state_entries_after} "
            f"(last modified: {state_last_modified})"
        ),
    ]
    if no_work_reason:
        lines.append(no_work_reason)
    lines.extend([f"State file: {state_path}", f"Log file: {log_path}"])

    ctx.console.print(Panel.fit("\n".join(lines), title="Financial Pipeline Summary"))

    return 0 if stats.failed == 0 else 1
