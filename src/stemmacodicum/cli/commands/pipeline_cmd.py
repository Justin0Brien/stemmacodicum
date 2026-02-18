from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from rich.panel import Panel

from stemmacodicum.application.services.extraction_service import ExtractionService
from stemmacodicum.application.services.ingestion_service import IngestionService
from stemmacodicum.application.services.pipeline_service import FinancialPipelineService
from stemmacodicum.application.services.project_service import ProjectService
from stemmacodicum.cli.context import CLIContext
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

    stats = service.run(
        root=Path(args.root),
        max_files=args.max_files,
        run_extraction=not args.skip_extraction,
        extract_timeout_seconds=args.extract_timeout_seconds,
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
