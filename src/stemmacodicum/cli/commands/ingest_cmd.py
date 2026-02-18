from __future__ import annotations

import argparse
from pathlib import Path

from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from stemmacodicum.application.services.ingestion_service import IngestionService
from stemmacodicum.application.services.project_service import ProjectService
from stemmacodicum.cli.context import CLIContext
from stemmacodicum.core.errors import ProjectNotInitializedError, ResourceIngestError
from stemmacodicum.infrastructure.archive.store import ArchiveStore
from stemmacodicum.infrastructure.db.repos.resource_repo import ResourceRepo


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("ingest", help="Ingest one or more source files into immutable archive")
    parser.add_argument("paths", nargs="+", help="Local file paths to ingest")
    parser.add_argument("--source-uri", help="Optional source URI metadata for all provided files")
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace, ctx: CLIContext) -> int:
    project_service = ProjectService(ctx.paths)
    if not project_service.is_initialized():
        raise ProjectNotInitializedError(
            f"Project is not initialized. Run 'stemma init' first in {ctx.paths.project_root}"
        )

    repo = ResourceRepo(ctx.paths.db_path)
    store = ArchiveStore(ctx.paths.archive_dir)
    service = IngestionService(repo, store)

    table = Table(title="Ingest Results")
    table.add_column("File", overflow="fold")
    table.add_column("Status")
    table.add_column("Digest (sha256)", overflow="fold")

    exit_code = 0
    paths = [Path(p) for p in args.paths]

    progress = Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=ctx.console,
    )

    with progress:
        task = progress.add_task("Ingesting", total=len(paths))
        for p in paths:
            try:
                result = service.ingest_file(p, source_uri=args.source_uri)
                table.add_row(str(p), result.status, result.resource.digest_sha256)
            except ResourceIngestError as exc:
                table.add_row(str(p), "error", str(exc))
                exit_code = 1
            finally:
                progress.advance(task, 1)

    ctx.console.print(table)
    return exit_code
