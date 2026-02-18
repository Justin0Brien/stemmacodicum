from __future__ import annotations

import argparse

from rich.table import Table

from stemmacodicum.application.services.project_service import ProjectService
from stemmacodicum.cli.context import CLIContext
from stemmacodicum.core.errors import ProjectNotInitializedError
from stemmacodicum.infrastructure.db.repos.resource_repo import ResourceRepo


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("resources", help="List ingested resources")
    parser.add_argument("--limit", type=int, default=50)
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace, ctx: CLIContext) -> int:
    project_service = ProjectService(ctx.paths)
    if not project_service.is_initialized():
        raise ProjectNotInitializedError(
            f"Project is not initialized. Run 'stemma init' first in {ctx.paths.project_root}"
        )

    repo = ResourceRepo(ctx.paths.db_path)
    resources = repo.list(limit=args.limit)

    table = Table(title=f"Resources ({len(resources)})")
    table.add_column("ID")
    table.add_column("Filename")
    table.add_column("Media Type")
    table.add_column("Size")
    table.add_column("Digest (sha256)", overflow="fold")

    for r in resources:
        table.add_row(r.id, r.original_filename, r.media_type, str(r.size_bytes), r.digest_sha256)

    ctx.console.print(table)
    return 0
