from __future__ import annotations

import argparse

from stemmacodicum.application.services.project_service import ProjectService
from stemmacodicum.cli.context import CLIContext


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("init", help="Initialize Stemma project metadata and database")
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace, ctx: CLIContext) -> int:
    service = ProjectService(ctx.paths)
    result = service.init_project()

    if result.paths_created:
        for path in result.paths_created:
            ctx.console.print(f"[green]Created[/green] {path}")
    else:
        ctx.console.print("[yellow]Project paths already existed[/yellow]")

    ctx.console.print(f"[green]Database ready[/green] {result.db_path}")
    return 0
