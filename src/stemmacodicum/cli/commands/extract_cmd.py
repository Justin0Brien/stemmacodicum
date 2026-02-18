from __future__ import annotations

import argparse
import json

from rich.panel import Panel
from rich.table import Table

from stemmacodicum.application.services.extraction_service import ExtractionService
from stemmacodicum.cli.docling_options import add_docling_runtime_args, get_docling_runtime_options
from stemmacodicum.application.services.project_service import ProjectService
from stemmacodicum.cli.context import CLIContext
from stemmacodicum.core.errors import ProjectNotInitializedError, ValidationError
from stemmacodicum.infrastructure.db.repos.extraction_repo import ExtractionRepo
from stemmacodicum.infrastructure.db.repos.resource_repo import ResourceRepo


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("extract", help="Run and inspect resource extraction")
    extract_subparsers = parser.add_subparsers(dest="extract_command", required=True)

    run_parser = extract_subparsers.add_parser("run", help="Extract structured tables from a resource")
    _add_resource_selectors(run_parser)
    run_parser.add_argument("--profile", default="default", help="Parser profile name")
    add_docling_runtime_args(run_parser)
    run_parser.set_defaults(handler=run_extract)

    list_parser = extract_subparsers.add_parser("tables", help="List extracted tables for a resource")
    _add_resource_selectors(list_parser)
    list_parser.add_argument("--limit", type=int, default=50)
    list_parser.set_defaults(handler=run_list_tables)


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


def run_extract(args: argparse.Namespace, ctx: CLIContext) -> int:
    _require_initialized_project(ctx)

    resource_repo = ResourceRepo(ctx.paths.db_path)
    extraction_repo = ExtractionRepo(ctx.paths.db_path)
    resource_id = _resolve_resource_id(args, resource_repo)

    service = ExtractionService(
        resource_repo=resource_repo,
        extraction_repo=extraction_repo,
        archive_dir=ctx.paths.archive_dir,
        docling_runtime_options=get_docling_runtime_options(args),
    )
    summary = service.extract_resource(resource_id=resource_id, parser_profile=args.profile)

    ctx.console.print(
        Panel.fit(
            "\n".join(
                [
                    f"Run ID: {summary.run_id}",
                    f"Resource ID: {summary.resource_id}",
                    f"Tables found: {summary.tables_found}",
                ]
            ),
            title="Extraction Summary",
        )
    )
    return 0


def run_list_tables(args: argparse.Namespace, ctx: CLIContext) -> int:
    _require_initialized_project(ctx)

    resource_repo = ResourceRepo(ctx.paths.db_path)
    extraction_repo = ExtractionRepo(ctx.paths.db_path)
    resource_id = _resolve_resource_id(args, resource_repo)

    service = ExtractionService(
        resource_repo=resource_repo,
        extraction_repo=extraction_repo,
        archive_dir=ctx.paths.archive_dir,
    )
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
