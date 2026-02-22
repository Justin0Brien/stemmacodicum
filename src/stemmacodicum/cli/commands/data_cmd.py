from __future__ import annotations

import argparse

from rich.panel import Panel
from rich.table import Table

from stemmacodicum.application.services.ingestion_policy_service import IngestionPolicyService
from stemmacodicum.application.services.project_service import ProjectService
from stemmacodicum.application.services.structured_data_service import StructuredDataService
from stemmacodicum.cli.context import CLIContext
from stemmacodicum.core.errors import ProjectNotInitializedError, ValidationError
from stemmacodicum.infrastructure.db.repos.resource_repo import ResourceRepo
from stemmacodicum.infrastructure.db.repos.structured_data_repo import StructuredDataRepo


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("data", help="Structured data profiling and datapoint lookup")
    data_subparsers = parser.add_subparsers(dest="data_command", required=True)

    profile = data_subparsers.add_parser("profile", help="Profile a structured resource (CSV/XLSX)")
    _add_resource_selectors(profile)
    profile.add_argument("--force", action="store_true")
    profile.set_defaults(handler=run_profile)

    catalog = data_subparsers.add_parser("catalog", help="Show structured catalog for a resource")
    _add_resource_selectors(catalog)
    catalog.add_argument("--limit", type=int, default=200)
    catalog.set_defaults(handler=run_catalog)

    lookup = data_subparsers.add_parser("lookup-cell", help="Resolve an exact cell by filters")
    _add_resource_selectors(lookup)
    lookup.add_argument("--value-column", required=True)
    lookup.add_argument("--sheet-name")
    lookup.add_argument(
        "--filter",
        action="append",
        default=[],
        help="Filter expression KEY=VALUE; may be passed multiple times.",
    )
    lookup.set_defaults(handler=run_lookup_cell)


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
    if args.resource_digest:
        resource = repo.get_by_digest(args.resource_digest)
        if resource is None:
            raise ValidationError(f"Resource digest not found: {args.resource_digest}")
        return resource.id
    raise ValidationError("Provide --resource-id or --resource-digest")


def _service(ctx: CLIContext) -> tuple[StructuredDataService, ResourceRepo]:
    resource_repo = ResourceRepo(ctx.paths.db_path)
    service = StructuredDataService(
        resource_repo=resource_repo,
        structured_repo=StructuredDataRepo(ctx.paths.db_path),
        archive_dir=ctx.paths.archive_dir,
        policy_service=IngestionPolicyService(),
    )
    return service, resource_repo


def run_profile(args: argparse.Namespace, ctx: CLIContext) -> int:
    _require_initialized_project(ctx)
    service, repo = _service(ctx)
    resource_id = _resolve_resource_id(args, repo)
    summary = service.profile_resource(resource_id, force=bool(args.force))
    lines = [
        f"Run ID: {summary.run_id or 'n/a'}",
        f"Resource ID: {summary.resource_id}",
        f"Status: {summary.status}",
        f"Format: {summary.data_format}",
        f"Tables: {summary.table_count}",
        f"Rows observed: {summary.row_count_observed}",
        f"Scan truncated: {summary.scan_truncated}",
        f"Error: {summary.error or 'none'}",
    ]
    ctx.console.print(Panel.fit("\n".join(lines), title="Structured Profile"))
    return 0 if summary.status in {"success", "skipped"} else 1


def run_catalog(args: argparse.Namespace, ctx: CLIContext) -> int:
    _require_initialized_project(ctx)
    service, repo = _service(ctx)
    resource_id = _resolve_resource_id(args, repo)
    catalog = service.get_catalog(resource_id)
    tables = service.list_tables(resource_id, limit=int(args.limit))

    if catalog is None:
        ctx.console.print(Panel.fit("No structured catalog entry for this resource.", title="Structured Catalog"))
        return 0

    summary_lines = [
        f"Resource ID: {catalog['resource_id']}",
        f"Latest run: {catalog['latest_run_id']}",
        f"Format: {catalog['data_format']}",
        f"Tables: {catalog['table_count']}",
        f"Rows observed: {catalog['row_count_observed']}",
        f"Scan truncated: {catalog['scan_truncated']}",
        f"Updated: {catalog['updated_at']}",
    ]
    ctx.console.print(Panel.fit("\n".join(summary_lines), title="Structured Catalog"))

    table = Table(title=f"Structured Tables ({len(tables)})")
    table.add_column("Table")
    table.add_column("Sheet")
    table.add_column("Rows")
    table.add_column("Columns")
    table.add_column("Truncated")
    for item in tables:
        cols = item.get("columns")
        table.add_row(
            str(item.get("table_name") or ""),
            str(item.get("sheet_name") or ""),
            str(item.get("row_count_observed") or 0),
            str(len(cols) if isinstance(cols, list) else 0),
            str(bool(item.get("scan_truncated"))),
        )
    ctx.console.print(table)
    return 0


def run_lookup_cell(args: argparse.Namespace, ctx: CLIContext) -> int:
    _require_initialized_project(ctx)
    service, repo = _service(ctx)
    resource_id = _resolve_resource_id(args, repo)
    filters = _parse_filters(args.filter or [])
    selector: dict[str, object] = {
        "value_column": str(args.value_column),
        "filters": filters,
    }
    if args.sheet_name:
        selector["sheet_name"] = str(args.sheet_name)

    match = service.resolve_data_cell(resource_id, selector)
    lines = [
        f"Resource ID: {match.resource_id}",
        f"Format: {match.data_format}",
        f"Table/Sheet: {match.table_name}",
        f"Row number: {match.row_number}",
        f"Column: {match.column_name} (index {match.column_index})",
        f"Value: {match.value_raw}",
        f"Filters: {match.filters}",
    ]
    ctx.console.print(Panel.fit("\n".join(lines), title="Structured Cell Lookup"))
    return 0


def _parse_filters(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        raw = str(item or "")
        if "=" not in raw:
            raise ValidationError(f"Invalid filter expression: {raw}. Expected KEY=VALUE.")
        key, value = raw.split("=", 1)
        k = key.strip()
        if not k:
            raise ValidationError(f"Invalid filter expression: {raw}. Empty key.")
        out[k] = value.strip()
    if not out:
        raise ValidationError("At least one --filter KEY=VALUE is required.")
    return out
