from __future__ import annotations

import argparse
from pathlib import Path

from rich.panel import Panel
from rich.table import Table

from stemmacodicum.application.services.claim_service import ClaimService
from stemmacodicum.application.services.project_service import ProjectService
from stemmacodicum.cli.context import CLIContext
from stemmacodicum.core.errors import ProjectNotInitializedError
from stemmacodicum.infrastructure.db.repos.claim_repo import ClaimRepo


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("claims", help="Claim set and claim management")
    claims_subparsers = parser.add_subparsers(dest="claims_command", required=True)

    import_parser = claims_subparsers.add_parser("import", help="Import claims from csv/json/markdown")
    import_parser.add_argument("--file", required=True)
    import_parser.add_argument("--format", required=True, choices=["csv", "json", "md", "markdown"])
    import_parser.add_argument("--claim-set", required=True, help="Claim set name")
    import_parser.add_argument("--description", help="Optional claim set description")
    import_parser.set_defaults(handler=run_import)

    list_parser = claims_subparsers.add_parser("list", help="List claims")
    list_parser.add_argument("--claim-set", help="Filter by claim set name")
    list_parser.add_argument("--limit", type=int, default=100)
    list_parser.set_defaults(handler=run_list)

    sets_parser = claims_subparsers.add_parser("sets", help="List claim sets")
    sets_parser.add_argument("--limit", type=int, default=100)
    sets_parser.set_defaults(handler=run_sets)


def _require_initialized_project(ctx: CLIContext) -> None:
    project_service = ProjectService(ctx.paths)
    if not project_service.is_initialized():
        raise ProjectNotInitializedError(
            f"Project is not initialized. Run 'stemma init' first in {ctx.paths.project_root}"
        )
    project_service.init_project()


def _service(ctx: CLIContext) -> ClaimService:
    return ClaimService(ClaimRepo(ctx.paths.db_path))


def run_import(args: argparse.Namespace, ctx: CLIContext) -> int:
    _require_initialized_project(ctx)
    service = _service(ctx)

    summary = service.import_claims(
        file_path=Path(args.file),
        fmt=args.format,
        claim_set_name=args.claim_set,
        claim_set_description=args.description,
    )

    ctx.console.print(
        Panel.fit(
            "\n".join(
                [
                    f"Claim set: {summary.claim_set_name}",
                    f"Claim set ID: {summary.claim_set_id}",
                    f"Imported: {summary.imported}",
                ]
            ),
            title="Claim Import Summary",
        )
    )
    return 0


def run_list(args: argparse.Namespace, ctx: CLIContext) -> int:
    _require_initialized_project(ctx)
    service = _service(ctx)
    claims = service.list_claims(claim_set_name=args.claim_set, limit=args.limit)

    out = Table(title=f"Claims ({len(claims)})")
    out.add_column("ID")
    out.add_column("Type")
    out.add_column("Subject", overflow="fold")
    out.add_column("Predicate")
    out.add_column("Narrative/Text", overflow="fold")
    out.add_column("Value")
    out.add_column("Period")

    for c in claims:
        text = c.narrative_text or c.object_text or ""
        value = c.value_raw if c.value_raw is not None else (str(c.value_parsed) if c.value_parsed is not None else "")
        out.add_row(c.id, c.claim_type, c.subject or "", c.predicate or "", text, value, c.period_label or "")

    ctx.console.print(out)
    return 0


def run_sets(args: argparse.Namespace, ctx: CLIContext) -> int:
    _require_initialized_project(ctx)
    service = _service(ctx)
    claim_sets = service.list_claim_sets(limit=args.limit)

    out = Table(title=f"Claim Sets ({len(claim_sets)})")
    out.add_column("ID")
    out.add_column("Name")
    out.add_column("Description", overflow="fold")
    out.add_column("Created")

    for s in claim_sets:
        out.add_row(s.id, s.name, s.description or "", s.created_at)

    ctx.console.print(out)
    return 0
