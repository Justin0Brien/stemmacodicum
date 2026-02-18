from __future__ import annotations

import argparse
from pathlib import Path

from rich.panel import Panel
from rich.table import Table

from stemmacodicum.application.services.project_service import ProjectService
from stemmacodicum.application.services.reporting_service import ReportingService
from stemmacodicum.cli.context import CLIContext
from stemmacodicum.core.errors import ProjectNotInitializedError
from stemmacodicum.infrastructure.db.repos.verification_repo import VerificationRepo


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("report", help="Verification reporting")
    report_subparsers = parser.add_subparsers(dest="report_command", required=True)

    verification = report_subparsers.add_parser("verification", help="Show/export one verification run")
    verification.add_argument("--run-id", required=True)
    verification.add_argument("--json-out")
    verification.add_argument("--md-out")
    verification.set_defaults(handler=run_verification_report)


def _require_initialized_project(ctx: CLIContext) -> None:
    project_service = ProjectService(ctx.paths)
    if not project_service.is_initialized():
        raise ProjectNotInitializedError(
            f"Project is not initialized. Run 'stemma init' first in {ctx.paths.project_root}"
        )
    project_service.init_project()


def run_verification_report(args: argparse.Namespace, ctx: CLIContext) -> int:
    _require_initialized_project(ctx)

    service = ReportingService(VerificationRepo(ctx.paths.db_path))
    summary = service.build_run_summary(args.run_id)

    table = Table(title=f"Verification Run {summary.run_id}")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Policy", summary.policy_profile)
    table.add_row("Created", summary.created_at)
    table.add_row("Total", str(summary.total))
    table.add_row("Passed", str(summary.passed))
    table.add_row("Failed", str(summary.failed))
    ctx.console.print(table)

    if args.json_out:
        out = service.export_json_report(summary.run_id, Path(args.json_out))
        ctx.console.print(f"[green]JSON report written[/green] {out}")

    if args.md_out:
        out = service.export_markdown_report(summary.run_id, Path(args.md_out))
        ctx.console.print(f"[green]Markdown report written[/green] {out}")

    if summary.failed:
        first_fail = next((r for r in summary.results if r.get("status") != "pass"), None)
        if first_fail:
            ctx.console.print(
                Panel.fit(
                    f"First failure claim: {first_fail['claim_id']}\n"
                    f"Diagnostics: {first_fail['diagnostics']}",
                    title="Failure Snapshot",
                )
            )

    return 0
