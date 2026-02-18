from __future__ import annotations

import argparse

from rich.panel import Panel
from rich.table import Table

from stemmacodicum.application.services.health_service import HealthService
from stemmacodicum.application.services.project_service import ProjectService
from stemmacodicum.cli.context import CLIContext
from stemmacodicum.core.errors import ProjectNotInitializedError


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("doctor", help="Run integrity and consistency checks")
    parser.set_defaults(handler=run_doctor)


def run_doctor(args: argparse.Namespace, ctx: CLIContext) -> int:
    project_service = ProjectService(ctx.paths)
    if not project_service.is_initialized():
        raise ProjectNotInitializedError(
            f"Project is not initialized. Run 'stemma init' first in {ctx.paths.project_root}"
        )
    project_service.init_project()

    service = HealthService(db_path=ctx.paths.db_path, archive_dir=ctx.paths.archive_dir)
    report = service.run_doctor()

    summary = Panel.fit(
        f"Checks run: {report.checks_run}\n"
        f"Issues: {len(report.issues)}\n"
        f"Status: {'PASS' if report.ok else 'FAIL'}",
        title="Doctor Summary",
    )
    ctx.console.print(summary)

    runtime = Table(title="Database Runtime")
    runtime.add_column("Setting")
    runtime.add_column("Value", overflow="fold")
    for key, value in report.db_runtime.items():
        runtime.add_row(str(key), str(value))
    ctx.console.print(runtime)

    if report.issues:
        out = Table(title="Doctor Issues")
        out.add_column("Level")
        out.add_column("Check")
        out.add_column("Message", overflow="fold")
        for issue in report.issues:
            out.add_row(issue.level, issue.check, issue.message)
        ctx.console.print(out)

    return 0 if report.ok else 1
