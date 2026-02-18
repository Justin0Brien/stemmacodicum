from __future__ import annotations

import argparse
import json

from rich.panel import Panel
from rich.table import Table

from stemmacodicum.application.services.project_service import ProjectService
from stemmacodicum.application.services.trace_service import TraceService
from stemmacodicum.cli.context import CLIContext
from stemmacodicum.core.errors import ProjectNotInitializedError


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("trace", help="Trace claim/resource/citation chains")
    trace_subparsers = parser.add_subparsers(dest="trace_command", required=True)

    claim = trace_subparsers.add_parser("claim", help="Trace one claim")
    claim.add_argument("--claim-id", required=True)
    claim.set_defaults(handler=run_trace_claim)

    resource = trace_subparsers.add_parser("resource", help="Trace one resource")
    group = resource.add_mutually_exclusive_group(required=True)
    group.add_argument("--resource-id")
    group.add_argument("--resource-digest")
    resource.set_defaults(handler=run_trace_resource)

    citation = trace_subparsers.add_parser("citation", help="Trace one citation")
    citation.add_argument("--cite-id", required=True)
    citation.set_defaults(handler=run_trace_citation)


def _require_initialized_project(ctx: CLIContext) -> None:
    project_service = ProjectService(ctx.paths)
    if not project_service.is_initialized():
        raise ProjectNotInitializedError(
            f"Project is not initialized. Run 'stemma init' first in {ctx.paths.project_root}"
        )
    project_service.init_project()


def run_trace_claim(args: argparse.Namespace, ctx: CLIContext) -> int:
    _require_initialized_project(ctx)
    service = TraceService(ctx.paths.db_path)

    trace = service.trace_claim(args.claim_id)

    ctx.console.print(
        Panel.fit(
            f"Claim ID: {trace.claim_id}\n"
            f"Claim Type: {trace.claim_type}\n"
            f"Text: {trace.claim_text}\n"
            f"Evidence items: {len(trace.evidence)}",
            title="Claim Trace",
        )
    )

    table = Table(title="Evidence")
    table.add_column("Evidence ID")
    table.add_column("Role")
    table.add_column("Resource ID")
    table.add_column("Selectors")
    for e in trace.evidence:
        table.add_row(
            str(e["evidence_id"]),
            str(e["role"]),
            str(e["resource_id"]),
            json.dumps(e["selectors"]),
        )
    ctx.console.print(table)
    return 0


def run_trace_resource(args: argparse.Namespace, ctx: CLIContext) -> int:
    _require_initialized_project(ctx)
    service = TraceService(ctx.paths.db_path)

    trace = service.trace_resource(resource_id=args.resource_id, digest_sha256=args.resource_digest)

    ctx.console.print(
        Panel.fit(
            f"Resource ID: {trace.resource_id}\n"
            f"Digest: {trace.digest_sha256}\n"
            f"Linked references: {len(trace.references)}\n"
            f"Linked claims: {len(trace.claims)}",
            title="Resource Trace",
        )
    )
    return 0


def run_trace_citation(args: argparse.Namespace, ctx: CLIContext) -> int:
    _require_initialized_project(ctx)
    service = TraceService(ctx.paths.db_path)

    trace = service.trace_citation(args.cite_id)

    ref = trace.reference or {}
    ctx.console.print(
        Panel.fit(
            f"Cite ID: {trace.cite_id}\n"
            f"Title: {ref.get('title', '')}\n"
            f"Year: {ref.get('year', '')}\n"
            f"Resources: {len(trace.resources)}\n"
            f"Claims: {len(trace.claims)}",
            title="Citation Trace",
        )
    )
    return 0
