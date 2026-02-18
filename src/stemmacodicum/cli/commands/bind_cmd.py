from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.panel import Panel

from stemmacodicum.application.services.evidence_binding_service import EvidenceBindingService
from stemmacodicum.application.services.project_service import ProjectService
from stemmacodicum.cli.context import CLIContext
from stemmacodicum.core.errors import ProjectNotInitializedError, ValidationError
from stemmacodicum.infrastructure.db.repos.claim_repo import ClaimRepo
from stemmacodicum.infrastructure.db.repos.evidence_repo import EvidenceRepo
from stemmacodicum.infrastructure.db.repos.resource_repo import ResourceRepo


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("bind", help="Bind claims to evidence items")
    bind_subparsers = parser.add_subparsers(dest="bind_command", required=True)

    add_parser = bind_subparsers.add_parser("add", help="Add one evidence item binding")
    add_parser.add_argument("--claim-id", required=True)
    resource_group = add_parser.add_mutually_exclusive_group(required=True)
    resource_group.add_argument("--resource-id")
    resource_group.add_argument("--resource-digest")
    add_parser.add_argument("--role", required=True)
    add_parser.add_argument("--page-index", type=int)
    add_parser.add_argument("--note")
    selector_group = add_parser.add_mutually_exclusive_group(required=True)
    selector_group.add_argument("--selectors-file", help="JSON file with an array of selector objects")
    selector_group.add_argument("--selectors-json", help="JSON array literal")
    add_parser.set_defaults(handler=run_add)

    validate_parser = bind_subparsers.add_parser("validate", help="Validate evidence completeness for a claim")
    validate_parser.add_argument("--claim-id", required=True)
    validate_parser.set_defaults(handler=run_validate)


def _require_initialized_project(ctx: CLIContext) -> None:
    project_service = ProjectService(ctx.paths)
    if not project_service.is_initialized():
        raise ProjectNotInitializedError(
            f"Project is not initialized. Run 'stemma init' first in {ctx.paths.project_root}"
        )
    project_service.init_project()


def _service(ctx: CLIContext) -> EvidenceBindingService:
    return EvidenceBindingService(
        claim_repo=ClaimRepo(ctx.paths.db_path),
        resource_repo=ResourceRepo(ctx.paths.db_path),
        evidence_repo=EvidenceRepo(ctx.paths.db_path),
    )


def _resolve_resource_id(args: argparse.Namespace, resource_repo: ResourceRepo) -> str:
    if args.resource_id:
        resource = resource_repo.get_by_id(args.resource_id)
        if resource is None:
            raise ValidationError(f"Resource not found: {args.resource_id}")
        return resource.id

    if not args.resource_digest:
        raise ValidationError("Provide --resource-id or --resource-digest")

    resource = resource_repo.get_by_digest(args.resource_digest)
    if resource is None:
        raise ValidationError(f"Resource digest not found: {args.resource_digest}")
    return resource.id


def _load_selectors(args: argparse.Namespace) -> list[dict[str, object]]:
    if args.selectors_file:
        payload = json.loads(Path(args.selectors_file).read_text(encoding="utf-8"))
    else:
        payload = json.loads(args.selectors_json)

    if not isinstance(payload, list) or not all(isinstance(x, dict) for x in payload):
        raise ValidationError("Selectors payload must be a JSON array of objects")

    return [dict(x) for x in payload]


def run_add(args: argparse.Namespace, ctx: CLIContext) -> int:
    _require_initialized_project(ctx)

    resource_repo = ResourceRepo(ctx.paths.db_path)
    resource_id = _resolve_resource_id(args, resource_repo)
    selectors = _load_selectors(args)

    service = _service(ctx)
    evidence_id = service.bind_evidence(
        claim_id=args.claim_id,
        resource_id=resource_id,
        role=args.role,
        selectors=selectors,
        page_index=args.page_index,
        note=args.note,
    )

    ctx.console.print(f"[green]Bound[/green] evidence item {evidence_id} to claim {args.claim_id}")
    return 0


def run_validate(args: argparse.Namespace, ctx: CLIContext) -> int:
    _require_initialized_project(ctx)

    service = _service(ctx)
    result = service.validate_binding(args.claim_id)

    lines = [f"Claim ID: {result.claim_id}", f"Validation: {'PASS' if result.ok else 'FAIL'}"]
    if result.missing_roles:
        lines.append(f"Missing roles: {', '.join(result.missing_roles)}")
    if result.evidence_with_too_few_selectors:
        lines.append(
            "Evidence with <2 selector types: "
            + ", ".join(result.evidence_with_too_few_selectors)
        )

    ctx.console.print(Panel.fit("\n".join(lines), title="Binding Validation"))
    return 0 if result.ok else 1
