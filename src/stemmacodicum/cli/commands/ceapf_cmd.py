from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.panel import Panel
from rich.table import Table

from stemmacodicum.application.services.ceapf_service import CEAPFService
from stemmacodicum.application.services.project_service import ProjectService
from stemmacodicum.cli.context import CLIContext
from stemmacodicum.core.errors import ProjectNotInitializedError, ValidationError


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("ceapf", help="CEAPF proposition/assertion/argument operations")
    ceapf_subparsers = parser.add_subparsers(dest="ceapf_command", required=True)

    add_prop = ceapf_subparsers.add_parser("add-proposition", help="Add one proposition JSON object")
    group = add_prop.add_mutually_exclusive_group(required=True)
    group.add_argument("--json")
    group.add_argument("--json-file")
    add_prop.set_defaults(handler=run_add_proposition)

    add_assert = ceapf_subparsers.add_parser("add-assertion", help="Add one assertion event")
    add_assert.add_argument("--proposition-id", required=True)
    add_assert.add_argument("--agent", required=True)
    add_assert.add_argument("--modality", required=True, choices=["asserts", "denies", "speculates", "predicts", "recommends"])
    add_assert.add_argument("--evidence-id")
    add_assert.set_defaults(handler=run_add_assertion)

    add_rel = ceapf_subparsers.add_parser("add-relation", help="Add one argument relation")
    add_rel.add_argument("--type", required=True, choices=["supports", "rebuts", "undercuts", "qualifies"])
    add_rel.add_argument("--from-type", required=True)
    add_rel.add_argument("--from-id", required=True)
    add_rel.add_argument("--to-type", required=True)
    add_rel.add_argument("--to-id", required=True)
    add_rel.set_defaults(handler=run_add_relation)

    list_props = ceapf_subparsers.add_parser("list-propositions", help="List propositions")
    list_props.add_argument("--limit", type=int, default=50)
    list_props.set_defaults(handler=run_list_propositions)


def _require_initialized_project(ctx: CLIContext) -> CEAPFService:
    project_service = ProjectService(ctx.paths)
    if not project_service.is_initialized():
        raise ProjectNotInitializedError(
            f"Project is not initialized. Run 'stemma init' first in {ctx.paths.project_root}"
        )
    project_service.init_project()
    return CEAPFService(ctx.paths.db_path)


def _load_json(args: argparse.Namespace) -> dict[str, object]:
    if args.json_file:
        text = Path(args.json_file).read_text(encoding="utf-8")
    else:
        text = args.json

    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValidationError("Proposition JSON must be an object")
    return payload


def run_add_proposition(args: argparse.Namespace, ctx: CLIContext) -> int:
    service = _require_initialized_project(ctx)
    proposition = _load_json(args)
    prop_id = service.create_proposition(proposition)
    ctx.console.print(f"[green]Created[/green] proposition {prop_id}")
    return 0


def run_add_assertion(args: argparse.Namespace, ctx: CLIContext) -> int:
    service = _require_initialized_project(ctx)
    assertion_id = service.create_assertion_event(
        proposition_id=args.proposition_id,
        asserting_agent=args.agent,
        modality=args.modality,
        evidence_id=args.evidence_id,
    )
    ctx.console.print(f"[green]Created[/green] assertion event {assertion_id}")
    return 0


def run_add_relation(args: argparse.Namespace, ctx: CLIContext) -> int:
    service = _require_initialized_project(ctx)
    rel_id = service.add_argument_relation(
        relation_type=args.type,
        from_node_type=args.from_type,
        from_node_id=args.from_id,
        to_node_type=args.to_type,
        to_node_id=args.to_id,
    )
    ctx.console.print(f"[green]Created[/green] argument relation {rel_id}")
    return 0


def run_list_propositions(args: argparse.Namespace, ctx: CLIContext) -> int:
    service = _require_initialized_project(ctx)
    props = service.list_propositions(limit=args.limit)

    table = Table(title=f"Propositions ({len(props)})")
    table.add_column("ID")
    table.add_column("Payload", overflow="fold")
    for p in props:
        table.add_row(p.id, json.dumps(p.proposition, ensure_ascii=True))

    ctx.console.print(table)
    return 0
