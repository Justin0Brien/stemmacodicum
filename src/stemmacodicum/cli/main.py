from __future__ import annotations

import argparse
import logging
from pathlib import Path

from rich.console import Console

from stemmacodicum.cli.commands import (
    bind_cmd,
    ceapf_cmd,
    claims_cmd,
    doctor_cmd,
    extract_cmd,
    ingest_cmd,
    init_cmd,
    report_cmd,
    refs_cmd,
    resources_cmd,
    pipeline_cmd,
    trace_cmd,
    vector_cmd,
    verify_cmd,
    web_cmd,
)
from stemmacodicum.cli.context import CLIContext
from stemmacodicum.core.config import load_paths
from stemmacodicum.core.errors import StemmaError
from stemmacodicum.core.logging import configure_logging

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stemma",
        description="Stemma Codicum CLI",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root to use for .stemma data (default: current working directory)",
    )
    parser.add_argument("-v", "--verbose", action="count", default=0)

    subparsers = parser.add_subparsers(dest="command", required=True)
    init_cmd.register(subparsers)
    ingest_cmd.register(subparsers)
    resources_cmd.register(subparsers)
    refs_cmd.register(subparsers)
    extract_cmd.register(subparsers)
    claims_cmd.register(subparsers)
    bind_cmd.register(subparsers)
    verify_cmd.register(subparsers)
    report_cmd.register(subparsers)
    trace_cmd.register(subparsers)
    doctor_cmd.register(subparsers)
    ceapf_cmd.register(subparsers)
    pipeline_cmd.register(subparsers)
    vector_cmd.register(subparsers)
    web_cmd.register(subparsers)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    configure_logging(args.verbose)
    console = Console()

    paths = load_paths(args.project_root)
    ctx = CLIContext(paths=paths, console=console)

    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 2

    try:
        return handler(args, ctx)
    except StemmaError as exc:
        logger.error(str(exc))
        return 1
