from __future__ import annotations

import argparse

from stemmacodicum.cli.context import CLIContext
from stemmacodicum.web.app import create_app


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("web", help="Run the web GUI server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--reload", action="store_true")
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace, ctx: CLIContext) -> int:
    try:
        import uvicorn
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("uvicorn is required for web mode. Install project dependencies.") from exc

    app = create_app(ctx.paths)
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)
    return 0
