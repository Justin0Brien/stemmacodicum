from __future__ import annotations

import argparse
from pathlib import Path

from rich.panel import Panel
from rich.table import Table

from stemmacodicum.application.services.project_service import ProjectService
from stemmacodicum.application.services.reference_service import ReferenceService
from stemmacodicum.cli.context import CLIContext
from stemmacodicum.core.errors import ProjectNotInitializedError
from stemmacodicum.infrastructure.db.repos.citation_repo import CitationRepo
from stemmacodicum.infrastructure.db.repos.reference_repo import ReferenceRepo
from stemmacodicum.infrastructure.db.repos.resource_repo import ResourceRepo


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("refs", help="Citation and reference management")
    refs_subparsers = parser.add_subparsers(dest="refs_command", required=True)

    import_bib = refs_subparsers.add_parser("import-bib", help="Import or update references from BibTeX")
    import_bib.add_argument("bib_path", help="Path to .bib file")
    import_bib.set_defaults(handler=run_import_bib)

    list_refs = refs_subparsers.add_parser("list", help="List references")
    list_refs.add_argument("--limit", type=int, default=50)
    list_refs.set_defaults(handler=run_list_refs)

    list_citations = refs_subparsers.add_parser("citations", help="List citation mappings")
    list_citations.add_argument("--limit", type=int, default=50)
    list_citations.set_defaults(handler=run_list_citations)

    link_resource = refs_subparsers.add_parser(
        "link-resource",
        help="Link a citation reference to an ingested resource by digest",
    )
    link_resource.add_argument("--cite-id", required=True)
    link_resource.add_argument("--resource-digest", required=True)
    link_resource.set_defaults(handler=run_link_resource)


def _require_initialized_project(ctx: CLIContext) -> None:
    project_service = ProjectService(ctx.paths)
    if not project_service.is_initialized():
        raise ProjectNotInitializedError(
            f"Project is not initialized. Run 'stemma init' first in {ctx.paths.project_root}"
        )
    # Ensure latest schema objects exist even on older DBs.
    project_service.init_project()


def _build_service(ctx: CLIContext) -> ReferenceService:
    citation_repo = CitationRepo(ctx.paths.db_path)
    reference_repo = ReferenceRepo(ctx.paths.db_path)
    resource_repo = ResourceRepo(ctx.paths.db_path)
    return ReferenceService(
        citation_repo=citation_repo,
        reference_repo=reference_repo,
        resource_repo=resource_repo,
    )


def run_import_bib(args: argparse.Namespace, ctx: CLIContext) -> int:
    _require_initialized_project(ctx)
    service = _build_service(ctx)

    summary = service.import_bibtex(Path(args.bib_path))

    panel = Panel.fit(
        "\n".join(
            [
                f"Entries seen: {summary.entries_seen}",
                f"Citation mappings created: {summary.mappings_created}",
                f"References inserted: {summary.references_inserted}",
                f"References updated: {summary.references_updated}",
            ]
        ),
        title="BibTeX Import Summary",
    )
    ctx.console.print(panel)
    return 0


def run_list_refs(args: argparse.Namespace, ctx: CLIContext) -> int:
    _require_initialized_project(ctx)

    refs = ReferenceRepo(ctx.paths.db_path).list(limit=args.limit)

    table = Table(title=f"References ({len(refs)})")
    table.add_column("Cite ID")
    table.add_column("Type")
    table.add_column("Year")
    table.add_column("Title", overflow="fold")
    table.add_column("DOI")
    table.add_column("URL", overflow="fold")

    for ref in refs:
        table.add_row(
            ref.cite_id,
            ref.entry_type,
            ref.year or "",
            ref.title or "",
            ref.doi or "",
            ref.url or "",
        )

    ctx.console.print(table)
    return 0


def run_list_citations(args: argparse.Namespace, ctx: CLIContext) -> int:
    _require_initialized_project(ctx)

    citations = CitationRepo(ctx.paths.db_path).list(limit=args.limit)
    table = Table(title=f"Citations ({len(citations)})")
    table.add_column("Cite ID")
    table.add_column("Original Key")
    table.add_column("Normalized Key")
    table.add_column("Created At")

    for citation in citations:
        table.add_row(
            citation.cite_id,
            citation.original_key,
            citation.normalized_key,
            citation.created_at,
        )

    ctx.console.print(table)
    return 0


def run_link_resource(args: argparse.Namespace, ctx: CLIContext) -> int:
    _require_initialized_project(ctx)
    service = _build_service(ctx)

    service.link_reference_to_resource(cite_id=args.cite_id, resource_digest=args.resource_digest)
    ctx.console.print(
        f"[green]Linked[/green] cite ID {args.cite_id} to resource digest {args.resource_digest}"
    )
    return 0
