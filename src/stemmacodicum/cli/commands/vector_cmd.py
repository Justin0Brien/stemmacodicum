from __future__ import annotations

import argparse
import os
import time

from rich.panel import Panel
from rich.table import Table

from stemmacodicum.application.services.project_service import ProjectService
from stemmacodicum.application.services.vector_service import VectorIndexingService
from stemmacodicum.cli.context import CLIContext
from stemmacodicum.core.errors import ProjectNotInitializedError, ValidationError
from stemmacodicum.infrastructure.db.repos.extraction_repo import ExtractionRepo
from stemmacodicum.infrastructure.db.repos.resource_repo import ResourceRepo
from stemmacodicum.infrastructure.db.repos.vector_repo import VectorRepo
from stemmacodicum.infrastructure.vector.chunking import VectorChunker
from stemmacodicum.infrastructure.vector.embeddings import EmbeddingConfig, SentenceTransformerEmbedder
from stemmacodicum.infrastructure.vector.qdrant_store import QdrantLocalStore


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("vector", help="Vector indexing and semantic search")
    vector_subparsers = parser.add_subparsers(dest="vector_command", required=True)

    status = vector_subparsers.add_parser("status", help="Show vector indexing status")
    _add_resource_selectors(status, required=False)
    status.add_argument("--limit-runs", type=int, default=50)
    status.set_defaults(handler=run_status)

    search = vector_subparsers.add_parser("search", help="Semantic search over indexed chunks")
    search.add_argument("--query", required=True)
    search.add_argument("--limit", type=int, default=10)
    _add_resource_selectors(search, required=False)
    search.add_argument("--extraction-run-id")
    search.set_defaults(handler=run_search)

    index = vector_subparsers.add_parser("index", help="Index vectors for a resource extraction")
    _add_resource_selectors(index, required=True)
    index.add_argument("--extraction-run-id")
    index.add_argument("--force", action="store_true")
    index.set_defaults(handler=run_index)

    reindex = vector_subparsers.add_parser("reindex", help="Force reindex vectors for a resource extraction")
    _add_resource_selectors(reindex, required=True)
    reindex.add_argument("--extraction-run-id")
    reindex.set_defaults(handler=run_reindex)

    backfill = vector_subparsers.add_parser("backfill", help="Backfill vectors for latest extraction runs")
    backfill.add_argument("--limit-resources", type=int, default=100000)
    backfill.add_argument("--max-process", type=int, default=None)
    backfill.set_defaults(handler=run_backfill)

    prune = vector_subparsers.add_parser(
        "prune-structured",
        help="Remove structured-data (CSV/XLSX/XLS/ODS) vectors from Qdrant store",
    )
    prune.add_argument("--min-size-mb", type=float, default=0.0, help="Only prune resources >= this size (MB)")
    prune.add_argument("--limit-resources", type=int, default=100000)
    prune.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Preview candidates without deleting points (default: enabled).",
    )
    prune.set_defaults(handler=run_prune_structured)

    migrate = vector_subparsers.add_parser(
        "migrate-server",
        help="Copy vectors from local Qdrant storage to a Qdrant server and activate server mode",
    )
    migrate.add_argument("--url", default="http://127.0.0.1:6333", help="Qdrant server URL")
    migrate.add_argument("--api-key", default=None, help="Optional Qdrant API key")
    migrate.add_argument("--batch-size", type=int, default=2000)
    migrate.add_argument("--timeout-seconds", type=float, default=120.0)
    migrate.add_argument("--retry-attempts", type=int, default=3)
    migrate.add_argument(
        "--collection",
        default=None,
        help="Collection name override (defaults to STEMMA_QDRANT_COLLECTION or stemma_chunks).",
    )
    migrate.add_argument("--drop-existing", action="store_true", help="Drop destination collection before copy.")
    migrate.add_argument(
        "--activate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Persist server URL to .stemma/vector/qdrant_url.txt (default: enabled).",
    )
    migrate.set_defaults(handler=run_migrate_server)


def _add_resource_selectors(parser: argparse.ArgumentParser, *, required: bool) -> None:
    if required:
        group = parser.add_mutually_exclusive_group(required=True)
    else:
        group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--resource-id")
    group.add_argument("--resource-digest")


def _require_initialized_project(ctx: CLIContext) -> None:
    project_service = ProjectService(ctx.paths)
    if not project_service.is_initialized():
        raise ProjectNotInitializedError(
            f"Project is not initialized. Run 'stemma init' first in {ctx.paths.project_root}"
        )
    project_service.init_project()


def _service(ctx: CLIContext) -> VectorIndexingService:
    resource_repo = ResourceRepo(ctx.paths.db_path)
    extraction_repo = ExtractionRepo(ctx.paths.db_path)
    return VectorIndexingService(
        resource_repo=resource_repo,
        extraction_repo=extraction_repo,
        vector_repo=VectorRepo(ctx.paths.db_path),
        vector_store=QdrantLocalStore(storage_path=ctx.paths.qdrant_dir),
        embedder=SentenceTransformerEmbedder(config=EmbeddingConfig()),
        chunker=VectorChunker(),
    )


def _resolve_resource_id(
    args: argparse.Namespace,
    *,
    resource_repo: ResourceRepo,
) -> str | None:
    if getattr(args, "resource_id", None):
        resource = resource_repo.get_by_id(args.resource_id)
        if resource is None:
            raise ValidationError(f"Resource ID not found: {args.resource_id}")
        return resource.id
    if getattr(args, "resource_digest", None):
        resource = resource_repo.get_by_digest(args.resource_digest)
        if resource is None:
            raise ValidationError(f"Resource digest not found: {args.resource_digest}")
        return resource.id
    return None


def run_status(args: argparse.Namespace, ctx: CLIContext) -> int:
    _require_initialized_project(ctx)
    resource_repo = ResourceRepo(ctx.paths.db_path)
    resource_id = _resolve_resource_id(args, resource_repo=resource_repo)
    payload = _service(ctx).status(resource_id=resource_id, limit_runs=args.limit_runs)

    table = Table(title="Vector Status")
    table.add_column("Run ID")
    table.add_column("Resource")
    table.add_column("Extraction Run")
    table.add_column("Status")
    table.add_column("Chunks")
    table.add_column("Error", overflow="fold")

    for run in payload["runs"]:
        table.add_row(
            str(run["id"]),
            str(run["resource_id"]),
            str(run["extraction_run_id"]),
            str(run["status"]),
            f"{run['chunks_indexed']}/{run['chunks_total']}",
            str(run["error_message"] or ""),
        )

    ctx.console.print(table)
    ctx.console.print(
        Panel.fit(
            "\n".join(
                [
                    f"Backend: {payload['backend']}",
                    f"Collection: {payload['collection_name']}",
                    f"Embedding model: {payload['embedding_model']}",
                    f"Vector chunk rows: {payload['vector_chunk_rows']}",
                    f"Distinct chunk IDs: {payload['distinct_chunk_ids']}",
                    f"Qdrant points: {payload['qdrant_points']}",
                ]
            ),
            title="Vector Store",
        )
    )
    return 0


def run_search(args: argparse.Namespace, ctx: CLIContext) -> int:
    _require_initialized_project(ctx)
    resource_repo = ResourceRepo(ctx.paths.db_path)
    resource_id = _resolve_resource_id(args, resource_repo=resource_repo)
    hits = _service(ctx).search(
        query=args.query,
        limit=args.limit,
        resource_id=resource_id,
        extraction_run_id=args.extraction_run_id,
    )

    table = Table(title=f"Vector Search Hits ({len(hits)})")
    table.add_column("Score")
    table.add_column("Resource")
    table.add_column("Source")
    table.add_column("Page")
    table.add_column("Text", overflow="fold")
    for hit in hits:
        text = str(hit.get("text_content") or "")
        snippet = text[:220] + ("..." if len(text) > 220 else "")
        table.add_row(
            f"{float(hit.get('score', 0.0)):.4f}",
            str(hit.get("resource_id") or ""),
            str(hit.get("source_ref") or hit.get("source_type") or ""),
            str(hit.get("page_index") or ""),
            snippet,
        )
    ctx.console.print(table)
    return 0


def run_index(args: argparse.Namespace, ctx: CLIContext) -> int:
    _require_initialized_project(ctx)
    resource_repo = ResourceRepo(ctx.paths.db_path)
    extraction_repo = ExtractionRepo(ctx.paths.db_path)
    resource_id = _resolve_resource_id(args, resource_repo=resource_repo)
    if not resource_id:
        raise ValidationError("Provide --resource-id or --resource-digest.")

    extraction_run_id = args.extraction_run_id
    if extraction_run_id is None:
        latest = extraction_repo.get_latest_run(resource_id)
        if latest is None:
            raise ValidationError("No extraction run found for resource. Run `stemma extract run` first.")
        extraction_run_id = latest.id

    summary = _service(ctx).index_extraction(
        resource_id=resource_id,
        extraction_run_id=extraction_run_id,
        force=bool(args.force),
    )
    ctx.console.print(
        Panel.fit(
            "\n".join(
                [
                    f"Run ID: {summary.run_id}",
                    f"Status: {summary.status}",
                    f"Resource ID: {summary.resource_id}",
                    f"Extraction Run ID: {summary.extraction_run_id}",
                    f"Chunks indexed: {summary.chunks_indexed}/{summary.chunks_total}",
                    f"Embedding model: {summary.embedding_model}",
                    f"Embedding dim: {summary.embedding_dim if summary.embedding_dim is not None else 'n/a'}",
                    f"Error: {summary.error or 'none'}",
                ]
            ),
            title="Vector Index",
        )
    )
    return 0 if summary.status in {"success", "skipped"} else 1


def run_reindex(args: argparse.Namespace, ctx: CLIContext) -> int:
    args.force = True
    return run_index(args, ctx)


def run_backfill(args: argparse.Namespace, ctx: CLIContext) -> int:
    _require_initialized_project(ctx)
    summary = _service(ctx).backfill_latest(
        limit_resources=args.limit_resources,
        max_process=args.max_process,
    )
    ctx.console.print(
        Panel.fit(
            "\n".join(
                [
                    f"Candidates: {summary.candidates}",
                    f"Processed: {summary.processed}",
                    f"Indexed: {summary.indexed}",
                    f"Skipped: {summary.skipped}",
                    f"Failed: {summary.failed}",
                ]
            ),
            title="Vector Backfill",
        )
    )
    return 0 if summary.failed == 0 else 1


def run_prune_structured(args: argparse.Namespace, ctx: CLIContext) -> int:
    _require_initialized_project(ctx)
    min_size_bytes = max(0, int(float(args.min_size_mb) * 1024 * 1024))
    summary = _service(ctx).prune_structured_vectors(
        min_size_bytes=min_size_bytes,
        limit_resources=int(args.limit_resources),
        dry_run=bool(args.dry_run),
    )

    overview_lines = [
        f"Dry run: {summary.dry_run}",
        f"Candidates: {summary.candidates}",
        f"Points before: {summary.points_before}",
        f"Points removed: {summary.points_removed}",
        f"Points after: {summary.points_after}",
        f"Runs marked pruned: {summary.runs_marked_pruned}",
    ]
    ctx.console.print(Panel.fit("\n".join(overview_lines), title="Structured Vector Prune"))

    if summary.resources:
        table = Table(title=f"Structured Vector Candidates ({len(summary.resources)})")
        table.add_column("Resource ID")
        table.add_column("Media")
        table.add_column("Filename", overflow="fold")
        table.add_column("Size bytes")
        table.add_column("Chunk rows")
        for item in summary.resources[:200]:
            table.add_row(
                str(item.get("resource_id") or ""),
                str(item.get("media_type") or ""),
                str(item.get("original_filename") or ""),
                str(item.get("size_bytes") or 0),
                str(item.get("chunk_rows") or 0),
            )
        ctx.console.print(table)
    return 0


def run_migrate_server(args: argparse.Namespace, ctx: CLIContext) -> int:
    _require_initialized_project(ctx)

    try:
        from qdrant_client import QdrantClient
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ValidationError(
            "Qdrant dependency is missing. Install with `pip install -e '.[vector]'`."
        ) from exc

    collection_name = args.collection or os.getenv("STEMMA_QDRANT_COLLECTION") or "stemma_chunks"
    local_path = ctx.paths.qdrant_dir
    remote_url = str(args.url).strip()
    if not remote_url:
        raise ValidationError("Qdrant server URL cannot be empty.")

    local_client = QdrantClient(path=str(local_path))
    remote_client = QdrantClient(
        url=remote_url,
        api_key=(args.api_key or os.getenv("STEMMA_QDRANT_API_KEY")),
        timeout=float(args.timeout_seconds),
    )

    if not local_client.collection_exists(collection_name=collection_name):
        raise ValidationError(
            f"Local collection not found at {local_path}: {collection_name}"
        )

    source_info = local_client.get_collection(collection_name=collection_name)
    if remote_client.collection_exists(collection_name=collection_name):
        if args.drop_existing:
            remote_client.delete_collection(collection_name=collection_name)
        else:
            # Ensure destination vector config is compatible before writing.
            dest_info = remote_client.get_collection(collection_name=collection_name)
            src_vectors = getattr(getattr(source_info, "config", None), "params", None)
            dst_vectors = getattr(getattr(dest_info, "config", None), "params", None)
            src_size = getattr(getattr(src_vectors, "vectors", None), "size", None)
            dst_size = getattr(getattr(dst_vectors, "vectors", None), "size", None)
            if src_size is not None and dst_size is not None and int(src_size) != int(dst_size):
                raise ValidationError(
                    "Destination collection exists with incompatible vector size. "
                    "Use --drop-existing to recreate it."
                )
    if not remote_client.collection_exists(collection_name=collection_name):
        source_vectors_config = getattr(getattr(source_info, "config", None), "params", None)
        vectors = getattr(source_vectors_config, "vectors", None)
        if vectors is None:
            raise ValidationError("Unable to read source collection vector config.")
        remote_client.create_collection(
            collection_name=collection_name,
            vectors_config=vectors,
        )

    started = time.perf_counter()
    moved = 0
    offset = None
    batch_size = max(1, int(args.batch_size))
    while True:
        points, next_offset = local_client.scroll(
            collection_name=collection_name,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        if not points:
            break
        upload_points = [
            {
                "id": point.id,
                "vector": point.vector,
                "payload": point.payload,
            }
            for point in points
        ]
        attempt = 0
        while True:
            attempt += 1
            try:
                remote_client.upsert(collection_name=collection_name, points=upload_points, wait=True)
                break
            except Exception:
                if attempt >= max(1, int(args.retry_attempts)):
                    raise
                time.sleep(min(5.0, attempt * 1.5))
        moved += len(points)
        if next_offset is None:
            break
        offset = next_offset

    elapsed = time.perf_counter() - started
    local_count = int(local_client.count(collection_name=collection_name, exact=True).count)
    remote_count = int(remote_client.count(collection_name=collection_name, exact=True).count)

    config_path = ctx.paths.vector_dir / "qdrant_url.txt"
    if args.activate:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(remote_url + "\n", encoding="utf-8")

    ctx.console.print(
        Panel.fit(
            "\n".join(
                [
                    f"Collection: {collection_name}",
                    f"Source (local): {local_path}",
                    f"Destination (server): {remote_url}",
                    f"Points copied this run: {moved}",
                    f"Local points: {local_count}",
                    f"Server points: {remote_count}",
                    f"Elapsed: {elapsed:.2f}s",
                    f"Activated server mode: {'yes' if args.activate else 'no'}",
                    (f"Config file: {config_path}" if args.activate else "Config file: unchanged"),
                ]
            ),
            title="Qdrant Migration",
        )
    )

    if local_count != remote_count:
        return 1
    return 0
