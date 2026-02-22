from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from stemmacodicum.core.ids import new_uuid
from stemmacodicum.core.time import now_utc_iso
from stemmacodicum.domain.models.vector import VectorChunk, VectorIndexRun
from stemmacodicum.infrastructure.db.repos.extraction_repo import ExtractionRepo
from stemmacodicum.infrastructure.db.repos.resource_repo import ResourceRepo
from stemmacodicum.infrastructure.db.repos.vector_repo import VectorRepo
from stemmacodicum.infrastructure.vector.chunking import ChunkDraft, VectorChunker
from stemmacodicum.infrastructure.vector.embeddings import SentenceTransformerEmbedder
from stemmacodicum.infrastructure.vector.qdrant_store import QdrantLocalStore, VectorPoint


@dataclass(slots=True)
class VectorIndexSummary:
    run_id: str
    resource_id: str
    extraction_run_id: str
    status: str
    chunks_total: int
    chunks_indexed: int
    embedding_model: str
    embedding_dim: int | None
    error: str | None = None


@dataclass(slots=True)
class VectorBackfillSummary:
    candidates: int
    processed: int
    indexed: int
    skipped: int
    failed: int


@dataclass(slots=True)
class VectorPruneSummary:
    candidates: int
    points_removed: int
    points_before: int
    points_after: int
    runs_marked_pruned: int
    dry_run: bool
    resources: list[dict[str, object]]


class VectorIndexingService:
    STRUCTURED_VECTOR_MEDIA_TYPES = [
        "text/csv",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.ms-excel",
        "application/vnd.oasis.opendocument.spreadsheet",
    ]

    def __init__(
        self,
        *,
        resource_repo: ResourceRepo,
        extraction_repo: ExtractionRepo,
        vector_repo: VectorRepo,
        vector_store: QdrantLocalStore,
        embedder: SentenceTransformerEmbedder,
        chunker: VectorChunker,
    ) -> None:
        self.resource_repo = resource_repo
        self.extraction_repo = extraction_repo
        self.vector_repo = vector_repo
        self.vector_store = vector_store
        self.embedder = embedder
        self.chunker = chunker

    def index_extraction(
        self,
        *,
        resource_id: str,
        extraction_run_id: str,
        force: bool = False,
        progress_callback: Callable[[dict[str, object]], None] | None = None,
    ) -> VectorIndexSummary:
        def emit(payload: dict[str, object]) -> None:
            if progress_callback is None:
                return
            progress_callback(payload)

        emit(
            {
                "stage": "vector",
                "state": "active",
                "progress": 3,
                "detail": "Preparing vector indexing.",
            }
        )
        try:
            ensure_ready = getattr(self.vector_store, "ensure_ready", None)
            if callable(ensure_ready):
                ensure_ready()
        except Exception as exc:
            emit(
                {
                    "stage": "vector",
                    "state": "error",
                    "progress": 100,
                    "detail": f"Vector store initialization failed: {exc}",
                }
            )
            return VectorIndexSummary(
                run_id="",
                resource_id=resource_id,
                extraction_run_id=extraction_run_id,
                status="failed",
                chunks_total=0,
                chunks_indexed=0,
                embedding_model=self.embedder.model_name,
                embedding_dim=None,
                error=str(exc),
            )
        extraction_run = self.extraction_repo.get_run_by_id(extraction_run_id)
        if extraction_run is None:
            emit(
                {
                    "stage": "vector",
                    "state": "error",
                    "progress": 100,
                    "detail": f"Extraction run not found: {extraction_run_id}",
                }
            )
            return VectorIndexSummary(
                run_id="",
                resource_id=resource_id,
                extraction_run_id=extraction_run_id,
                status="failed",
                chunks_total=0,
                chunks_indexed=0,
                embedding_model=self.embedder.model_name,
                embedding_dim=None,
                error=f"Extraction run not found: {extraction_run_id}",
            )
        if extraction_run.resource_id != resource_id:
            emit(
                {
                    "stage": "vector",
                    "state": "error",
                    "progress": 100,
                    "detail": (
                        f"Extraction run {extraction_run_id} belongs to resource "
                        f"{extraction_run.resource_id}, not {resource_id}"
                    ),
                }
            )
            return VectorIndexSummary(
                run_id="",
                resource_id=resource_id,
                extraction_run_id=extraction_run_id,
                status="failed",
                chunks_total=0,
                chunks_indexed=0,
                embedding_model=self.embedder.model_name,
                embedding_dim=None,
                error=(
                    f"Extraction run {extraction_run_id} belongs to resource "
                    f"{extraction_run.resource_id}, not {resource_id}"
                ),
            )

        existing = self.vector_repo.get_latest_run_for_extraction(
            extraction_run_id=extraction_run_id,
            vector_backend=self.vector_store.backend_name,
            embedding_model=self.embedder.model_name,
            chunking_version=self.chunker.chunking_version,
            collection_name=self.vector_store.collection_name,
        )
        if existing is not None and existing.status == "success" and not force:
            emit(
                {
                    "stage": "vector",
                    "state": "done",
                    "progress": 100,
                    "detail": "Existing vector index reused.",
                    "stats": f"{existing.chunks_indexed}/{existing.chunks_total} chunks",
                }
            )
            return VectorIndexSummary(
                run_id=existing.id,
                resource_id=resource_id,
                extraction_run_id=extraction_run_id,
                status="skipped",
                chunks_total=existing.chunks_total,
                chunks_indexed=existing.chunks_indexed,
                embedding_model=existing.embedding_model,
                embedding_dim=existing.embedding_dim,
                error=None,
            )

        run_id = new_uuid()
        run = VectorIndexRun(
            id=run_id,
            resource_id=resource_id,
            extraction_run_id=extraction_run_id,
            vector_backend=self.vector_store.backend_name,
            collection_name=self.vector_store.collection_name,
            embedding_model=self.embedder.model_name,
            embedding_dim=None,
            chunking_version=self.chunker.chunking_version,
            status="running",
            chunks_total=0,
            chunks_indexed=0,
            error_message=None,
            created_at=now_utc_iso(),
            finished_at=None,
        )
        self.vector_repo.insert_run(run)

        indexed = 0
        chunks_total = 0
        embedding_dim: int | None = None

        try:
            doc_text = self.extraction_repo.get_document_text_for_run(extraction_run_id)
            segments = self.extraction_repo.list_segments_for_resource(
                resource_id=resource_id,
                extraction_run_id=extraction_run_id,
                limit=250_000,
            )
            tables = self.extraction_repo.list_tables_for_run(extraction_run_id=extraction_run_id, limit=25_000)
            chunk_drafts = self.chunker.build_chunks(
                resource_id=resource_id,
                extraction_run_id=extraction_run_id,
                document_text=doc_text,
                segments=segments,
                tables=tables,
            )
            chunks_total = len(chunk_drafts)
            emit(
                {
                    "stage": "vector",
                    "state": "active",
                    "progress": 12,
                    "detail": "Chunk set prepared for embedding.",
                    "stats": f"{chunks_total} chunks total",
                }
            )

            if not chunk_drafts:
                self.vector_repo.finalize_run(
                    run_id,
                    status="success",
                    chunks_total=0,
                    chunks_indexed=0,
                    embedding_dim=None,
                    error_message=None,
                    finished_at=now_utc_iso(),
                )
                emit(
                    {
                        "stage": "vector",
                        "state": "done",
                        "progress": 100,
                        "detail": "No vector chunks were produced for this extraction.",
                        "stats": "0/0 chunks",
                    }
                )
                return VectorIndexSummary(
                    run_id=run_id,
                    resource_id=resource_id,
                    extraction_run_id=extraction_run_id,
                    status="success",
                    chunks_total=0,
                    chunks_indexed=0,
                    embedding_model=self.embedder.model_name,
                    embedding_dim=None,
                    error=None,
                )

            embedding_dim = self.embedder.embedding_dim()
            self.vector_store.ensure_collection(embedding_dim)

            batch_size = max(8, min(256, getattr(self.embedder.config, "batch_size", 128)))
            for offset in range(0, len(chunk_drafts), batch_size):
                batch = chunk_drafts[offset : offset + batch_size]
                batch_vectors = self.embedder.embed_texts([c.text_content for c in batch])
                points = [
                    VectorPoint(
                        point_id=chunk.vector_point_id,
                        vector=vector,
                        payload={
                            "chunk_id": chunk.chunk_id,
                            "resource_id": resource_id,
                            "extraction_run_id": extraction_run_id,
                            "embedding_model": self.embedder.model_name,
                            "chunking_version": self.chunker.chunking_version,
                            "source_type": chunk.source_type,
                            "source_ref": chunk.source_ref,
                            "page_index": chunk.page_index,
                            "start_offset": chunk.start_offset,
                            "end_offset": chunk.end_offset,
                            "token_count_est": chunk.token_count_est,
                            "text_digest_sha256": chunk.text_digest_sha256,
                            "text_content": chunk.text_content,
                        },
                    )
                    for chunk, vector in zip(batch, batch_vectors, strict=True)
                ]
                self.vector_store.upsert_points(points)
                self.vector_repo.insert_chunks(
                    [
                        self._to_vector_chunk(
                            run_id=run_id,
                            resource_id=resource_id,
                            extraction_run_id=extraction_run_id,
                            chunk=chunk,
                            embedding_dim=embedding_dim,
                            vector_backend=self.vector_store.backend_name,
                            collection_name=self.vector_store.collection_name,
                        )
                        for chunk in batch
                    ]
                )
                indexed += len(batch)
                progress = 12 + int((indexed / max(chunks_total, 1)) * 82)
                emit(
                    {
                        "stage": "vector",
                        "state": "active",
                        "progress": min(98, progress),
                        "detail": "Embedding and writing vector points.",
                        "stats": f"{indexed}/{chunks_total} chunks",
                    }
                )

            self.vector_repo.finalize_run(
                run_id,
                status="success",
                chunks_total=chunks_total,
                chunks_indexed=indexed,
                embedding_dim=embedding_dim,
                error_message=None,
                finished_at=now_utc_iso(),
            )
            emit(
                {
                    "stage": "vector",
                    "state": "done",
                    "progress": 100,
                    "detail": "Vector indexing complete.",
                    "stats": f"{indexed}/{chunks_total} chunks",
                }
            )
            return VectorIndexSummary(
                run_id=run_id,
                resource_id=resource_id,
                extraction_run_id=extraction_run_id,
                status="success",
                chunks_total=chunks_total,
                chunks_indexed=indexed,
                embedding_model=self.embedder.model_name,
                embedding_dim=embedding_dim,
                error=None,
            )
        except Exception as exc:
            self.vector_repo.finalize_run(
                run_id,
                status="failed",
                chunks_total=chunks_total,
                chunks_indexed=indexed,
                embedding_dim=embedding_dim,
                error_message=str(exc),
                finished_at=now_utc_iso(),
            )
            emit(
                {
                    "stage": "vector",
                    "state": "error",
                    "progress": 100,
                    "detail": f"Vector indexing failed: {exc}",
                    "stats": f"{indexed}/{chunks_total} chunks",
                }
            )
            return VectorIndexSummary(
                run_id=run_id,
                resource_id=resource_id,
                extraction_run_id=extraction_run_id,
                status="failed",
                chunks_total=chunks_total,
                chunks_indexed=indexed,
                embedding_model=self.embedder.model_name,
                embedding_dim=embedding_dim,
                error=str(exc),
            )

    def index_latest_for_resource(self, resource_id: str, *, force: bool = False) -> VectorIndexSummary:
        run = self.extraction_repo.get_latest_run(resource_id)
        if run is None:
            return VectorIndexSummary(
                run_id="",
                resource_id=resource_id,
                extraction_run_id="",
                status="failed",
                chunks_total=0,
                chunks_indexed=0,
                embedding_model=self.embedder.model_name,
                embedding_dim=None,
                error="No extraction run found for resource.",
            )
        return self.index_extraction(resource_id=resource_id, extraction_run_id=run.id, force=force)

    def backfill_latest(self, *, limit_resources: int = 100000, max_process: int | None = None) -> VectorBackfillSummary:
        candidates = self.vector_repo.list_runs_missing_success_for_latest_extraction(
            vector_backend=self.vector_store.backend_name,
            embedding_model=self.embedder.model_name,
            chunking_version=self.chunker.chunking_version,
            collection_name=self.vector_store.collection_name,
            limit=limit_resources,
        )
        work = candidates[:max_process] if max_process is not None else candidates
        processed = 0
        indexed = 0
        skipped = 0
        failed = 0
        for candidate in work:
            summary = self.index_extraction(
                resource_id=candidate["resource_id"],
                extraction_run_id=candidate["extraction_run_id"],
                force=False,
            )
            processed += 1
            if summary.status == "success":
                indexed += 1
            elif summary.status == "skipped":
                skipped += 1
            else:
                failed += 1
        return VectorBackfillSummary(
            candidates=len(candidates),
            processed=processed,
            indexed=indexed,
            skipped=skipped,
            failed=failed,
        )

    def search(
        self,
        *,
        query: str,
        limit: int = 10,
        resource_id: str | None = None,
        extraction_run_id: str | None = None,
    ) -> list[dict[str, object]]:
        text = query.strip()
        if not text:
            return []
        vector = self.embedder.embed_texts([text])[0]
        hits = self.vector_store.search(
            query_vector=vector,
            limit=limit,
            resource_id=resource_id,
            extraction_run_id=extraction_run_id,
            embedding_model=self.embedder.model_name,
            chunking_version=self.chunker.chunking_version,
        )
        return [
            {
                "id": item["id"],
                "score": item["score"],
                "chunk_id": item["payload"].get("chunk_id"),
                "resource_id": item["payload"].get("resource_id"),
                "extraction_run_id": item["payload"].get("extraction_run_id"),
                "source_type": item["payload"].get("source_type"),
                "source_ref": item["payload"].get("source_ref"),
                "page_index": item["payload"].get("page_index"),
                "start_offset": item["payload"].get("start_offset"),
                "end_offset": item["payload"].get("end_offset"),
                "text_content": item["payload"].get("text_content"),
            }
            for item in hits
        ]

    def status(self, *, resource_id: str | None = None, limit_runs: int = 50) -> dict[str, object]:
        if resource_id:
            runs = self.vector_repo.list_recent_runs_for_resource(resource_id=resource_id, limit=limit_runs)
        else:
            runs = self.vector_repo.list_recent_runs(limit=limit_runs)
        try:
            qdrant_points = self.vector_store.count_points()
        except Exception:
            qdrant_points = 0
        return {
            "backend": self.vector_store.backend_name,
            "collection_name": self.vector_store.collection_name,
            "embedding_model": self.embedder.model_name,
            "runs": [
                {
                    "id": run.id,
                    "resource_id": run.resource_id,
                    "extraction_run_id": run.extraction_run_id,
                    "status": run.status,
                    "chunks_total": run.chunks_total,
                    "chunks_indexed": run.chunks_indexed,
                    "error_message": run.error_message,
                    "created_at": run.created_at,
                    "finished_at": run.finished_at,
                }
                for run in runs
            ],
            "vector_chunk_rows": self.vector_repo.count_chunks(),
            "distinct_chunk_ids": self.vector_repo.count_distinct_chunk_ids(),
            "qdrant_points": qdrant_points,
        }

    def prune_structured_vectors(
        self,
        *,
        min_size_bytes: int = 0,
        limit_resources: int = 100000,
        dry_run: bool = True,
    ) -> VectorPruneSummary:
        resources = self.vector_repo.list_resources_with_vector_chunks(
            media_types=self.STRUCTURED_VECTOR_MEDIA_TYPES,
            min_size_bytes=min_size_bytes,
            limit=limit_resources,
        )
        try:
            points_before = self.vector_store.count_points()
        except Exception:
            points_before = 0
        if dry_run or not resources:
            return VectorPruneSummary(
                candidates=len(resources),
                points_removed=0,
                points_before=points_before,
                points_after=points_before,
                runs_marked_pruned=0,
                dry_run=dry_run,
                resources=resources,
            )

        resource_ids = [str(item["resource_id"]) for item in resources]
        removed = self.vector_store.delete_points_for_resource_ids(resource_ids)
        try:
            points_after = self.vector_store.count_points()
        except Exception:
            points_after = max(0, points_before - removed)
        runs_marked = self.vector_repo.mark_runs_pruned(
            resource_ids=resource_ids,
            reason=f"pruned structured media vectors ({len(resource_ids)} resources)",
        )
        return VectorPruneSummary(
            candidates=len(resources),
            points_removed=removed,
            points_before=points_before,
            points_after=points_after,
            runs_marked_pruned=runs_marked,
            dry_run=False,
            resources=resources,
        )

    @staticmethod
    def _to_vector_chunk(
        *,
        run_id: str,
        resource_id: str,
        extraction_run_id: str,
        chunk: ChunkDraft,
        embedding_dim: int,
        vector_backend: str,
        collection_name: str,
    ) -> VectorChunk:
        return VectorChunk(
            id=new_uuid(),
            vector_index_run_id=run_id,
            resource_id=resource_id,
            extraction_run_id=extraction_run_id,
            chunk_id=chunk.chunk_id,
            vector_point_id=chunk.vector_point_id,
            source_type=chunk.source_type,
            source_ref=chunk.source_ref,
            page_index=chunk.page_index,
            start_offset=chunk.start_offset,
            end_offset=chunk.end_offset,
            token_count_est=chunk.token_count_est,
            embedding_dim=embedding_dim,
            vector_backend=vector_backend,
            collection_name=collection_name,
            text_digest_sha256=chunk.text_digest_sha256,
            text_content=chunk.text_content,
            created_at=now_utc_iso(),
        )
