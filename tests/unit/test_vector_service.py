from __future__ import annotations

from pathlib import Path

from stemmacodicum.application.services.extraction_service import ExtractionService
from stemmacodicum.application.services.ingestion_service import IngestionService
from stemmacodicum.application.services.vector_service import VectorIndexingService
from stemmacodicum.infrastructure.archive.store import ArchiveStore
from stemmacodicum.infrastructure.db.repos.extraction_repo import ExtractionRepo
from stemmacodicum.infrastructure.db.repos.resource_repo import ResourceRepo
from stemmacodicum.infrastructure.db.repos.vector_repo import VectorRepo
from stemmacodicum.infrastructure.db.sqlite import initialize_schema
from stemmacodicum.infrastructure.vector.chunking import VectorChunker


class _FakeEmbedder:
    def __init__(self) -> None:
        self.model_name = "fake-model"
        self.config = type("Cfg", (), {"batch_size": 32})()

    def embedding_dim(self) -> int:
        return 3

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            length = float(len(text))
            vowels = float(sum(1 for c in text.lower() if c in {"a", "e", "i", "o", "u"}))
            out.append([length, vowels, 1.0])
        return out


class _FakeStore:
    backend_name = "fake-store"

    def __init__(self) -> None:
        self.collection_name = "stemma_chunks"
        self._points: list[object] = []
        self._dim: int | None = None

    def ensure_collection(self, vector_size: int) -> None:
        self._dim = vector_size

    def upsert_points(self, points: list[object]) -> None:
        self._points.extend(points)

    def search(
        self,
        *,
        query_vector: list[float],
        limit: int,
        resource_id: str | None = None,
        extraction_run_id: str | None = None,
        embedding_model: str | None = None,
        chunking_version: str | None = None,
    ) -> list[dict[str, object]]:
        scored: list[dict[str, object]] = []
        for point in self._points:
            payload = point.payload
            if resource_id and payload.get("resource_id") != resource_id:
                continue
            if extraction_run_id and payload.get("extraction_run_id") != extraction_run_id:
                continue
            if embedding_model and payload.get("embedding_model") != embedding_model:
                continue
            if chunking_version and payload.get("chunking_version") != chunking_version:
                continue
            score = sum(float(a) * float(b) for a, b in zip(query_vector, point.vector, strict=True))
            scored.append({"id": point.point_id, "score": score, "payload": payload})
        scored.sort(key=lambda row: row["score"], reverse=True)
        return scored[:limit]

    def count_points(self) -> int:
        return len(self._points)


def _bootstrap(tmp_path: Path) -> tuple[ResourceRepo, ExtractionRepo, VectorRepo, Path]:
    db_path = tmp_path / "stemma.db"
    schema_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "stemmacodicum"
        / "infrastructure"
        / "db"
        / "schema.sql"
    )
    initialize_schema(db_path, schema_path)
    return ResourceRepo(db_path), ExtractionRepo(db_path), VectorRepo(db_path), tmp_path / "archive"


def test_vector_index_service_indexes_and_searches(tmp_path: Path) -> None:
    resource_repo, extraction_repo, vector_repo, archive_dir = _bootstrap(tmp_path)

    source = tmp_path / "report.md"
    source.write_text(
        """
| Item | Value |
|---|---:|
| Cash | 5631 |
Liquidity improved in FY2025.
""",
        encoding="utf-8",
    )

    ingest_service = IngestionService(resource_repo=resource_repo, archive_store=ArchiveStore(archive_dir))
    resource = ingest_service.ingest_file(source).resource
    extract_service = ExtractionService(
        resource_repo=resource_repo,
        extraction_repo=extraction_repo,
        archive_dir=archive_dir,
    )
    extract_summary = extract_service.extract_resource(resource.id)

    service = VectorIndexingService(
        resource_repo=resource_repo,
        extraction_repo=extraction_repo,
        vector_repo=vector_repo,
        vector_store=_FakeStore(),
        embedder=_FakeEmbedder(),
        chunker=VectorChunker(),
    )

    first = service.index_extraction(resource_id=resource.id, extraction_run_id=extract_summary.run_id)
    assert first.status == "success"
    assert first.chunks_total > 0
    assert vector_repo.count_chunks() > 0

    second = service.index_extraction(resource_id=resource.id, extraction_run_id=extract_summary.run_id)
    assert second.status == "skipped"

    hits = service.search(query="cash at bank", limit=5, resource_id=resource.id)
    assert len(hits) >= 1
