from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stemmacodicum.infrastructure.vector.server_runtime import ensure_qdrant_server

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class VectorPoint:
    point_id: str
    vector: list[float]
    payload: dict[str, Any]


class QdrantLocalStore:
    def __init__(
        self,
        *,
        storage_path: Path,
        collection_name: str | None = None,
        url: str | None = None,
        api_key: str | None = None,
        prefer_grpc: bool | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        self.storage_path = storage_path
        self.server_url = self._resolve_server_url(url)
        self.collection_name = (
            collection_name
            or os.getenv("STEMMA_QDRANT_COLLECTION")
            or "stemma_chunks"
        )
        self.api_key = api_key or os.getenv("STEMMA_QDRANT_API_KEY")
        self.prefer_grpc = (
            prefer_grpc
            if prefer_grpc is not None
            else self._read_bool_env("STEMMA_QDRANT_PREFER_GRPC", False)
        )
        self.timeout_seconds = (
            timeout_seconds
            if timeout_seconds is not None
            else self._read_float_env("STEMMA_QDRANT_TIMEOUT_SECONDS", 10.0)
        )
        self.backend_name = "qdrant-server" if self.server_url else "qdrant-local"
        self._client = None
        self._models = None

    def ensure_collection(self, vector_size: int) -> None:
        if vector_size <= 0:
            raise ValueError("vector_size must be positive")
        client, models = self._client_and_models()
        if not self.server_url:
            self.storage_path.mkdir(parents=True, exist_ok=True)

        exists = False
        try:
            exists = bool(client.collection_exists(collection_name=self.collection_name))
        except Exception:
            exists = False

        if not exists:
            client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
            )
            return

        info = client.get_collection(collection_name=self.collection_name)
        configured_size = getattr(getattr(info, "config", None), "params", None)
        vectors_conf = getattr(configured_size, "vectors", None)
        configured_dim = getattr(vectors_conf, "size", None)
        if configured_dim is None:
            return
        if int(configured_dim) != int(vector_size):
            raise RuntimeError(
                f"Qdrant collection '{self.collection_name}' has vector size {configured_dim}, "
                f"but embedder produced {vector_size}."
            )

    def ensure_ready(self) -> None:
        self._client_and_models()

    def upsert_points(self, points: list[VectorPoint]) -> None:
        if not points:
            return
        client, models = self._client_and_models()
        client.upsert(
            collection_name=self.collection_name,
            wait=True,
            points=[
                models.PointStruct(id=point.point_id, vector=point.vector, payload=point.payload)
                for point in points
            ],
        )

    def search(
        self,
        *,
        query_vector: list[float],
        limit: int,
        resource_id: str | None = None,
        extraction_run_id: str | None = None,
        embedding_model: str | None = None,
        chunking_version: str | None = None,
    ) -> list[dict[str, Any]]:
        client, models = self._client_and_models()
        clauses = []
        if resource_id:
            clauses.append(
                models.FieldCondition(key="resource_id", match=models.MatchValue(value=resource_id))
            )
        if extraction_run_id:
            clauses.append(
                models.FieldCondition(
                    key="extraction_run_id",
                    match=models.MatchValue(value=extraction_run_id),
                )
            )
        if embedding_model:
            clauses.append(
                models.FieldCondition(
                    key="embedding_model",
                    match=models.MatchValue(value=embedding_model),
                )
            )
        if chunking_version:
            clauses.append(
                models.FieldCondition(
                    key="chunking_version",
                    match=models.MatchValue(value=chunking_version),
                )
            )
        query_filter = models.Filter(must=clauses) if clauses else None
        if hasattr(client, "query_points"):
            response = client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=query_filter,
                with_payload=True,
                with_vectors=False,
                limit=max(1, limit),
            )
            hits = list(getattr(response, "points", []) or [])
        else:
            hits = client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=query_filter,
                with_payload=True,
                with_vectors=False,
                limit=max(1, limit),
            )
        out: list[dict[str, Any]] = []
        for hit in hits:
            out.append(
                {
                    "id": str(getattr(hit, "id", "")),
                    "score": float(getattr(hit, "score", 0.0)),
                    "payload": dict(getattr(hit, "payload", {}) or {}),
                }
            )
        return out

    def count_points(self) -> int:
        client, _ = self._client_and_models()
        try:
            result = client.count(collection_name=self.collection_name, exact=True)
        except Exception:
            return 0
        return int(getattr(result, "count", 0))

    def _client_and_models(self):
        if self._client is not None and self._models is not None:
            return self._client, self._models

        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "Qdrant dependency is missing. Install with `pip install -e '.[vector]'`."
            ) from exc

        if self.server_url:
            try:
                ensure_qdrant_server(
                    server_url=self.server_url,
                    default_storage_dir=self.storage_path.parent / "qdrant-server-data",
                )
                self._client = QdrantClient(
                    url=self.server_url,
                    api_key=self.api_key,
                    prefer_grpc=self.prefer_grpc,
                    timeout=self.timeout_seconds,
                )
            except Exception as exc:
                if self._read_bool_env("STEMMA_QDRANT_LOCAL_FALLBACK", True):
                    logger.warning(
                        "Qdrant server startup failed (%s). Falling back to local embedded store at %s.",
                        exc,
                        self.storage_path,
                    )
                    base_path = self.storage_path
                    client, used_path = self._open_local_client(QdrantClient, base_path)
                    self._client = client
                    self.storage_path = used_path
                    self.backend_name = (
                        "qdrant-local-fallback-isolated"
                        if used_path != base_path
                        else "qdrant-local-fallback"
                    )
                    self.server_url = None
                else:
                    raise RuntimeError(
                        f"Qdrant server initialization failed: {exc}"
                    ) from exc
        else:
            client, used_path = self._open_local_client(QdrantClient, self.storage_path)
            self._client = client
            self.backend_name = "qdrant-local-isolated" if used_path != self.storage_path else "qdrant-local"
            self.storage_path = used_path
        self._models = models
        return self._client, self._models

    def _open_local_client(self, qdrant_client_cls: type, base_path: Path) -> tuple[object, Path]:
        target = base_path.expanduser().resolve()
        target.mkdir(parents=True, exist_ok=True)
        try:
            return qdrant_client_cls(path=str(target)), target
        except Exception as exc:
            if not self._is_storage_lock_error(exc):
                raise
            if not self._read_bool_env("STEMMA_QDRANT_LOCAL_ISOLATED_FALLBACK", True):
                raise
            isolated = (target.parent / "qdrant-isolated" / f"pid-{os.getpid()}").resolve()
            isolated.mkdir(parents=True, exist_ok=True)
            logger.warning(
                "Primary local Qdrant storage %s is locked by another process; using isolated fallback %s.",
                target,
                isolated,
            )
            return qdrant_client_cls(path=str(isolated)), isolated

    @staticmethod
    def _is_storage_lock_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        return "already accessed by another instance of qdrant client" in msg

    def _resolve_server_url(self, explicit_url: str | None) -> str | None:
        if explicit_url and explicit_url.strip():
            return explicit_url.strip()

        env_url = os.getenv("STEMMA_QDRANT_URL")
        if env_url and env_url.strip():
            return env_url.strip()

        config_path = self.storage_path.parent / "qdrant_url.txt"
        if config_path.exists():
            value = config_path.read_text(encoding="utf-8").strip()
            if value:
                return value
        return None

    @staticmethod
    def _read_bool_env(name: str, default: bool) -> bool:
        raw = os.getenv(name)
        if raw is None:
            return default
        normalized = raw.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        return default

    @staticmethod
    def _read_float_env(name: str, default: float) -> float:
        raw = os.getenv(name)
        if raw is None:
            return default
        try:
            value = float(raw)
        except ValueError:
            return default
        return value if value > 0 else default
