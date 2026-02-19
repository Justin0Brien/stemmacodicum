from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(slots=True)
class EmbeddingConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "auto"
    batch_size: int = 128


class SentenceTransformerEmbedder:
    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        self.config = config or EmbeddingConfig()
        self._model = None
        self._embedding_dim: int | None = None

    @property
    def model_name(self) -> str:
        return self.config.model_name

    def embedding_dim(self) -> int:
        if self._embedding_dim is None:
            self._load_model()
            if self._embedding_dim is None:
                raise RuntimeError("Unable to determine embedding dimension.")
        return self._embedding_dim

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        self._load_model()
        vectors = self._model.encode(
            texts,
            batch_size=max(1, self.config.batch_size),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        if hasattr(vectors, "tolist"):
            out = vectors.tolist()
        else:
            out = [list(v) for v in vectors]
        if out and self._embedding_dim is None:
            self._embedding_dim = len(out[0])
        return [[float(x) for x in row] for row in out]

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "Vector embedding dependencies are missing. Install with "
                "`pip install -e '.[vector]'`."
            ) from exc

        # Keep CPU thread counts bounded when running on large machines.
        if "OMP_NUM_THREADS" not in os.environ:
            os.environ["OMP_NUM_THREADS"] = "8"

        device = self._resolve_device(torch)
        self._model = SentenceTransformer(self.config.model_name, device=device)
        dim = self._model.get_sentence_embedding_dimension()
        self._embedding_dim = int(dim) if dim else None

    def _resolve_device(self, torch_module) -> str:
        configured = (self.config.device or "auto").strip().lower()
        if configured and configured != "auto":
            return configured

        if bool(getattr(torch_module.backends, "mps", None)) and torch_module.backends.mps.is_available():
            return "mps"
        if torch_module.cuda.is_available():
            return "cuda"
        return "cpu"
