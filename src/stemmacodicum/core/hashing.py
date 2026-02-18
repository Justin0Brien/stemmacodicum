from __future__ import annotations

import hashlib
from pathlib import Path


def compute_bytes_digest(data: bytes, alg: str = "sha256") -> str:
    h = hashlib.new(alg)
    h.update(data)
    return h.hexdigest()


def compute_file_digest(path: Path, alg: str = "sha256", chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.new(alg)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()
