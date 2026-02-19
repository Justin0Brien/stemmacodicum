from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppPaths:
    project_root: Path
    stemma_dir: Path
    db_path: Path
    archive_dir: Path
    vector_dir: Path
    qdrant_dir: Path


DEFAULT_STEMMA_DIRNAME = ".stemma"


def load_paths(project_root: Path | None = None) -> AppPaths:
    root = (project_root or Path.cwd()).expanduser().resolve()

    stemma_home_raw = os.getenv("STEMMA_HOME")
    if stemma_home_raw:
        stemma_dir = Path(stemma_home_raw).expanduser().resolve()
    else:
        stemma_dir = root / DEFAULT_STEMMA_DIRNAME

    return AppPaths(
        project_root=root,
        stemma_dir=stemma_dir,
        db_path=stemma_dir / "stemma.db",
        archive_dir=stemma_dir / "archive",
        vector_dir=stemma_dir / "vector",
        qdrant_dir=stemma_dir / "vector" / "qdrant",
    )
