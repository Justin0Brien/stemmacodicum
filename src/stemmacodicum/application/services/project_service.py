from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from stemmacodicum.core.config import AppPaths
from stemmacodicum.core.files import ensure_directory
from stemmacodicum.infrastructure.db.sqlite import initialize_schema


@dataclass(slots=True)
class InitResult:
    paths_created: list[Path]
    db_path: Path


class ProjectService:
    def __init__(self, paths: AppPaths) -> None:
        self.paths = paths

    def init_project(self) -> InitResult:
        paths_created: list[Path] = []

        for path in (
            self.paths.stemma_dir,
            self.paths.archive_dir,
            self.paths.vector_dir,
            self.paths.qdrant_dir,
        ):
            if not path.exists():
                paths_created.append(path)
            ensure_directory(path)

        schema_path = (
            Path(__file__).resolve().parents[2] / "infrastructure" / "db" / "schema.sql"
        )
        initialize_schema(self.paths.db_path, schema_path)

        return InitResult(paths_created=paths_created, db_path=self.paths.db_path)

    def is_initialized(self) -> bool:
        return self.paths.db_path.exists()
