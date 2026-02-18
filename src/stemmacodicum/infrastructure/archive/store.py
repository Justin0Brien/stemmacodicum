from __future__ import annotations

from pathlib import Path

from stemmacodicum.core.files import ensure_directory, make_read_only, safe_copy_atomic
from stemmacodicum.core.hashing import compute_file_digest


class ArchiveStore:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir

    def ensure_archive_layout(self) -> None:
        ensure_directory(self.base_dir)
        ensure_directory(self.base_dir / "sha256")

    def archive_relpath_for_digest(self, digest_sha256: str, suffix: str = "") -> Path:
        shard_a = digest_sha256[:2]
        shard_b = digest_sha256[2:4]
        name = f"{digest_sha256}{suffix}"
        return Path("sha256") / shard_a / shard_b / name

    def archive_abspath_for_digest(self, digest_sha256: str, suffix: str = "") -> Path:
        return self.base_dir / self.archive_relpath_for_digest(digest_sha256, suffix)

    def store_file_immutable(self, src: Path, digest_sha256: str, suffix: str = "") -> Path:
        self.ensure_archive_layout()
        dst = self.archive_abspath_for_digest(digest_sha256, suffix)
        ensure_directory(dst.parent)

        if not dst.exists():
            safe_copy_atomic(src, dst)
            make_read_only(dst)

        return dst

    @staticmethod
    def verify_archived_integrity(path: Path, expected_digest: str) -> bool:
        if not path.exists():
            return False
        return compute_file_digest(path, "sha256") == expected_digest
