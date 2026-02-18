from __future__ import annotations

import os
import shutil
from pathlib import Path


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def make_read_only(path: Path) -> None:
    current_mode = path.stat().st_mode
    # Strip write permissions for user/group/other.
    path.chmod(current_mode & ~0o222)


def safe_copy_atomic(src: Path, dst: Path) -> None:
    ensure_directory(dst.parent)
    temp_path = dst.parent / f".{dst.name}.tmp"
    shutil.copy2(src, temp_path)
    os.replace(temp_path, dst)
