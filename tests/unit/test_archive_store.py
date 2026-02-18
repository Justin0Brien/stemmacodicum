from pathlib import Path

from stemmacodicum.infrastructure.archive.store import ArchiveStore


def test_archive_relpath_uses_sha256_sharding() -> None:
    store = ArchiveStore(Path("/tmp/archive"))
    digest = "a" * 64
    rel = store.archive_relpath_for_digest(digest, ".pdf")
    assert str(rel) == "sha256/aa/aa/" + ("a" * 64) + ".pdf"
