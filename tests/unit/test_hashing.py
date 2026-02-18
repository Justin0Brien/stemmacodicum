from stemmacodicum.core.hashing import compute_bytes_digest


def test_compute_bytes_digest_sha256() -> None:
    assert (
        compute_bytes_digest(b"stemma")
        == "8bef299c7aae5f17363b5c466dc0f5a47bcb8a921d80a044bbd5bd198bf1d554"
    )
