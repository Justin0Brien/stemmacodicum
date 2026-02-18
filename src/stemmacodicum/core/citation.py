from __future__ import annotations

import hashlib
import string

BASE62_ALPHABET = string.ascii_uppercase + string.ascii_lowercase + string.digits


def normalize_cite_key(cite_key: str) -> str:
    return cite_key.strip().lower()


def _int_to_base62(value: int, width: int) -> str:
    chars: list[str] = []
    base = len(BASE62_ALPHABET)
    for _ in range(width):
        value, idx = divmod(value, base)
        chars.append(BASE62_ALPHABET[idx])
    return "".join(reversed(chars))


def generate_cite_id(normalized_key: str, attempt: int = 0) -> str:
    seed = f"{normalized_key}:{attempt}".encode("utf-8")
    digest = hashlib.sha256(seed).digest()
    value = int.from_bytes(digest[:10], "big")
    return _int_to_base62(value, 4)
