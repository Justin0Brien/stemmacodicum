import re

from stemmacodicum.core.citation import generate_cite_id, normalize_cite_key


def test_normalize_cite_key() -> None:
    assert normalize_cite_key("  SomeKey  ") == "somekey"


def test_generate_cite_id_is_deterministic_and_4_chars() -> None:
    cite_id_a = generate_cite_id("my-citation-key", 0)
    cite_id_b = generate_cite_id("my-citation-key", 0)

    assert cite_id_a == cite_id_b
    assert len(cite_id_a) == 4
    assert re.fullmatch(r"[A-Za-z0-9]{4}", cite_id_a)


def test_generate_cite_id_changes_with_attempt() -> None:
    assert generate_cite_id("my-citation-key", 0) != generate_cite_id("my-citation-key", 1)
