from __future__ import annotations

from pathlib import Path


def load_claim_rows_from_markdown(path: Path) -> list[dict[str, str]]:
    """
    Minimal markdown importer:
    - Each bullet line (`- ...`) becomes one narrative claim.
    - Optional cite ID suffix supported as `... @Ab12`.
    """
    rows: list[dict[str, str]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line.startswith("- "):
            continue

        text = line[2:].strip()
        source_cite_id = ""

        parts = text.rsplit("@", 1)
        if len(parts) == 2 and len(parts[1].strip()) == 4:
            text = parts[0].rstrip()
            source_cite_id = parts[1].strip()

        if not text:
            continue

        rows.append(
            {
                "claim_type": "narrative",
                "narrative_text": text,
                "source_cite_id": source_cite_id,
            }
        )

    return rows
