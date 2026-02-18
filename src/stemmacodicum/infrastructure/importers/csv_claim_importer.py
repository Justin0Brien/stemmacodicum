from __future__ import annotations

import csv
from pathlib import Path


def load_claim_rows_from_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, str]] = []
        for row in reader:
            cleaned = {str(k).strip(): (v.strip() if isinstance(v, str) else "") for k, v in row.items()}
            rows.append(cleaned)
        return rows
