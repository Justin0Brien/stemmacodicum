from __future__ import annotations

import json
from pathlib import Path


class ClaimJsonError(ValueError):
    pass


def load_claim_rows_from_json(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(payload, list):
        return [_ensure_dict(item) for item in payload]

    if isinstance(payload, dict):
        claims = payload.get("claims")
        if isinstance(claims, list):
            return [_ensure_dict(item) for item in claims]

    raise ClaimJsonError("JSON must be a list of claim objects or an object containing 'claims'.")


def _ensure_dict(item: object) -> dict[str, object]:
    if not isinstance(item, dict):
        raise ClaimJsonError("Each claim entry must be a JSON object.")
    return {str(k): v for k, v in item.items()}
