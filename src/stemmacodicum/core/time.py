from __future__ import annotations

from datetime import datetime, timezone


def now_utc_iso() -> str:
    """Return an RFC 3339/ISO timestamp in UTC with seconds precision."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
