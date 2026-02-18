from __future__ import annotations

import uuid


def new_uuid() -> str:
    """Generate a new UUID4 as a string."""
    return str(uuid.uuid4())
