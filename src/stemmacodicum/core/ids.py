from __future__ import annotations

import uuid


def new_uuid() -> str:
    """Generate a new UUID4 as a string."""
    return str(uuid.uuid4())


def deterministic_uuid(name: str, namespace: uuid.UUID = uuid.NAMESPACE_URL) -> str:
    """Generate a deterministic UUID5 from a stable name."""
    return str(uuid.uuid5(namespace, name))
