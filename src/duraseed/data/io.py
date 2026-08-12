"""Small standard-library helpers for data artifacts."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile


def atomic_write_bytes(path: str | os.PathLike[str], payload: bytes) -> Path:
    """Replace ``path`` atomically with ``payload`` and return the path."""

    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.tmp-",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


__all__ = ["atomic_write_bytes"]
