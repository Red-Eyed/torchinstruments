"""Atomic text output for the run reading guide."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def write_text_atomic(path: Path, value: str) -> None:
    """Atomically replace one UTF-8 text file after durable temporary-file flushing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = value if value.endswith("\n") else f"{value}\n"

    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            file.write(payload)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
