"""Strict JSON conversion and crash-safe atomic file replacement."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from torchinstruments.serialization import json_text


def write_json_atomic(path: Path, value: object) -> None:
    """Serialize ``value`` as strict JSON and atomically replace ``path``.

    The temporary file is flushed and synchronized before replacement so readers never observe
    a partially written live record.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json_text(value)
    write_text_atomic(path, payload)


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
