"""Strict JSON conversion and crash-safe atomic file replacement."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path

from torchinstruments.records import Absent


def write_json_atomic(path: Path, value: object) -> None:
    """Serialize ``value`` as strict JSON and atomically replace ``path``.

    The temporary file is flushed and synchronized before replacement so readers never observe
    a partially written live record.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        to_json_compatible(value),
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
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


def to_json_compatible(value: object) -> object:
    """Convert typed records into standard-library JSON-compatible values."""
    if isinstance(value, Absent):
        return {"status": "absent", "reason": value.reason}
    if isinstance(value, datetime):
        normalized = value.astimezone(UTC)
        return normalized.isoformat().replace("+00:00", "Z")
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: to_json_compatible(getattr(value, field.name)) for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): to_json_compatible(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [to_json_compatible(item) for item in value]
    if isinstance(value, (bool, float, int, str)):
        return value
    raise TypeError(f"value is not JSON-compatible: {type(value).__qualname__}")
