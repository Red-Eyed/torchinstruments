"""Deterministic conversion of typed telemetry records into strict JSON."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from datetime import UTC, datetime
from enum import Enum

from torchinstruments.records import Absent


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
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        field_names = value._fields
        return {
            str(field_name): to_json_compatible(getattr(value, field_name))
            for field_name in field_names
        }
    if isinstance(value, Mapping):
        return {str(key): to_json_compatible(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [to_json_compatible(item) for item in value]
    if isinstance(value, (bool, float, int, str)):
        return value
    raise TypeError(f"value is not JSON-compatible: {type(value).__qualname__}")


def json_text(value: object) -> str:
    """Serialize one value as deterministic, indented, strict JSON text."""
    return json.dumps(
        to_json_compatible(value),
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )


def json_size_bytes(value: object) -> int:
    """Return the exact UTF-8 size written by the atomic JSON adapter."""
    return len(f"{json_text(value)}\n".encode())
