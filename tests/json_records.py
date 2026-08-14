"""Typed boundary parsers for JSON artifacts exercised by tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict, cast


class JsonTensorRecord(TypedDict):
    """Describe the tensor-record fields consumed by tests."""

    shape: list[int]
    dtype: str
    device: str
    numel: int
    stats: dict[str, float]
    unavailable_stats: dict[str, str]


class JsonModuleCall(TypedDict):
    """Describe one serialized selected-module invocation."""

    call_index: int
    outputs: dict[str, JsonTensorRecord]
    output_gradients: dict[str, JsonTensorRecord]


class JsonErrorRecord(TypedDict):
    """Describe serialized collection-error fields consumed by tests."""

    exception_type: str
    message: str


class JsonSnapshotRecord(TypedDict):
    """Describe snapshot fields validated across behavior tests."""

    schema_version: int
    snapshot_id: int
    state: str
    collection_duration_ms: float
    modules: dict[str, list[JsonModuleCall]]
    errors: list[JsonErrorRecord]


class JsonModuleRecord(TypedDict):
    """Describe module-catalog fields consumed by alias tests."""

    aliases: list[str]


class JsonSamplingRecord(TypedDict):
    """Describe serialized sampling-policy metadata."""

    type: str
    settings: dict[str, bool | float | int | str]


class JsonRunRecord(TypedDict):
    """Describe immutable run metadata consumed by schema tests."""

    schema_version: int
    created_at: str
    observer_version: str
    torch_version: str
    sampling: JsonSamplingRecord


def read_snapshot(path: Path) -> JsonSnapshotRecord:
    """Parse a snapshot after validating its required top-level fields."""
    value = _read_object(path)
    required = {"schema_version", "snapshot_id", "state", "modules", "errors"}
    if not required.issubset(value):
        raise ValueError(f"snapshot is missing fields: {sorted(required - value.keys())}")
    return cast(JsonSnapshotRecord, value)


def read_modules(path: Path) -> dict[str, JsonModuleRecord]:
    """Parse the module catalog into its test-specific typed shape."""
    return cast(dict[str, JsonModuleRecord], _read_object(path))


def read_run(path: Path) -> JsonRunRecord:
    """Parse run metadata after validating its required top-level fields."""
    value = _read_object(path)
    required = {"schema_version", "created_at", "observer_version", "torch_version", "sampling"}
    if not required.issubset(value):
        raise ValueError(f"run record is missing fields: {sorted(required - value.keys())}")
    return cast(JsonRunRecord, value)


def _read_object(path: Path) -> dict[str, object]:
    """Read one JSON object and reject non-object roots at the test boundary."""
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value
