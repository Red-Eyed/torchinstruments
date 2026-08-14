"""Structural persistence contract for run metadata and snapshots."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from torchinstruments.records import ModuleRecord, RunRecord, SnapshotRecord


class Sink(Protocol):
    """Persist normalized records without coupling the observer to storage."""

    def initialize(self, run: RunRecord, modules: Mapping[str, ModuleRecord]) -> None:
        """Initialize one run and persist its immutable module catalog."""
        ...

    def write_snapshot(self, snapshot: SnapshotRecord) -> None:
        """Persist or atomically enrich one identified snapshot."""
        ...

    def close(self) -> None:
        """Flush pending records and release sink-owned resources."""
        ...
