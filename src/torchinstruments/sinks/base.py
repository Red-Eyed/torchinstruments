"""Structural contract for live telemetry destinations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from torchinstruments.records import ModuleRecord, RunRecord, SampleRecord


class Sink(Protocol):
    """Persist normalized records without coupling the observer to storage."""

    def initialize(self, run: RunRecord, modules: Mapping[str, ModuleRecord]) -> None:
        """Initialize one run and persist its immutable module catalog."""
        ...

    def observe(self, sample: SampleRecord) -> None:
        """Consume one transient forward or backward sample event."""
        ...

    def close(self) -> None:
        """Flush pending records and release sink-owned resources."""
        ...
