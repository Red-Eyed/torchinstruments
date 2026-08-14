"""Structural contract for bounded live telemetry aggregation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from torchinstruments.records import LiveStatsRecord, ModuleRecord, RunRecord, SampleRecord


class Aggregator(Protocol):
    """Convert transient sampled-forward events into one bounded live record."""

    def initialize(self, run: RunRecord, modules: Mapping[str, ModuleRecord]) -> LiveStatsRecord:
        """Initialize and return an empty live record for one observer run."""
        ...

    def observe(self, sample: SampleRecord) -> LiveStatsRecord:
        """Update aggregation with one forward or backward lifecycle event."""
        ...
