"""Synchronous persistence for one atomically updated live telemetry file."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

from torchinstruments.aggregation import Aggregator, LiveAggregator
from torchinstruments.errors import SinkAlreadyInitializedError
from torchinstruments.records import LiveStatsRecord, ModuleRecord, RunRecord, SampleRecord
from torchinstruments.sinks.index_markdown import render_run_index
from torchinstruments.sinks.json import write_json_atomic, write_text_atomic


class DirectorySink:
    """Persist one strict live JSON record in a collision-safe run directory."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        aggregator_factory: Callable[[], Aggregator] = LiveAggregator,
    ) -> None:
        """Bind output and a fresh-aggregator factory without touching the filesystem."""
        self._output_dir = Path(output_dir)
        self._stats_path = self._output_dir / "stats.json"
        self._aggregator = aggregator_factory()
        self._initialized = False
        self._stats: LiveStatsRecord

    def initialize(self, run: RunRecord, modules: Mapping[str, ModuleRecord]) -> None:
        """Create live output and reject current or legacy telemetry collisions."""
        if self._contains_telemetry():
            raise SinkAlreadyInitializedError(
                f"telemetry directory already contains a run: {self._output_dir}"
            )

        self._stats = self._aggregator.initialize(run, modules)
        write_json_atomic(self._stats_path, self._stats)
        self._write_index()
        self._initialized = True

    def observe(self, sample: SampleRecord) -> None:
        """Fold one transient lifecycle event into the canonical live statistics."""
        if not self._initialized:
            raise RuntimeError("sink must be initialized before writing telemetry")
        self._stats = self._aggregator.observe(sample)
        write_json_atomic(self._stats_path, self._stats)
        self._write_index()

    def close(self) -> None:
        """Mark the synchronous sink closed; no buffered data remains to flush."""
        self._initialized = False

    def _write_index(self) -> None:
        """Atomically replace the bounded guide from current live state."""
        content = render_run_index(self._stats)
        write_text_atomic(self._output_dir / "index.md", content)

    def _contains_telemetry(self) -> bool:
        """Detect both current and pre-live layouts to prevent accidental mixing."""
        legacy_paths = (
            self._output_dir / "run.json",
            self._output_dir / "modules.json",
            self._output_dir / "snapshots",
        )
        return self._stats_path.exists() or any(path.exists() for path in legacy_paths)
