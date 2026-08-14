"""Synchronous directory-backed telemetry persistence."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from torchinstruments.errors import SinkAlreadyInitializedError
from torchinstruments.records import ModuleRecord, RunRecord, SnapshotRecord
from torchinstruments.sinks.index_markdown import (
    ObservedTelemetry,
    render_run_index,
    summarize_snapshot,
)
from torchinstruments.sinks.json import write_json_atomic, write_text_atomic


class DirectorySink:
    """Persist strict JSON records in one collision-safe run directory."""

    def __init__(self, output_dir: str | Path) -> None:
        """Bind the sink to a directory without touching the filesystem yet."""
        self._output_dir = Path(output_dir)
        self._snapshots_dir = self._output_dir / "snapshots"
        self._initialized = False
        self._snapshot_count = 0
        self._observed = ObservedTelemetry()

    def initialize(self, run: RunRecord, modules: Mapping[str, ModuleRecord]) -> None:
        """Create the run layout and reject directories already containing telemetry."""
        run_path = self._output_dir / "run.json"
        if run_path.exists():
            raise SinkAlreadyInitializedError(
                f"telemetry directory already contains a run: {self._output_dir}"
            )

        self._snapshots_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(run_path, run)
        write_json_atomic(self._output_dir / "modules.json", modules)
        self._run = run
        self._modules = dict(modules)
        self._write_index()
        self._initialized = True

    def write_snapshot(self, snapshot: SnapshotRecord) -> None:
        """Write one snapshot under its zero-padded monotonic identifier."""
        if not self._initialized:
            raise RuntimeError("sink must be initialized before writing snapshots")
        filename = f"{snapshot.snapshot_id:06d}.json"
        write_json_atomic(self._snapshots_dir / filename, snapshot)
        self._snapshot_count = max(self._snapshot_count, snapshot.snapshot_id + 1)
        self._observed = self._observed.merged(summarize_snapshot(snapshot))
        self._write_index()

    def close(self) -> None:
        """Mark the synchronous sink closed; no buffered data remains to flush."""
        self._initialized = False

    def _write_index(self) -> None:
        """Atomically replace the bounded run guide from current normalized state."""
        content = render_run_index(
            self._run,
            self._modules,
            snapshot_count=self._snapshot_count,
            observed=self._observed,
        )
        write_text_atomic(self._output_dir / "index.md", content)
