"""Synchronous persistence for bounded JSON and Markdown research reports."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from torchinstruments.aggregation import Aggregator, LiveAggregator
from torchinstruments.distributed import RankInfo
from torchinstruments.errors import SinkAlreadyInitializedError
from torchinstruments.records import LiveStatsRecord, ModuleRecord, RunRecord, SampleRecord
from torchinstruments.reporting.builder import build_report
from torchinstruments.reporting.records import ReportConfig, ReportRecord
from torchinstruments.reporting.rules import FindingRule
from torchinstruments.sinks.index_markdown import render_run_index
from torchinstruments.sinks.json import write_json_atomic, write_text_atomic

_DEFAULT_REPORT_CONFIG = ReportConfig()
_LOCAL_RANK = RankInfo(rank=0, world_size=1)


class DirectorySink:
    """Persist bounded reports without writing exhaustive telemetry by default."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        aggregator_factory: Callable[[], Aggregator] = LiveAggregator,
        report_config: ReportConfig = _DEFAULT_REPORT_CONFIG,
        finding_rules: Sequence[FindingRule] = (),
        rank: RankInfo = _LOCAL_RANK,
        isolate_rank: bool = False,
        write_full_details: bool = False,
    ) -> None:
        """Bind explicit report policy and optional expensive full-detail diagnostics."""
        root = Path(output_dir)
        self._output_dir = root / f"rank-{rank.rank:03d}" if isolate_rank else root
        self._report_path = self._output_dir / "report.json"
        self._details_path = self._output_dir / "details.json"
        self._aggregator = aggregator_factory()
        self._report_config = report_config
        self._finding_rules = tuple(finding_rules)
        self._rank = rank
        self._write_full_details = write_full_details
        self._initialized = False
        self._stats: LiveStatsRecord
        self._report: ReportRecord

    def initialize(self, run: RunRecord, modules: Mapping[str, ModuleRecord]) -> None:
        """Create initial bounded output and reject current or legacy run collisions."""
        if self._contains_telemetry():
            raise SinkAlreadyInitializedError(
                f"telemetry directory already contains a run: {self._output_dir}"
            )

        self._stats = self._aggregator.initialize(run, modules)
        self._write_report()
        self._initialized = True

    def observe(self, sample: SampleRecord) -> None:
        """Fold one transient event into live state and replace only bounded artifacts."""
        if not self._initialized:
            raise RuntimeError("sink must be initialized before writing telemetry")
        self._stats = self._aggregator.observe(sample)
        self._write_report()

    def close(self) -> None:
        """Mark the synchronous sink closed; no buffered data remains to flush."""
        self._initialized = False

    def _write_report(self) -> None:
        """Build and atomically replace typed JSON plus its compact human projection."""
        self._report = build_report(
            self._stats,
            rank=self._rank,
            config=self._report_config,
            rules=self._finding_rules,
        )
        write_json_atomic(self._report_path, self._report)
        content = render_run_index(self._report)
        write_text_atomic(self._output_dir / "index.md", content)
        if self._write_full_details:
            write_json_atomic(self._details_path, self._stats)

    def _contains_telemetry(self) -> bool:
        """Detect both current and pre-live layouts to prevent accidental mixing."""
        telemetry_paths = (
            self._report_path,
            self._details_path,
            self._output_dir / "index.md",
            self._output_dir / "stats.json",
            self._output_dir / "run.json",
            self._output_dir / "modules.json",
            self._output_dir / "snapshots",
        )
        return any(path.exists() for path in telemetry_paths)
