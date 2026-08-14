"""TensorBoard projection derived exclusively from normalized telemetry records."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Protocol

from torchinstruments.records import (
    HistogramRecord,
    ModuleRecord,
    RunRecord,
    SnapshotRecord,
    SnapshotState,
    TensorRecord,
)
from torchinstruments.sinks.logger import MetricLogger, MetricLoggerSink
from torchinstruments.sinks.paths import path_segment, tensor_path_prefix


class HistogramWriter(Protocol):
    """Accept TensorBoard's compact pre-aggregated histogram representation."""

    def add_histogram_raw(
        self,
        tag: str,
        min: float,
        max: float,
        num: int,
        sum: float,
        sum_squares: float,
        bucket_limits: Sequence[float],
        bucket_counts: Sequence[float],
        global_step: int | None = None,
        walltime: float | None = None,
    ) -> None:
        """Write one histogram without receiving its original tensor values."""
        ...


class TensorBoardLogger(MetricLogger, Protocol):
    """Expose scalar logging and an externally owned TensorBoard experiment writer."""

    @property
    def experiment(self) -> HistogramWriter:
        """Return the writer used by the logger for TensorBoard event output."""
        ...


class TensorBoardSink:
    """Project canonical scalar and histogram records through a TensorBoard logger.

    Snapshot identifiers become dashboard steps because TorchInstruments cannot observe a
    universal optimizer-step counter. The supplied logger and writer remain caller-owned and are
    never flushed or finalized by this sink.
    """

    def __init__(self, logger: TensorBoardLogger, *, prefix: str = "torchinstruments") -> None:
        """Bind a Lightning-compatible TensorBoard logger without importing Lightning."""
        self._logger = logger
        self._metric_sink = MetricLoggerSink(logger, prefix=prefix)
        self._prefix = prefix.strip().strip("/")
        self._initialized = False

    def initialize(self, run: RunRecord, modules: Mapping[str, ModuleRecord]) -> None:
        """Initialize the scalar projection and retain no duplicate run metadata."""
        self._metric_sink.initialize(run, modules)
        self._initialized = True

    def write_snapshot(self, snapshot: SnapshotRecord) -> None:
        """Write scalars and only the newly observed lifecycle stage's histograms."""
        if not self._initialized:
            raise RuntimeError("sink must be initialized before writing snapshots")
        self._metric_sink.write_snapshot(snapshot)
        writer = self._logger.experiment
        for tag, histogram in _snapshot_histograms(snapshot, prefix=self._prefix):
            limits, counts = _tensorboard_buckets(histogram)
            writer.add_histogram_raw(
                tag=tag,
                min=histogram.minimum,
                max=histogram.maximum,
                num=histogram.finite_count,
                sum=histogram.sum,
                sum_squares=histogram.sum_squares,
                bucket_limits=limits,
                bucket_counts=counts,
                global_step=snapshot.snapshot_id,
                walltime=snapshot.timestamp.timestamp(),
            )

    def close(self) -> None:
        """Detach both projections without finalizing externally owned logger resources."""
        self._metric_sink.close()
        self._initialized = False


def _snapshot_histograms(
    snapshot: SnapshotRecord,
    *,
    prefix: str,
) -> list[tuple[str, HistogramRecord]]:
    """Flatten histograms added by the current forward or backward lifecycle write."""
    records: list[tuple[str, HistogramRecord]] = []
    for module_name in sorted(snapshot.modules):
        for call in snapshot.modules[module_name]:
            tensors = (
                call.outputs
                if snapshot.state is SnapshotState.FORWARD_COMPLETE
                else call.output_gradients
            )
            records.extend(
                _tensor_histograms(
                    module_name,
                    call.call_index,
                    tensors,
                    prefix=prefix,
                )
            )
    return records


def _tensor_histograms(
    module_name: str,
    call_index: int,
    tensors: Mapping[str, TensorRecord],
    *,
    prefix: str,
) -> list[tuple[str, HistogramRecord]]:
    """Build stable TensorBoard tags for one module invocation's histogram records."""
    records: list[tuple[str, HistogramRecord]] = []
    for tensor_path in sorted(tensors):
        tensor = tensors[tensor_path]
        base = tensor_path_prefix(module_name, call_index, tensor_path)
        for histogram_name in sorted(tensor.histograms):
            tag = f"{prefix}/{base}/histograms/{path_segment(histogram_name)}"
            records.append((tag, tensor.histograms[histogram_name]))
    return records


def _tensorboard_buckets(histogram: HistogramRecord) -> tuple[list[float], list[float]]:
    """Derive TensorBoard bucket bounds and counts losslessly from one JSON record."""
    lower = histogram.bin_edges[0]
    upper = histogram.bin_edges[-1]
    underflow_limit = math.nextafter(lower, -math.inf)
    overflow_limit = max(histogram.maximum, math.nextafter(upper, math.inf))
    limits = [underflow_limit, *histogram.bin_edges[1:], overflow_limit]
    counts = [
        float(histogram.underflow_count),
        *(float(count) for count in histogram.bin_counts),
        float(histogram.overflow_count),
    ]
    return limits, counts
