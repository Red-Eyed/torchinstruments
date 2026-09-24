"""TensorBoard projection derived exclusively from normalized telemetry records."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from torchinstruments.records import (
    HistogramRecord,
    ModuleRecord,
    RunRecord,
    SampleRecord,
    SampleState,
    TensorRecord,
)
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


@runtime_checkable
class TensorBoardLogger(Protocol):
    """Expose an externally owned TensorBoard experiment writer."""

    @property
    def experiment(self) -> HistogramWriter:
        """Return the writer used by the logger for TensorBoard event output."""
        ...


class TensorBoardSink:
    """Project compact histogram records through a TensorBoard logger.

    Sample identifiers become dashboard steps because TorchInstruments cannot observe a
    universal optimizer-step counter. The supplied logger and writer remain caller-owned and are
    never flushed or finalized by this sink.
    """

    def __init__(
        self, logger: TensorBoardLogger | HistogramWriter, *, prefix: str = "torchinstruments"
    ) -> None:
        """Bind a Lightning-compatible TensorBoard logger without importing Lightning."""
        match logger:
            case TensorBoardLogger():
                self._writer = logger.experiment
            case _:
                self._writer = logger
        self._prefix = prefix.strip().strip("/")
        self._initialized = False

    def initialize(self, run: RunRecord, modules: dict[str, ModuleRecord]) -> None:
        """Initialize histogram projection without retaining duplicate metadata."""
        del run, modules
        self._initialized = True

    def observe(self, sample: SampleRecord) -> None:
        """Write only the newly observed lifecycle stage's histograms."""
        if not self._initialized:
            raise RuntimeError("sink must be initialized before observing samples")
        writer = self._writer
        for tag, histogram in _sample_histograms(sample, prefix=self._prefix):
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
                global_step=sample.sample_id,
                walltime=sample.timestamp.timestamp(),
            )

    def close(self) -> None:
        """Detach histogram projection without finalizing externally owned logger resources."""
        self._initialized = False


def _sample_histograms(
    sample: SampleRecord,
    *,
    prefix: str,
) -> list[tuple[str, HistogramRecord]]:
    """Flatten histograms added by the current forward or backward lifecycle write."""
    records: list[tuple[str, HistogramRecord]] = []
    for module_name in sorted(sample.modules):
        for call in sample.modules[module_name]:
            tensors = (
                call.outputs
                if sample.state is SampleState.FORWARD_COMPLETE
                else call.output_gradients
            )
            records.extend(
                _tensor_histograms(
                    module_name,
                    call.call_index,
                    tensors,
                    prefix=f"{prefix}/{call.context.mode.value}/grad_enabled_{str(call.context.grad_enabled).lower()}",
                )
            )
    return records


def _tensor_histograms(
    module_name: str,
    call_index: int,
    tensors: dict[str, TensorRecord],
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
    """Derive TensorBoard bucket bounds and counts from one compact histogram record."""
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
