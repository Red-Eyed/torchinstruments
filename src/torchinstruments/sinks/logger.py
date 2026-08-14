"""Project transient sampled-forward events onto flat scalar metric loggers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from torchinstruments.records import (
    ModuleRecord,
    RunRecord,
    SampleRecord,
    SampleState,
    TensorRecord,
)
from torchinstruments.sinks.paths import path_segment, tensor_path_prefix


class MetricLogger(Protocol):
    """Accept flat scalar metrics at an explicitly identified telemetry step."""

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        """Record named scalar values at ``step`` without taking ownership of the caller."""
        ...


class MetricLoggerSink:
    """Send sampled statistics to a Lightning-compatible metric logger.

    Sample IDs are used as logger steps because TorchInstruments cannot observe a universal
    optimizer-step counter. The supplied logger remains caller-owned and is never saved, flushed,
    or finalized by this sink.
    """

    def __init__(self, logger: MetricLogger, *, prefix: str = "torchinstruments") -> None:
        """Bind an externally owned logger under a non-empty metric-name prefix."""
        normalized_prefix = prefix.strip().strip("/")
        if not normalized_prefix:
            raise ValueError("metric logger prefix must not be empty")
        self._logger = logger
        self._prefix = normalized_prefix
        self._initialized = False

    def initialize(self, run: RunRecord, modules: Mapping[str, ModuleRecord]) -> None:
        """Mark the sink ready without projecting non-scalar run metadata."""
        del run, modules
        self._initialized = True

    def observe(self, sample: SampleRecord) -> None:
        """Log forward values once and only new gradient values after backward."""
        if not self._initialized:
            raise RuntimeError("sink must be initialized before observing samples")
        metrics = _flatten_sample(sample, prefix=self._prefix)
        self._logger.log_metrics(metrics, step=sample.sample_id)

    def close(self) -> None:
        """Detach this sink without finalizing its externally owned logger."""
        self._initialized = False


def _flatten_sample(sample: SampleRecord, *, prefix: str) -> dict[str, float]:
    """Flatten the lifecycle stage newly available in one sample event."""
    if sample.state is SampleState.FORWARD_COMPLETE:
        metric_records = _forward_records(sample)
        duration_name = "forward_collection_duration_ms"
    else:
        metric_records = _backward_records(sample)
        duration_name = "total_collection_duration_ms"

    metrics = {f"{prefix}/observer/{duration_name}": sample.collection_duration_ms}
    for metric_name, value in metric_records:
        metrics[f"{prefix}/{metric_name}"] = value
    return metrics


def _forward_records(sample: SampleRecord) -> list[tuple[str, float]]:
    """Flatten selected-module output statistics in deterministic order."""
    records: list[tuple[str, float]] = []
    for module_name in sorted(sample.modules):
        for call in sample.modules[module_name]:
            records.extend(_tensor_metrics(module_name, call.call_index, call.outputs))
    return records


def _backward_records(sample: SampleRecord) -> list[tuple[str, float]]:
    """Flatten only output-gradient statistics added by the backward rewrite."""
    records: list[tuple[str, float]] = []
    for module_name in sorted(sample.modules):
        for call in sample.modules[module_name]:
            records.extend(_tensor_metrics(module_name, call.call_index, call.output_gradients))
    return records


def _tensor_metrics(
    module_name: str,
    call_index: int,
    tensors: Mapping[str, TensorRecord],
) -> list[tuple[str, float]]:
    """Build stable metric paths for one selected module invocation."""
    records: list[tuple[str, float]] = []
    for tensor_path in sorted(tensors):
        tensor = tensors[tensor_path]
        base = tensor_path_prefix(module_name, call_index, tensor_path)
        for statistic_name in sorted(tensor.stats):
            metric_name = f"{base}/{path_segment(statistic_name)}"
            records.append((metric_name, tensor.stats[statistic_name]))
    return records
