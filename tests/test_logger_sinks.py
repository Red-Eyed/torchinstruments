"""Tests for scalar logger projection and multi-sink fan-out."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pytest
import torch
from torch import nn

from tests.json_records import read_stats
from torchinstruments import (
    AlwaysSampler,
    CompositeSink,
    DirectorySink,
    MetricLoggerSink,
    TensorBoardSink,
    histogram,
    inject_observer,
    remove_observer,
)
from torchinstruments.records import ModuleRecord, RunRecord, SampleRecord


@dataclass(frozen=True)
class _LogEvent:
    """Capture one flat metric-logger call with its required sample step."""

    metrics: Mapping[str, float]
    step: int


@dataclass(frozen=True)
class _HistogramEvent:
    """Capture one pre-aggregated TensorBoard histogram call."""

    tag: str
    minimum: float
    maximum: float
    count: int
    sum: float
    sum_squares: float
    bucket_limits: tuple[float, ...]
    bucket_counts: tuple[float, ...]
    step: int
    walltime: float


class _RecordingLogger:
    """Record structurally compatible metric calls without an external service."""

    def __init__(self) -> None:
        """Start with no observed metric events."""
        self.events: list[_LogEvent] = []

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        """Preserve one logger call and reject a missing telemetry step."""
        if step is None:
            raise ValueError("test logger requires an explicit step")
        self.events.append(_LogEvent(metrics=dict(metrics), step=step))


class _RecordingHistogramWriter:
    """Record compact histogram calls without importing a dashboard implementation."""

    def __init__(self) -> None:
        """Start with no projected histogram events."""
        self.events: list[_HistogramEvent] = []

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
        """Preserve one raw histogram call and require replay metadata."""
        if global_step is None or walltime is None:
            raise ValueError("test writer requires a step and walltime")
        self.events.append(
            _HistogramEvent(
                tag=tag,
                minimum=min,
                maximum=max,
                count=num,
                sum=sum,
                sum_squares=sum_squares,
                bucket_limits=tuple(bucket_limits),
                bucket_counts=tuple(bucket_counts),
                step=global_step,
                walltime=walltime,
            )
        )


class _RecordingTensorBoardLogger(_RecordingLogger):
    """Expose one recording histogram writer through Lightning's logger shape."""

    def __init__(self) -> None:
        """Initialize scalar and histogram event collections."""
        super().__init__()
        self._experiment = _RecordingHistogramWriter()

    @property
    def experiment(self) -> _RecordingHistogramWriter:
        """Return the externally owned recording writer."""
        return self._experiment


class _FailingWriteSink:
    """Fail sample deliveries while honoring the rest of the sink lifecycle."""

    def initialize(self, run: RunRecord, modules: Mapping[str, ModuleRecord]) -> None:
        """Accept initialization metadata without retaining it."""
        del run, modules

    def observe(self, sample: SampleRecord) -> None:
        """Raise a stable delivery failure for fan-out isolation coverage."""
        del sample
        raise RuntimeError("logger destination unavailable")

    def close(self) -> None:
        """Release no resources."""


def test_metric_logger_sink_projects_forward_and_backward_once() -> None:
    """Use one sample step while keeping forward and gradient tags distinct."""
    logger = _RecordingLogger()
    model = nn.Identity()
    inject_observer(
        model,
        sampler=AlwaysSampler(),
        sink=MetricLoggerSink(logger),
        error_policy="raise",
    )

    output = model(torch.tensor([3.0, 4.0], requires_grad=True))
    output.sum().backward()
    remove_observer(model)

    assert [event.step for event in logger.events] == [0, 0]
    forward, backward = logger.events
    assert forward.metrics["torchinstruments/modules/@root/call_0/output/rms"] == pytest.approx(
        12.5**0.5
    )
    assert "torchinstruments/modules/@root/call_0/grad_output/rms" not in forward.metrics
    assert backward.metrics[
        "torchinstruments/modules/@root/call_0/grad_output/rms"
    ] == pytest.approx(1.0)
    assert "torchinstruments/modules/@root/call_0/output/rms" not in backward.metrics


def test_composite_sink_preserves_live_json_and_logger_outputs(telemetry_dir: Path) -> None:
    """Fan out one observer lifecycle to live JSON and flat scalar metrics."""
    logger = _RecordingLogger()
    model = nn.Linear(4, 1)
    sink = CompositeSink(
        DirectorySink(telemetry_dir),
        MetricLoggerSink(logger, prefix="research/telemetry"),
    )
    inject_observer(model, sampler=AlwaysSampler(), sink=sink, error_policy="raise")

    model(torch.ones(2, 4)).sum().backward()
    remove_observer(model)

    stats = read_stats(telemetry_dir / "stats.json")
    assert stats["backward_samples_observed"] == 1
    assert len(logger.events) == 2
    assert any(name.startswith("research/telemetry/") for name in logger.events[0].metrics)


def test_composite_sink_requires_a_destination() -> None:
    """Reject a fan-out sink that cannot deliver records anywhere."""
    with pytest.raises(ValueError, match="at least one sink"):
        CompositeSink()


def test_composite_sink_attempts_json_after_another_sink_fails(telemetry_dir: Path) -> None:
    """Preserve canonical output when an earlier dashboard destination fails."""
    model = nn.Identity()
    sink = CompositeSink(_FailingWriteSink(), DirectorySink(telemetry_dir))
    inject_observer(model, sampler=AlwaysSampler(), sink=sink, error_policy="ignore")

    model(torch.ones(1))
    remove_observer(model)

    stats = read_stats(telemetry_dir / "stats.json")
    assert stats["samples_observed"] == 1


def test_metric_logger_sink_requires_a_non_empty_prefix() -> None:
    """Reject metric paths without an identifying namespace."""
    with pytest.raises(ValueError, match="prefix"):
        MetricLoggerSink(_RecordingLogger(), prefix=" / ")


def test_tensorboard_histogram_is_derived_from_canonical_json(telemetry_dir: Path) -> None:
    """Project the exact JSON histogram moments and counts without a raw tensor."""
    logger = _RecordingTensorBoardLogger()
    model = nn.Identity()
    sink = CompositeSink(
        DirectorySink(telemetry_dir),
        TensorBoardSink(logger),
    )
    inject_observer(
        model,
        sampler=AlwaysSampler(),
        histograms=[
            histogram(bins=2, value_range=(-1.0, 1.0), every_n_samples=1),
        ],
        sink=sink,
        error_policy="raise",
    )

    model(torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]))
    remove_observer(model)

    stats = read_stats(telemetry_dir / "stats.json")
    output = stats["layers"][""][0]["outputs"]["output"]
    json_record = output["histograms"]["distribution"]["latest"]
    event = logger.experiment.events[0]
    assert event.tag == "torchinstruments/modules/@root/call_0/output/histograms/distribution"
    assert event.minimum == json_record["minimum"]
    assert event.maximum == json_record["maximum"]
    assert event.count == json_record["finite_count"]
    assert event.sum == json_record["sum"]
    assert event.sum_squares == json_record["sum_squares"]
    assert event.bucket_counts == (
        float(json_record["underflow_count"]),
        *(float(count) for count in json_record["bin_counts"]),
        float(json_record["overflow_count"]),
    )
    rms_latest = output["statistics"]["rms"]["latest"]
    assert event.step == rms_latest["sample_id"]
    assert (
        event.walltime
        == datetime.fromisoformat(rms_latest["timestamp"].replace("Z", "+00:00")).timestamp()
    )
