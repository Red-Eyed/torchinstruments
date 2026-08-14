"""Tests for scalar logger projection and multi-sink fan-out."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from torch import nn

from tests.json_records import read_snapshot
from torchinstruments import (
    AlwaysSampler,
    CompositeSink,
    DirectorySink,
    MetricLoggerSink,
    inject_observer,
    remove_observer,
)
from torchinstruments.records import ModuleRecord, RunRecord, SnapshotRecord


@dataclass(frozen=True)
class _LogEvent:
    """Capture one flat metric-logger call with its required snapshot step."""

    metrics: Mapping[str, float]
    step: int


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


class _FailingWriteSink:
    """Fail snapshot writes while honoring the rest of the sink lifecycle."""

    def initialize(self, run: RunRecord, modules: Mapping[str, ModuleRecord]) -> None:
        """Accept initialization metadata without retaining it."""
        del run, modules

    def write_snapshot(self, snapshot: SnapshotRecord) -> None:
        """Raise a stable delivery failure for fan-out isolation coverage."""
        del snapshot
        raise RuntimeError("logger destination unavailable")

    def close(self) -> None:
        """Release no resources."""


def test_metric_logger_sink_projects_forward_and_backward_once() -> None:
    """Use one snapshot step while keeping forward and gradient tags distinct."""
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


def test_composite_sink_preserves_json_and_logger_outputs(telemetry_dir: Path) -> None:
    """Fan out one observer lifecycle to lossless JSON and flat scalar metrics."""
    logger = _RecordingLogger()
    model = nn.Linear(4, 1)
    sink = CompositeSink(
        DirectorySink(telemetry_dir),
        MetricLoggerSink(logger, prefix="research/telemetry"),
    )
    inject_observer(model, sampler=AlwaysSampler(), sink=sink, error_policy="raise")

    model(torch.ones(2, 4)).sum().backward()
    remove_observer(model)

    snapshot = read_snapshot(telemetry_dir / "snapshots" / "000000.json")
    assert snapshot["state"] == "backward_observed"
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

    snapshot = read_snapshot(telemetry_dir / "snapshots" / "000000.json")
    assert snapshot["state"] == "forward_complete"


def test_metric_logger_sink_requires_a_non_empty_prefix() -> None:
    """Reject metric paths without an identifying namespace."""
    with pytest.raises(ValueError, match="prefix"):
        MetricLoggerSink(_RecordingLogger(), prefix=" / ")
