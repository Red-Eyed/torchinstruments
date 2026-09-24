"""Regression checks for capture boundaries independent of final report formatting."""

from dataclasses import dataclass, field
from pathlib import Path

import polars as pl
import pytest
import torch
from torch import nn

from torchinstruments import AlwaysSampler, inject_observer, remove_observer
from torchinstruments.records import ModuleRecord, RunRecord, SampleRecord
from torchinstruments.reducers import ReducedScalar
from torchinstruments.sampling import SamplingEvent


@dataclass
class RecordedEvents:
    """Retain test-only compact events to assert callback lifecycle contracts."""

    samples: list[SampleRecord] = field(default_factory=list)

    def initialize(self, run: RunRecord, modules: dict[str, ModuleRecord]) -> None:
        """Accept the selected module catalog without taking ownership of it."""
        del run, modules

    def observe(self, sample: SampleRecord) -> None:
        """Record a transient event for a bounded test run."""
        self.samples.append(sample)

    def close(self) -> None:
        """Keep recorded events available after observer removal."""


@pytest.fixture
def events() -> RecordedEvents:
    """Provide an independent event recorder for each lifecycle test."""
    return RecordedEvents()


class Nested(nn.Module):
    """Return a used and an unused branch in a nested output structure."""

    def forward(self, inputs: torch.Tensor) -> dict[str, list[torch.Tensor]]:
        """Create distinct differentiable outputs with stable tensor paths."""
        return {"branches": [inputs * 2, inputs * 3]}


def test_nested_unused_branch_keeps_missing_gradient_distinct(
    telemetry_dir: Path,
    events: RecordedEvents,
) -> None:
    """Emit the used gradient once without manufacturing a zero for an unused output."""
    model = Nested()
    inject_observer(
        model, output_dir=telemetry_dir, sampler=AlwaysSampler(), sink=events, error_policy="raise"
    )
    outputs = model(torch.ones(2, requires_grad=True))
    outputs["branches"][0].sum().backward()
    remove_observer(model)
    forward, backward = events.samples
    assert set(forward.modules[""][0].outputs) == {"output.branches.0", "output.branches.1"}
    assert set(backward.modules[""][0].output_gradients) == {"grad_output.branches.0"}
    history = pl.read_parquet(telemetry_dir / "history.parquet")
    assert history.height == 27


class Never:
    """Disable samples without changing the underlying execution."""

    def should_sample(self, event: SamplingEvent) -> bool:
        """Reject each forward boundary."""
        del event
        return False


def test_unsampled_execution_does_no_reduction(telemetry_dir: Path) -> None:
    """Preserve the cheap inactive path even with all artifact destinations enabled."""
    model = nn.Identity()

    def forbidden(tensor: torch.Tensor) -> dict[str, ReducedScalar]:
        """Fail if a reducer runs on an unsampled forward."""
        raise AssertionError("unexpected reduction")

    inject_observer(
        model, output_dir=telemetry_dir, sampler=Never(), reducers=[forbidden], error_policy="raise"
    )
    for _ in range(5):
        model(torch.ones(2))
    remove_observer(model)
    assert pl.read_parquet(telemetry_dir / "history.parquet").is_empty()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_dtype_and_histogram_reduction(telemetry_dir: Path, dtype: torch.dtype) -> None:
    """Retain original dtype while safely reducing low-precision tensors."""
    model = nn.Identity()
    inject_observer(model, output_dir=telemetry_dir, sampler=AlwaysSampler(), error_policy="raise")
    model(torch.ones(4, dtype=dtype))
    remove_observer(model)
    history = pl.read_parquet(telemetry_dir / "history.parquet")
    assert history["dtype"].unique().to_list() == [str(dtype).removeprefix("torch.")]
    assert history.filter(pl.col("metric") == "mean")["value"].item() == 1


@pytest.mark.parametrize("policy", ["warn", "ignore", "raise"])
def test_reduction_failure_remains_visible(telemetry_dir: Path, policy: str) -> None:
    """Respect the error policy and preserve failed capture in the reading guide."""
    model = nn.Identity()

    def broken(tensor: torch.Tensor) -> dict[str, ReducedScalar]:
        """Introduce a stable instrumentation failure."""
        raise RuntimeError("broken reducer")

    inject_observer(
        model,
        output_dir=telemetry_dir,
        sampler=AlwaysSampler(),
        reducers=[broken],
        error_policy=policy,
    )
    match policy:
        case "raise":
            with pytest.raises(RuntimeError, match="broken reducer"):
                model(torch.ones(2))
        case "warn":
            with pytest.warns(RuntimeWarning, match="broken reducer"):
                model(torch.ones(2))
        case _:
            model(torch.ones(2))
    remove_observer(model)
    if policy != "raise":
        assert "broken reducer" in (telemetry_dir / "index.md").read_text()
