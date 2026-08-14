"""Tests for live telemetry structure, correlation, dtypes, and error isolation."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from datetime import UTC
from pathlib import Path

import pytest
import torch
from dirty_equals import IsNonNegative, IsNow, IsPartialDict
from torch import nn

from tests.json_records import read_stats, require_histogram
from torchinstruments import (
    AlwaysSampler,
    DirectorySink,
    histogram,
    inject_observer,
    remove_observer,
)
from torchinstruments.reducers import ReducedScalar


class _NestedOutputModel(nn.Module):
    """Return tensors inside mappings, lists, and nullable values."""

    def __init__(self) -> None:
        """Create one computational child used by the nested output."""
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, inputs: torch.Tensor) -> dict[str, object]:
        """Return logits and hidden state through a heterogeneous structure."""
        logits = self.linear(inputs)
        return {"hidden_states": [inputs, None], "logits": logits}


class _SharedModuleModel(nn.Module):
    """Invoke one aliased module object twice in a root forward."""

    def __init__(self) -> None:
        """Expose the same linear module under two registered aliases."""
        super().__init__()
        shared = nn.Linear(4, 4)
        self.first = shared
        self.second = shared

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Call the shared module twice without overwriting call telemetry."""
        return self.second(torch.relu(self.first(inputs)))


class _PartiallyUsedOutputs(nn.Module):
    """Return two differentiable outputs while allowing one to remain unused."""

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Create independent graph outputs for partial-backward coverage."""
        return inputs * 2, inputs * 3


def test_nested_outputs_receive_stable_paths(
    telemetry_dir: Path,
    detailed_sink: DirectorySink,
) -> None:
    """Assign deterministic dotted paths to tensors in nested model outputs."""
    model = _NestedOutputModel()
    inject_observer(
        model,
        sampler=AlwaysSampler(),
        selector=lambda name, module: name == "" and isinstance(module, _NestedOutputModel),
        sink=detailed_sink,
    )

    model(torch.randn(2, 4))

    stats = read_stats(telemetry_dir / "details.json")
    outputs = stats["layers"][""][0]["outputs"]
    assert list(outputs) == ["output.hidden_states.0", "output.logits"]
    remove_observer(model)


def test_shared_module_calls_are_not_overwritten(
    telemetry_dir: Path,
    detailed_sink: DirectorySink,
) -> None:
    """Preserve aliases and ordered calls when one module object is reused."""
    model = _SharedModuleModel()
    inject_observer(model, sampler=AlwaysSampler(), sink=detailed_sink)

    model(torch.randn(2, 4))

    stats = read_stats(telemetry_dir / "details.json")
    assert stats["module_catalog"]["first"]["aliases"] == ["first", "second"]
    assert [call["call_index"] for call in stats["layers"]["first"]] == [0, 1]
    remove_observer(model)


def test_multiple_forwards_are_correlated_with_their_backwards(
    telemetry_dir: Path,
    detailed_sink: DirectorySink,
) -> None:
    """Keep gradient telemetry separated for forwards combined into one backward."""
    model = nn.Linear(4, 3)
    inject_observer(model, sampler=AlwaysSampler(), sink=detailed_sink)
    first = model(torch.randn(2, 4))
    second = model(torch.randn(5, 4))

    (first.sum() + second.sum()).backward()

    stats = read_stats(telemetry_dir / "details.json")
    gradients = stats["layers"][""][0]["output_gradients"]["grad_output"]
    assert stats["samples_observed"] == 2
    assert stats["backward_samples_observed"] == 2
    assert gradients["observations"] == 2
    assert gradients["shape_changes"] == 1
    remove_observer(model)


def test_unused_differentiable_output_does_not_block_backward_aggregation(
    telemetry_dir: Path,
    detailed_sink: DirectorySink,
) -> None:
    """Finalize backward telemetry when only a subset of outputs contributes to loss."""
    model = _PartiallyUsedOutputs()
    inject_observer(model, sampler=AlwaysSampler(), sink=detailed_sink)
    used, _unused = model(torch.randn(2, 4, requires_grad=True))

    used.sum().backward()

    stats = read_stats(telemetry_dir / "details.json")
    gradients = stats["layers"][""][0]["output_gradients"]
    assert stats["backward_samples_observed"] == 1
    assert set(gradients) == {"grad_output.0"}
    remove_observer(model)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_supported_floating_dtypes_are_recorded(
    telemetry_dir: Path,
    detailed_sink: DirectorySink,
    dtype: torch.dtype,
) -> None:
    """Reduce and preserve metadata for every supported floating dtype."""
    model = nn.Identity()
    inject_observer(model, sampler=AlwaysSampler(), sink=detailed_sink)

    model(torch.ones(4, dtype=dtype))

    stats = read_stats(telemetry_dir / "details.json")
    output = stats["layers"][""][0]["outputs"]["output"]
    assert output["dtype"] == str(dtype).removeprefix("torch.")
    assert output["latest_statistics"]["mean"] == pytest.approx(1.0)
    remove_observer(model)


def test_statistics_use_finite_values_and_report_fraction(
    telemetry_dir: Path,
    detailed_sink: DirectorySink,
) -> None:
    """Keep useful finite statistics while exposing non-finite prevalence."""
    model = nn.Identity()
    inject_observer(model, sampler=AlwaysSampler(), sink=detailed_sink)

    model(torch.tensor([float("nan"), float("inf"), 2.0]))

    live = read_stats(telemetry_dir / "details.json")
    latest = live["layers"][""][0]["outputs"]["output"]["latest_statistics"]
    assert latest["mean"] == pytest.approx(2.0)
    assert latest["std"] == pytest.approx(0.0)
    assert latest["rms"] == pytest.approx(2.0)
    assert latest["max_abs"] == pytest.approx(2.0)
    assert latest["finite_fraction"] == pytest.approx(1 / 3)
    remove_observer(model)


def test_empty_tensor_records_unavailable_statistics(
    telemetry_dir: Path,
    detailed_sink: DirectorySink,
) -> None:
    """Record reason-carrying unavailable metrics for empty tensors."""
    model = nn.Identity()
    inject_observer(model, sampler=AlwaysSampler(), sink=detailed_sink)

    model(torch.empty(0))

    stats = read_stats(telemetry_dir / "details.json")
    output = stats["layers"][""][0]["outputs"]["output"]
    assert output["latest_statistics"] == {}
    assert {
        "finite_fraction",
        "max_abs",
        "mean",
        "rms",
        "std",
    }.issubset(output["latest_unavailable_statistics"])
    remove_observer(model)


def test_live_json_contains_lossless_histogram_data(
    telemetry_dir: Path,
    detailed_sink: DirectorySink,
) -> None:
    """Persist bins, outliers, moments, and non-finite counts in explicit JSON details."""
    model = nn.Identity()
    inject_observer(
        model,
        sampler=AlwaysSampler(),
        histograms=[
            histogram(bins=2, value_range=(-1.0, 1.0), every_n_samples=1),
        ],
        sink=detailed_sink,
    )

    model(torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0, float("nan")]))

    stats = read_stats(telemetry_dir / "details.json")
    output = stats["layers"][""][0]["outputs"]["output"]
    record = require_histogram(output["histograms"]["distribution"]["aggregate"])
    assert record["bin_edges"] == [-1.0, 0.0, 1.0]
    assert record["bin_counts"] == [1, 2]
    assert record["underflow_count"] == 1
    assert record["overflow_count"] == 1
    assert record["finite_count"] == 5
    assert record["nonfinite_count"] == 1
    assert record["sum"] == pytest.approx(0.0)
    assert record["sum_squares"] == pytest.approx(10.0)
    remove_observer(model)


@pytest.mark.parametrize("error_policy", ["warn", "ignore"])
def test_reducer_errors_are_recorded(
    telemetry_dir: Path,
    detailed_sink: DirectorySink,
    error_policy: str,
) -> None:
    """Persist reducer failures under non-raising error policies."""
    model = nn.Identity()

    def broken_reducer(tensor: torch.Tensor) -> Mapping[str, ReducedScalar]:
        """Raise a stable failure for telemetry error-policy coverage."""
        del tensor
        raise RuntimeError("broken reducer")

    inject_observer(
        model,
        sampler=AlwaysSampler(),
        reducers=[broken_reducer],
        sink=detailed_sink,
        error_policy=error_policy,
    )

    warning = pytest.warns(RuntimeWarning, match="broken reducer")
    context = warning if error_policy == "warn" else nullcontext()
    with context:
        model(torch.ones(1))

    stats = read_stats(telemetry_dir / "details.json")
    assert stats["errors"][0]["exception_type"] == "RuntimeError"
    assert stats["errors"][0]["message"] == "broken reducer"
    remove_observer(model)


def test_raise_error_policy_propagates_reducer_failure(telemetry_dir: Path) -> None:
    """Allow explicit raising policy to interrupt model execution."""
    model = nn.Identity()

    def broken_reducer(tensor: torch.Tensor) -> Mapping[str, ReducedScalar]:
        """Raise a stable failure that must escape the observer."""
        del tensor
        raise RuntimeError("broken reducer")

    inject_observer(
        model,
        sampler=AlwaysSampler(),
        reducers=[broken_reducer],
        output_dir=telemetry_dir,
        error_policy="raise",
    )

    with pytest.raises(RuntimeError, match="broken reducer"):
        model(torch.ones(1))
    remove_observer(model)


def test_live_record_is_versioned_and_self_describing(
    telemetry_dir: Path,
    detailed_sink: DirectorySink,
) -> None:
    """Emit one versioned live record with run, module, and reducer metadata."""
    model = nn.Identity()
    inject_observer(model, sampler=AlwaysSampler(), sink=detailed_sink)
    model(torch.ones(1))

    stats = read_stats(telemetry_dir / "details.json")
    run = stats["run"]
    assert stats["schema_version"] == 5
    assert run["schema_version"] == 5
    assert run["created_at"] == IsNow(iso_string=True, tz=UTC)
    assert run["sampling"] == {"settings": {}, "type": "always"}
    assert run["collection"]["invocation_capture"] == "pytorch_hooks"
    assert run["collection"]["signals"] == ["module_outputs", "module_output_gradients"]
    metrics = run["collection"]["scalar_reducers"][0]["settings"]["metrics"]
    if not isinstance(metrics, list):
        raise TypeError("serialized reducer metrics must be a list")
    assert {"skewness", "excess_kurtosis", "p999_abs", "max_to_rms"}.issubset(metrics)
    assert stats == IsPartialDict(
        schema_version=5,
        samples_observed=1,
        backward_samples_observed=0,
        dropped_series=IsNonNegative,
    )
    remove_observer(model)
