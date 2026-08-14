"""Tests for bounded live temporal, distribution, and histogram aggregation."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from tests.json_records import read_stats
from torchinstruments import (
    AlwaysSampler,
    DirectorySink,
    IndicatorConfig,
    LiveAggregator,
    histogram,
    inject_observer,
    remove_observer,
)


class _DynamicOutputKeys(nn.Module):
    """Return a new mapping key on every invocation."""

    def __init__(self) -> None:
        """Initialize the monotonic output-key suffix."""
        super().__init__()
        self._invocation = 0

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        """Expose one tensor under an invocation-dependent path."""
        key = f"value_{self._invocation}"
        self._invocation += 1
        return {key: inputs}


class _GrowingSharedCalls(nn.Module):
    """Invoke one leaf an increasing number of times across forwards."""

    def __init__(self) -> None:
        """Create one selected leaf and an invocation-count limit."""
        super().__init__()
        self.identity = nn.Identity()
        self._calls = 0

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the result of an increasing number of shared-leaf calls."""
        self._calls += 1
        output = inputs
        for _call in range(self._calls):
            output = self.identity(output)
        return output


def test_live_indicators_identify_a_persistent_linear_drift(telemetry_dir: Path) -> None:
    """Summarize trend, momentum, persistence, and extrema without retaining samples."""
    model = nn.Identity()
    inject_observer(model, sampler=AlwaysSampler(), output_dir=telemetry_dir)

    for value in range(1, 26):
        model(torch.tensor(float(value)))

    stats = read_stats(telemetry_dir / "stats.json")
    series = stats["layers"][""][0]["outputs"]["output"]["statistics"]["rms"]
    indicators = series["indicators"]
    assert series["count"] == 25
    assert series["warmup_complete"] is True
    assert series["first"]["value"] == pytest.approx(1.0)
    assert series["latest"]["value"] == pytest.approx(25.0)
    assert indicators["mean"] == pytest.approx(13.0)
    assert indicators["standard_deviation"] == pytest.approx(52**0.5)
    assert indicators["linear_slope_per_sample"] == pytest.approx(1.0)
    assert indicators["slope_r_squared"] == pytest.approx(1.0)
    assert indicators["momentum_5_samples"] == pytest.approx(5.0)
    assert indicators["relative_momentum_5_samples"] == pytest.approx(0.25)
    assert indicators["consecutive_increases"] == 24
    assert indicators["maximum_drawdown"] == pytest.approx(0.0)
    assert indicators["maximum_runup"] == pytest.approx(24.0)
    assert indicators["lag1_autocorrelation"] == pytest.approx(1.0)
    assert indicators["oscillation_fraction"] == pytest.approx(0.0)
    remove_observer(model)


def test_live_indicators_distinguish_oscillation_from_drift(telemetry_dir: Path) -> None:
    """Expose alternating instability that lifetime mean and variance cannot explain."""
    model = nn.Identity()
    inject_observer(model, sampler=AlwaysSampler(), output_dir=telemetry_dir)

    for index in range(25):
        model(torch.tensor(1.0 if index % 2 == 0 else 2.0))

    stats = read_stats(telemetry_dir / "stats.json")
    indicators = stats["layers"][""][0]["outputs"]["output"]["statistics"]["rms"]["indicators"]
    assert abs(indicators["linear_slope_per_sample"]) < 0.01
    assert indicators["lag1_autocorrelation"] == pytest.approx(-1.0)
    assert indicators["oscillation_fraction"] == pytest.approx(1.0)
    assert indicators["consecutive_increases"] == 0
    remove_observer(model)


def test_fixed_histograms_merge_live_without_per_sample_files(telemetry_dir: Path) -> None:
    """Add fixed-bin counts exactly while retaining only canonical live telemetry."""
    model = nn.Identity()
    inject_observer(
        model,
        sampler=AlwaysSampler(),
        histograms=[histogram(bins=2, value_range=(-1.0, 1.0), every_n_samples=1)],
        output_dir=telemetry_dir,
    )

    model(torch.tensor([-1.0, 0.5]))
    model(torch.tensor([-0.5, 1.0]))

    stats = read_stats(telemetry_dir / "stats.json")
    summary = stats["layers"][""][0]["outputs"]["output"]["histograms"]["distribution"]
    aggregate = summary["aggregate"]
    assert summary["samples"] == 2
    assert aggregate["bin_counts"] == [2, 2]
    assert aggregate["finite_count"] == 4
    assert sorted(path.name for path in telemetry_dir.iterdir()) == ["index.md", "stats.json"]
    remove_observer(model)


def test_dynamic_histogram_retains_latest_and_explains_unmergeable_history(
    telemetry_dir: Path,
) -> None:
    """Keep a changing-bin distribution useful without pretending it merges exactly."""
    model = nn.Identity()
    inject_observer(
        model,
        sampler=AlwaysSampler(),
        histograms=[histogram(bins=2, every_n_samples=1)],
        output_dir=telemetry_dir,
    )

    model(torch.tensor([0.0, 1.0]))
    model(torch.tensor([10.0, 20.0]))

    stats = read_stats(telemetry_dir / "stats.json")
    summary = stats["layers"][""][0]["outputs"]["output"]["histograms"]["distribution"]
    assert summary["latest"]["minimum"] == pytest.approx(10.0)
    assert summary["aggregate"] == {
        "status": "absent",
        "reason": (
            "histogram bin edges changed; configure a fixed value_range for live aggregation"
        ),
    }
    remove_observer(model)


def test_live_series_memory_cap_counts_distinct_dropped_metrics(telemetry_dir: Path) -> None:
    """Bound dynamic series state independently of the number of sampled forwards."""
    config = IndicatorConfig(max_series=1)
    sink = DirectorySink(
        telemetry_dir,
        aggregator_factory=lambda: LiveAggregator(config),
    )
    model = nn.Identity()
    inject_observer(model, sampler=AlwaysSampler(), sink=sink)

    model(torch.tensor([1.0, 2.0]))
    first = read_stats(telemetry_dir / "stats.json")
    model(torch.tensor([2.0, 3.0]))
    second = read_stats(telemetry_dir / "stats.json")

    assert first["dropped_series"] == len(config.temporal_metrics) - 1
    assert second["dropped_series"] == first["dropped_series"]
    assert second["indicator_configuration"]["max_series"] == 1
    assert second["indicator_configuration"]["momentum_horizons"] == [1, 5, 20]
    remove_observer(model)


def test_live_file_size_is_bounded_by_structure_not_training_duration(
    telemetry_dir: Path,
) -> None:
    """Prevent sampled-forward count from creating an unbounded persisted history."""
    model = nn.Identity()
    inject_observer(model, sampler=AlwaysSampler(), output_dir=telemetry_dir)

    for _sample in range(25):
        model(torch.ones(4))
    initial_size = (telemetry_dir / "stats.json").stat().st_size

    for _sample in range(225):
        model(torch.ones(4))
    final_size = (telemetry_dir / "stats.json").stat().st_size

    assert final_size - initial_size < 2_000
    assert sorted(path.name for path in telemetry_dir.iterdir()) == ["index.md", "stats.json"]
    remove_observer(model)


def test_dynamic_tensor_paths_respect_the_global_structure_limit(telemetry_dir: Path) -> None:
    """Prevent invocation-dependent mapping keys from growing live state forever."""
    config = IndicatorConfig(max_tensor_paths=1)
    model = _DynamicOutputKeys()
    sink = DirectorySink(telemetry_dir, aggregator_factory=lambda: LiveAggregator(config))
    inject_observer(
        model,
        sampler=AlwaysSampler(),
        selector=lambda name, module: name == "" and module is model,
        sink=sink,
    )

    model(torch.tensor(1.0))
    model(torch.tensor(2.0))

    stats = read_stats(telemetry_dir / "stats.json")
    assert set(stats["layers"][""][0]["outputs"]) == {"output.value_0"}
    assert stats["dropped_tensor_path_observations"] == 1
    remove_observer(model)


def test_dynamic_call_positions_respect_the_global_structure_limit(telemetry_dir: Path) -> None:
    """Prevent an increasing number of shared-module calls from growing live state forever."""
    config = IndicatorConfig(max_module_calls=1)
    model = _GrowingSharedCalls()
    sink = DirectorySink(telemetry_dir, aggregator_factory=lambda: LiveAggregator(config))
    inject_observer(model, sampler=AlwaysSampler(), sink=sink)

    model(torch.tensor(1.0))
    model(torch.tensor(2.0))

    stats = read_stats(telemetry_dir / "stats.json")
    assert [call["call_index"] for call in stats["layers"]["identity"]] == [0]
    assert stats["dropped_module_call_observations"] == 1
    remove_observer(model)


def test_histogram_identities_respect_the_global_structure_limit(telemetry_dir: Path) -> None:
    """Prevent custom or configured histogram names from bypassing structural limits."""
    config = IndicatorConfig(max_histograms=1)
    model = nn.Identity()
    sink = DirectorySink(telemetry_dir, aggregator_factory=lambda: LiveAggregator(config))
    inject_observer(
        model,
        sampler=AlwaysSampler(),
        histograms=[
            histogram(name="first", bins=2, value_range=(-1.0, 1.0), every_n_samples=1),
            histogram(name="second", bins=2, value_range=(-1.0, 1.0), every_n_samples=1),
        ],
        sink=sink,
    )

    model(torch.tensor([0.0]))

    stats = read_stats(telemetry_dir / "stats.json")
    histograms = stats["layers"][""][0]["outputs"]["output"]["histograms"]
    assert set(histograms) == {"first"}
    assert stats["dropped_histogram_observations"] == 1
    remove_observer(model)
