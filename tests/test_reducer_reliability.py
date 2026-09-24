"""Large-tensor, accelerator, and independent measurement failure regressions."""

from pathlib import Path

import polars as pl
import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch import nn

from torchinstruments import AlwaysSampler, inject_observer, remove_observer
from torchinstruments.reducers import HistogramReductionResult, default_reducers, reduce_tensor
from torchinstruments.reducers.quantiles import exact_quartiles


def test_default_statistics_above_quantile_limit() -> None:
    """Keep exact quartiles for a structured tensor larger than torch.quantile accepts."""
    values = torch.arange(2**24 + 1, dtype=torch.float32)
    result = reduce_tensor(values, default_reducers())
    assert result.stats["p25"] == 2**22
    assert result.stats["p50"] == 2**23
    assert result.stats["p75"] == 3 * 2**22
    assert result.stats["nonfinite_fraction"] == 0


def test_rare_nonfinite_value_does_not_round_to_healthy() -> None:
    """Count invalid entries directly instead of subtracting a rounded finite fraction."""
    values = torch.ones(2**25)
    values[0] = float("nan")
    result = reduce_tensor(values, default_reducers()[:1])
    assert result.stats["nonfinite_fraction"] == 1 / 2**25


@pytest.mark.parametrize("values", [[1.0], [1.0, 3.0], [0.0, 0.0, 2.0, 9.0, 10.0]])
def test_exact_selection_matches_linear_quantiles(values: list[float]) -> None:
    """Match fractional ranks, ties, and singleton populations without approximations."""
    tensor = torch.tensor(values)
    actual = exact_quartiles(tensor)
    expected = torch.quantile(tensor, torch.tensor([0.25, 0.5, 0.75]))
    torch.testing.assert_close(torch.stack(list(actual.values())), expected)


def test_quartile_failure_keeps_moments_and_reason(
    telemetry_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep default metric identities even when a numerical kernel fails."""

    def fail(*args: object, **kwargs: object) -> torch.Tensor:
        """Simulate an unavailable quantile operation."""
        raise RuntimeError("quantile kernel unavailable")

    monkeypatch.setattr(torch, "quantile", fail)
    model = nn.Identity()
    inject_observer(model, output_dir=telemetry_dir, sampler=AlwaysSampler(), error_policy="ignore")
    model(torch.tensor([1.0, 3.0]))
    remove_observer(model)
    history = pl.read_parquet(telemetry_dir / "history.parquet")
    assert history.filter(pl.col("metric") == "mean")["value"].item() == 2
    quartiles = history.filter(pl.col("metric").is_in(["p25", "p50", "p75"]))
    assert quartiles["value"].null_count() == 3
    assert all(
        "quantile kernel unavailable" in reason for reason in quartiles["unavailable_reason"]
    )


def test_histogram_failure_preserves_scalars_and_gradients(telemetry_dir: Path) -> None:
    """Keep measurements independent when one output projection fails."""

    def broken(tensor: torch.Tensor, *, sample_id: int) -> HistogramReductionResult:
        """Fail without modifying model execution."""
        raise RuntimeError("broken histogram")

    model = nn.Linear(2, 1)
    inject_observer(
        model,
        output_dir=telemetry_dir,
        sampler=AlwaysSampler(),
        histograms=[broken],
        error_policy="ignore",
    )
    model(torch.ones(1, 2)).sum().backward()
    remove_observer(model)
    history = pl.read_parquet(telemetry_dir / "history.parquet")
    assert set(history["signal"]) == {"output", "output_gradient"}
    assert "broken histogram" in (telemetry_dir / "index.md").read_text()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="native MPS unavailable")
def test_native_mps_artifacts(telemetry_dir: Path) -> None:
    """Exercise the complete default measurement path on an actual MPS device."""
    model = nn.Linear(2, 1).to("mps")
    inject_observer(model, output_dir=telemetry_dir, sampler=AlwaysSampler(), error_policy="raise")
    model(torch.ones(1, 2, device="mps")).sum().backward()
    remove_observer(model)
    assert pl.read_parquet(telemetry_dir / "history.parquet").height == 18
    events = EventAccumulator(str(telemetry_dir / "tensorboard")).Reload()
    tags = events.Tags()["histograms"]
    assert isinstance(tags, list)
    assert len(tags) == 2
    for tag in tags:
        distribution = events.Histograms(tag)[0].histogram_value
        assert distribution.num == 1
        assert sum(distribution.bucket) == 1


def test_unrepresentable_histogram_reason_is_persisted(telemetry_dir: Path) -> None:
    """Retain scalar values when histogram moments overflow finite representation."""
    model = nn.Identity()
    inject_observer(model, output_dir=telemetry_dir, sampler=AlwaysSampler(), error_policy="raise")
    model(torch.tensor([1e200], dtype=torch.float64))
    remove_observer(model)
    history = pl.read_parquet(telemetry_dir / "history.parquet")
    assert history.filter(pl.col("metric") == "mean")["value"].item() == 1e200
    assert "histogram aggregates are not finite" in (telemetry_dir / "index.md").read_text()
