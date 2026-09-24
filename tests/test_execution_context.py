"""Regression coverage for mode identity independent of autograd recording."""

from pathlib import Path

import polars as pl
import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch import nn

from torchinstruments import AlwaysSampler, inject_observer, remove_observer
from torchinstruments.history.summary import aggregate_history


def test_modes_survive_delayed_backward_and_split_summaries(telemetry_dir: Path) -> None:
    """Keep each forward's context when backward arrives under another mode."""
    model = nn.Linear(2, 1)
    inject_observer(model, output_dir=telemetry_dir, sampler=AlwaysSampler(), error_policy="raise")
    train = model(torch.ones(1, 2))
    model.eval()
    evaluation = model(torch.ones(1, 2))
    with torch.no_grad():
        model(torch.ones(1, 2))
    model.train()
    evaluation.sum().backward()
    train.sum().backward()
    model(torch.ones(1, 2)).sum().backward()
    remove_observer(model)
    history = pl.read_parquet(telemetry_dir / "history.parquet")
    gradients = history.filter(
        (pl.col("signal") == "output_gradient") & (pl.col("metric") == "mean")
    )
    assert gradients.sort("sample_id")["mode"].to_list() == ["train", "eval", "train"]
    tensors = (
        pl.read_json(telemetry_dir / "result.json")
        .explode("tensors", empty_as_null=True)
        .unnest("tensors")
    )
    contexts = tensors.filter(pl.col("signal") == "output").select("mode", "grad_enabled")
    assert set(contexts.iter_rows()) == {("train", True), ("eval", True), ("eval", False)}
    tags = EventAccumulator(str(telemetry_dir / "tensorboard")).Reload().Tags()["histograms"]
    assert isinstance(tags, list)
    assert any("eval/grad_enabled_false" in tag for tag in tags)
    assert any("train/grad_enabled_true" in tag for tag in tags)


def test_mixed_module_modes_and_no_grad_training(telemetry_dir: Path) -> None:
    """Record child modes rather than guessing from the root or gradient availability."""
    frozen = nn.BatchNorm1d(2)
    model = nn.Sequential(frozen, nn.Linear(2, 1)).train()
    frozen.eval()
    inject_observer(
        model,
        output_dir=telemetry_dir,
        sampler=AlwaysSampler(),
        error_policy="raise",
    )
    with torch.no_grad():
        model(torch.ones(2, 2))
    remove_observer(model)
    contexts = (
        pl.read_parquet(telemetry_dir / "history.parquet")
        .select("layer", "mode", "grad_enabled")
        .unique()
    )
    assert set(contexts.iter_rows()) == {
        ("", "train", False),
        ("0", "eval", False),
        ("1", "train", False),
    }


def test_legacy_history_requires_explicit_context(telemetry_dir: Path) -> None:
    """Reject old histories rather than silently attributing them to training."""
    model = nn.Identity()
    inject_observer(model, output_dir=telemetry_dir, error_policy="raise")
    model(torch.ones(2))
    remove_observer(model)
    legacy = telemetry_dir / "legacy.parquet"
    pl.read_parquet(telemetry_dir / "history.parquet").drop("mode", "grad_enabled").write_parquet(
        legacy
    )
    with pytest.raises(ValueError, match="history lacks execution context"):
        aggregate_history(legacy, 20)
