"""Regression coverage for gradient identity at mutation and graph boundaries."""

import gc
import weakref
from pathlib import Path

import polars as pl
import pytest
import torch
from torch import nn

from torchinstruments import AlwaysSampler, inject_observer, remove_observer


@pytest.mark.parametrize("inplace", [False, True])
def test_upstream_gradient_precedes_inplace_relu(telemetry_dir: Path, inplace: bool) -> None:
    """Observe the linear output before its gradient edge is rebased by ReLU."""
    linear = nn.Linear(2, 2, bias=False)
    model = nn.Sequential(linear, nn.ReLU(inplace=inplace))
    with torch.no_grad():
        linear.weight.copy_(torch.eye(2))
    inject_observer(model, output_dir=telemetry_dir, sampler=AlwaysSampler(), error_policy="raise")
    model(torch.tensor([[-1.0, 1.0]])).sum().backward()
    remove_observer(model)
    history = pl.read_parquet(telemetry_dir / "history.parquet")
    measured = history.filter(
        (pl.col("layer") == "0")
        & (pl.col("signal") == "output_gradient")
        & (pl.col("metric") == "mean")
    )
    assert measured["value"].to_list() == [0.5]
    assert linear.weight.grad is not None
    assert torch.equal(linear.weight.grad, torch.tensor([[0.0, 0.0], [-1.0, 1.0]]))


def test_only_first_backward_is_recorded(telemetry_dir: Path) -> None:
    """Do not silently combine repeated backwards over a retained graph."""
    model = nn.Linear(2, 1)
    inject_observer(model, output_dir=telemetry_dir, sampler=AlwaysSampler(), error_policy="raise")
    output = model(torch.ones(1, 2))
    output.sum().backward(retain_graph=True)
    (output.sum() * 7).backward()
    remove_observer(model)
    history = pl.read_parquet(telemetry_dir / "history.parquet")
    measured = history.filter(
        (pl.col("signal") == "output_gradient") & (pl.col("metric") == "mean")
    )
    assert measured["value"].to_list() == [1.0]


def test_autograd_grad_completion(telemetry_dir: Path) -> None:
    """Publish gradients when the caller requests derivatives without backward()."""
    model = nn.Linear(2, 1)
    inputs = torch.ones(1, 2, requires_grad=True)
    inject_observer(model, output_dir=telemetry_dir, sampler=AlwaysSampler(), error_policy="raise")
    output = model(inputs)
    torch.autograd.grad(output.sum(), inputs)
    remove_observer(model)
    history = pl.read_parquet(telemetry_dir / "history.parquet")
    assert set(history["signal"]) == {"output", "output_gradient"}


def test_abandoned_graph_is_reclaimable(telemetry_dir: Path) -> None:
    """Do not retain source tensors when a sampled forward never receives backward."""
    model = nn.Linear(2, 1)
    inject_observer(model, output_dir=telemetry_dir, sampler=AlwaysSampler(), error_policy="raise")
    output = model(torch.ones(1, 2))
    reference = weakref.ref(output)
    del output
    gc.collect()
    assert reference() is None
    remove_observer(model)
