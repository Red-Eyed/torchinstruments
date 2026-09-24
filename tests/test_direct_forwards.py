"""Coverage for mixed ``Module.__call__`` and direct ``forward`` capture."""

from __future__ import annotations

import copy
from enum import StrEnum
from pathlib import Path

import polars as pl
import pytest
import torch
from torch import nn

from torchinstruments import (
    AlwaysSampler,
    DirectorySink,
    inject_observer,
    leaf_modules,
    remove_observer,
)


class _InvocationStyle(StrEnum):
    """Select normal PyTorch dispatch or a direct forward-method call."""

    CALL = "call"
    FORWARD = "forward"


class _MixedInvocationModel(nn.Module):
    """Invoke one selected leaf through a configurable call boundary."""

    def __init__(self, leaf_style: _InvocationStyle) -> None:
        """Create a deterministic linear leaf and retain its invocation style."""
        super().__init__()
        self.linear = nn.Linear(4, 3, bias=False)
        self._leaf_style = leaf_style

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the linear projection through normal or direct dispatch."""
        if self._leaf_style is _InvocationStyle.FORWARD:
            return self.linear.forward(inputs)
        return self.linear(inputs)


@pytest.mark.parametrize("root_style", list(_InvocationStyle))
@pytest.mark.parametrize("leaf_style", list(_InvocationStyle))
def test_forward_capture_supports_call_and_forward_exactly_once(
    telemetry_dir: Path,
    detailed_sink: DirectorySink,
    root_style: _InvocationStyle,
    leaf_style: _InvocationStyle,
) -> None:
    """Capture all root/leaf call combinations without changing outputs or gradients."""
    observed = _MixedInvocationModel(leaf_style)
    baseline = copy.deepcopy(observed)
    inputs = torch.randn(2, 4)

    expected = _invoke(baseline, inputs, root_style)
    expected.square().sum().backward()
    state_before = {name: value.clone() for name, value in observed.state_dict().items()}

    inject_observer(
        observed,
        sampler=AlwaysSampler(),
        sink=detailed_sink,
        error_policy="raise",
    )
    actual = _invoke(observed, inputs, root_style)
    actual.square().sum().backward()

    assert torch.equal(actual, expected)
    assert observed.linear.weight.grad is not None
    assert baseline.linear.weight.grad is not None
    assert torch.equal(observed.linear.weight.grad, baseline.linear.weight.grad)
    assert all(
        torch.equal(state_before[name], value) for name, value in observed.state_dict().items()
    )

    remove_observer(observed)
    history = pl.read_parquet(telemetry_dir / "history.parquet")
    assert history["call_index"].unique().to_list() == [0]
    assert set(history["layer"]) == {"", "linear"}
    assert set(history["signal"]) == {"output", "output_gradient"}
    assert history["shape"].to_list() == [[2, 3]] * history.height

    assert "forward" not in observed.__dict__
    assert "forward" not in observed.linear.__dict__


def test_forward_capture_restores_an_existing_instance_override(telemetry_dir: Path) -> None:
    """Restore a caller-owned instance forward attribute instead of deleting it."""
    model = nn.Identity()

    def doubled(inputs: torch.Tensor) -> torch.Tensor:
        """Provide a stable caller-owned forward override."""
        return inputs * 2

    model.__dict__["forward"] = doubled
    original_override = model.__dict__["forward"]
    inject_observer(
        model,
        sampler=AlwaysSampler(),
        output_dir=telemetry_dir,
    )

    assert torch.equal(model.forward(torch.tensor(2.0)), torch.tensor(4.0))
    remove_observer(model)

    assert model.__dict__["forward"] is original_override


def _invoke(
    model: nn.Module,
    inputs: torch.Tensor,
    style: _InvocationStyle,
) -> torch.Tensor:
    """Invoke a root module through the selected public or direct boundary."""
    if style is _InvocationStyle.FORWARD:
        return model.forward(inputs)
    return model(inputs)


def test_nested_shared_modules_are_intercepted_once(telemetry_dir: Path) -> None:
    """Include composite blocks and root without installing twice through shared aliases."""
    block = nn.Sequential(nn.Linear(2, 2))
    model = nn.Sequential(block, block)
    originals = {name: module.forward for name, module in model.named_modules()}
    inject_observer(model, output_dir=telemetry_dir, error_policy="raise")
    try:
        model.forward(torch.ones(1, 2)).sum().backward()
        history = pl.read_parquet(telemetry_dir / "history.parquet")
        calls = history.filter(
            (pl.col("signal") == "output") & (pl.col("metric") == "mean")
        ).select("layer", "call_index")
        assert set(calls.iter_rows()) == {("", 0), ("0", 0), ("0", 1), ("0.0", 0), ("0.0", 1)}
        assert calls.height == 5
        catalog = pl.read_json(telemetry_dir / "result.json")
        assert catalog.filter(pl.col("layer") == "0")["aliases"].item().to_list() == ["0", "1"]
    finally:
        remove_observer(model)
    assert {name: module.forward for name, module in model.named_modules()} == originals


def test_leaf_selector_remains_available(telemetry_dir: Path) -> None:
    """Allow callers to retain leaf-only records while capturing direct forwards by default."""
    model = nn.Sequential(nn.Sequential(nn.Linear(2, 2)))
    inject_observer(model, selector=leaf_modules(), output_dir=telemetry_dir, error_policy="raise")
    try:
        model.forward(torch.ones(1, 2))
        assert pl.read_json(telemetry_dir / "result.json")["layer"].to_list() == ["0.0"]
        assert pl.read_parquet(telemetry_dir / "history.parquet").height == 9
    finally:
        remove_observer(model)


def test_removal_preserves_a_later_forward_replacement(telemetry_dir: Path) -> None:
    """Restore only interception still owned by the observer."""
    model = nn.Identity()
    inject_observer(model, output_dir=telemetry_dir)

    def replacement(inputs: torch.Tensor) -> torch.Tensor:
        """Represent a caller's subsequent customization."""
        return inputs + 1

    model.__dict__["forward"] = replacement
    remove_observer(model)
    assert model.__dict__["forward"] is replacement
