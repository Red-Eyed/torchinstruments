"""Coverage for mixed ``Module.__call__`` and direct ``forward`` capture."""

from __future__ import annotations

import copy
from enum import StrEnum
from pathlib import Path

import pytest
import torch
from torch import nn

from tests.json_records import read_run, read_snapshot
from torchinstruments import AlwaysSampler, inject_observer, remove_observer


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
        capture_direct_forwards=True,
        output_dir=telemetry_dir,
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

    snapshot = read_snapshot(telemetry_dir / "snapshots" / "000000.json")
    run = read_run(telemetry_dir / "run.json")
    assert run["collection"]["invocation_capture"] == "forward_wrappers"
    calls = snapshot["modules"]["linear"]
    assert len(calls) == 1
    assert calls[0]["call_index"] == 0
    assert calls[0]["outputs"]["output"]["shape"] == [2, 3]
    assert calls[0]["output_gradients"]["grad_output"]["shape"] == [2, 3]

    remove_observer(observed)
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
        capture_direct_forwards=True,
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
