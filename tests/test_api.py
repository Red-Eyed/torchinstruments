"""Tests for observer injection, removal, and model-behavior preservation."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path

import pytest
import torch
from torch import nn

from tests.json_records import read_snapshot
from torchinstruments import (
    AlwaysSampler,
    ObserverAlreadyAttachedError,
    has_observer,
    inject_observer,
    remove_observer,
)
from torchinstruments.reducers import ReducedScalar
from torchinstruments.sampling import SamplingEvent


def test_injection_does_not_change_state_dict(
    linear_model: nn.Linear,
    telemetry_dir: Path,
) -> None:
    """Keep every checkpoint key and tensor unchanged after injection."""
    before = {name: value.clone() for name, value in linear_model.state_dict().items()}

    result = inject_observer(linear_model, sampler=AlwaysSampler(), output_dir=telemetry_dir)

    after = linear_model.state_dict()
    assert result is None
    assert before.keys() == after.keys()
    assert all(torch.equal(before[name], after[name]) for name in before)

    remove_observer(linear_model)


def test_forward_output_is_bit_identical(
    linear_model: nn.Linear,
    telemetry_dir: Path,
) -> None:
    """Preserve model outputs bit-for-bit while sampled hooks are active."""
    inputs = torch.randn(2, 4)
    expected = linear_model(inputs)
    inject_observer(linear_model, sampler=AlwaysSampler(), output_dir=telemetry_dir)

    actual = linear_model(inputs)

    assert torch.equal(actual, expected)
    remove_observer(linear_model)


def test_gradients_are_unchanged(
    linear_model: nn.Linear,
    telemetry_dir: Path,
) -> None:
    """Preserve parameter gradients bit-for-bit under instrumentation."""
    baseline = copy.deepcopy(linear_model)
    observed = copy.deepcopy(linear_model)
    inputs = torch.randn(2, 4)

    baseline(inputs).square().sum().backward()
    inject_observer(observed, sampler=AlwaysSampler(), output_dir=telemetry_dir)
    observed(inputs).square().sum().backward()

    for baseline_parameter, observed_parameter in zip(
        baseline.parameters(), observed.parameters(), strict=True
    ):
        assert baseline_parameter.grad is not None
        assert observed_parameter.grad is not None
        assert torch.equal(baseline_parameter.grad, observed_parameter.grad)

    remove_observer(observed)


def test_duplicate_injection_is_rejected(
    linear_model: nn.Linear,
    telemetry_dir: Path,
) -> None:
    """Reject a second observer instead of silently duplicating hooks."""
    inject_observer(linear_model, sampler=AlwaysSampler(), output_dir=telemetry_dir)

    with pytest.raises(ObserverAlreadyAttachedError, match="already has"):
        inject_observer(
            linear_model,
            sampler=AlwaysSampler(),
            output_dir=telemetry_dir / "duplicate",
        )

    remove_observer(linear_model)


def test_remove_observer_stops_future_collection(
    linear_model: nn.Linear,
    telemetry_dir: Path,
) -> None:
    """Stop collecting new snapshots after explicit observer removal."""
    inject_observer(linear_model, sampler=AlwaysSampler(), output_dir=telemetry_dir)
    linear_model(torch.randn(1, 4))
    remove_observer(linear_model)

    linear_model(torch.randn(1, 4))

    assert not has_observer(linear_model)
    assert [path.name for path in (telemetry_dir / "snapshots").iterdir()] == ["000000.json"]


def test_remove_observer_detaches_pending_gradient_hooks(
    linear_model: nn.Linear,
    telemetry_dir: Path,
) -> None:
    """Prevent pending graph callbacks from writing after observer removal."""
    inject_observer(linear_model, sampler=AlwaysSampler(), output_dir=telemetry_dir)
    output = linear_model(torch.randn(1, 4))
    remove_observer(linear_model)

    output.sum().backward()

    snapshot = read_snapshot(telemetry_dir / "snapshots" / "000000.json")
    assert snapshot["state"] == "forward_complete"


def test_remove_observer_is_idempotent(linear_model: nn.Linear) -> None:
    """Allow repeated cleanup calls for models without attached observers."""
    remove_observer(linear_model)
    remove_observer(linear_model)

    assert not has_observer(linear_model)


class _NeverSampler:
    """Disable collection while preserving the normal injected hook path."""

    def should_sample(self, event: SamplingEvent) -> bool:
        """Reject every root-forward sampling event."""
        del event
        return False


def test_inactive_hooks_do_not_call_reducers(
    linear_model: nn.Linear,
    telemetry_dir: Path,
) -> None:
    """Avoid invoking reducers from selected-module hooks outside sampled forwards."""
    calls = 0

    def counting_reducer(tensor: torch.Tensor) -> Mapping[str, ReducedScalar]:
        """Count calls so the inactive hook path can prove reducers remain untouched."""
        nonlocal calls
        del tensor
        calls += 1
        return {"count": calls}

    inject_observer(
        linear_model,
        sampler=_NeverSampler(),
        reducers=[counting_reducer],
        output_dir=telemetry_dir,
    )

    for _ in range(20):
        linear_model(torch.randn(1, 4))

    assert calls == 0
    assert not any((telemetry_dir / "snapshots").iterdir())
    remove_observer(linear_model)


def test_forward_only_execution_writes_complete_forward_snapshot(
    linear_model: nn.Linear,
    telemetry_dir: Path,
) -> None:
    """Persist forward telemetry even when no backward pass follows."""
    inject_observer(linear_model, sampler=AlwaysSampler(), output_dir=telemetry_dir)

    linear_model(torch.randn(2, 4))

    snapshot = read_snapshot(telemetry_dir / "snapshots" / "000000.json")
    assert snapshot["state"] == "forward_complete"
    assert snapshot["modules"][""][0]["outputs"]["output"]["shape"] == [2, 3]
    remove_observer(linear_model)
