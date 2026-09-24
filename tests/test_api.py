"""Tests for observer injection, removal, and model-behavior preservation."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
from torch import nn

from torchinstruments import (
    ObserverAlreadyAttachedError,
    inject_observer,
    remove_observer,
)


def test_injection_does_not_change_state_dict(
    linear_model: nn.Linear,
    telemetry_dir: Path,
) -> None:
    """Keep every checkpoint key and tensor unchanged after injection."""
    before = {name: value.clone() for name, value in linear_model.state_dict().items()}

    result = inject_observer(linear_model, output_dir=telemetry_dir)

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
    inject_observer(linear_model, output_dir=telemetry_dir)

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
    inject_observer(observed, output_dir=telemetry_dir)
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
    inject_observer(linear_model, output_dir=telemetry_dir)

    with pytest.raises(ObserverAlreadyAttachedError, match="already has"):
        inject_observer(
            linear_model,
            output_dir=telemetry_dir / "duplicate",
        )

    remove_observer(linear_model)
