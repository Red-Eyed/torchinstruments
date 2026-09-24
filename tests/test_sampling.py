"""Tests for built-in root-forward sampling policies."""

from __future__ import annotations

from datetime import timedelta

import pytest

from torchinstruments import AlwaysSampler, EveryNForwardsSampler, TimedSampler
from torchinstruments.sampling import SamplingEvent


def test_timed_sampler_starts_immediately_then_waits() -> None:
    """Select the first eligible event at each monotonic deadline."""
    sampler = TimedSampler(timedelta(seconds=10), clock=lambda: 100.0)

    assert sampler.should_sample(SamplingEvent(forward_index=0, monotonic_time=100.0))
    assert not sampler.should_sample(SamplingEvent(forward_index=1, monotonic_time=109.9))
    assert sampler.should_sample(SamplingEvent(forward_index=1, monotonic_time=110.0))
    assert not sampler.should_sample(SamplingEvent(forward_index=2, monotonic_time=119.9))
    assert sampler.should_sample(SamplingEvent(forward_index=3, monotonic_time=120.0))


@pytest.mark.parametrize("n", [0, -1])
def test_periodic_sampler_rejects_non_positive_period(n: int) -> None:
    """Reject forward periods that could never define valid cadence."""
    with pytest.raises(ValueError, match="greater than zero"):
        EveryNForwardsSampler(n)


def test_every_n_forwards_sampler_counts_root_forwards() -> None:
    """Count root forwards with one-based periodic sampling semantics."""
    sampler = EveryNForwardsSampler(3)

    decisions = [
        sampler.should_sample(SamplingEvent(forward_index=index, monotonic_time=float(index)))
        for index in range(7)
    ]

    assert decisions == [False, False, True, False, False, True, False]


def test_independent_layers_have_separate_deadlines() -> None:
    """Avoid starving later layers when an earlier layer consumes its interval."""
    sampler = TimedSampler(timedelta(seconds=1), clock=lambda: 100.0)
    for module in ["", "encoder", "head"]:
        assert sampler.should_sample(SamplingEvent(0, 100.0, module))
        assert not sampler.should_sample(SamplingEvent(1, 100.5, module))
        assert sampler.should_sample(SamplingEvent(2, 101.0, module))


def test_always_sampler_selects_every_forward() -> None:
    """Select every event under the unconditional policy."""
    sampler = AlwaysSampler()

    assert sampler.should_sample(SamplingEvent(forward_index=0, monotonic_time=0.0))
