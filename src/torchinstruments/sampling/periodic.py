"""Sampling policies based on root-forward counts."""

from __future__ import annotations

from collections.abc import Mapping

from torchinstruments.records import JsonScalar
from torchinstruments.sampling.base import SamplingEvent


class AlwaysSampler:
    """Sample every root forward."""

    def should_sample(self, event: SamplingEvent) -> bool:
        """Select every sampling event."""
        return True

    def sampling_type(self) -> str:
        """Return the stable run-metadata name for this policy."""
        return "always"

    def sampling_settings(self) -> Mapping[str, JsonScalar]:
        """Return the empty settings required by an unconditional policy."""
        return {}


class EveryNForwardsSampler:
    """Sample each Nth root forward without assuming a trainer step."""

    def __init__(self, n: int) -> None:
        """Configure a positive root-forward sampling period."""
        if n <= 0:
            raise ValueError("n must be greater than zero")
        self._n = n

    def should_sample(self, event: SamplingEvent) -> bool:
        """Select events whose one-based forward count is divisible by ``n``."""
        return (event.forward_index + 1) % self._n == 0

    def sampling_type(self) -> str:
        """Return the stable run-metadata name for this policy."""
        return "every_n_forwards"

    def sampling_settings(self) -> Mapping[str, JsonScalar]:
        """Return the configured forward period for run metadata."""
        return {"n": self._n}
