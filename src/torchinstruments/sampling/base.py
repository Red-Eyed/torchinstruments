"""Sampling-policy events and structural interfaces."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from torchinstruments.records import JsonScalar


@dataclass(frozen=True)
class SamplingEvent:
    """Describe one root-forward sampling opportunity using monotonic time."""

    forward_index: int
    monotonic_time: float


class SamplingPolicy(Protocol):
    """Decide whether a root forward should create a telemetry sample."""

    def should_sample(self, event: SamplingEvent) -> bool:
        """Return whether the supplied root-forward event should be sampled."""
        ...


@runtime_checkable
class DescribedSamplingPolicy(Protocol):
    """Optionally expose stable metadata for a sampling policy."""

    def sampling_type(self) -> str:
        """Return the stable policy type written to run metadata."""
        ...

    def sampling_settings(self) -> Mapping[str, JsonScalar]:
        """Return JSON-compatible policy settings written once per run."""
        ...
