"""Sampling-policy events and structural interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from torchinstruments.records import JsonScalar


@dataclass(frozen=True)
class SamplingEvent:
    """Describe a root or independent child invocation using monotonic time.

    An empty module name denotes a root invocation. Independent children have their
    own invocation counts and use their canonical module path as the sampling key.
    """

    forward_index: int
    monotonic_time: float
    module_name: str = ""


class SamplingPolicy(Protocol):
    """Decide whether a root or independent child invocation should be sampled."""

    def should_sample(self, event: SamplingEvent) -> bool:
        """Select an event whose count is local to its module sampling scope."""
        ...


@runtime_checkable
class DescribedSamplingPolicy(Protocol):
    """Optionally expose stable metadata for a sampling policy."""

    def sampling_type(self) -> str:
        """Return the stable policy type written to run metadata."""
        ...

    def sampling_settings(self) -> dict[str, JsonScalar]:
        """Return JSON-compatible policy settings written once per run."""
        ...
