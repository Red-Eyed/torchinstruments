"""Structural contracts and normalized results for tensor reducers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, TypeAlias, runtime_checkable

import torch

from torchinstruments.records import JsonSetting

ReducedScalar: TypeAlias = bool | float | int | torch.Tensor


class Reducer(Protocol):
    """Convert one detached tensor into named compact scalar values."""

    def __call__(self, tensor: torch.Tensor) -> Mapping[str, ReducedScalar]:
        """Return named scalar diagnostics without retaining the autograd graph."""
        ...


@runtime_checkable
class DescribedReducer(Protocol):
    """Expose stable run metadata for a scalar or histogram reducer."""

    def reducer_type(self) -> str:
        """Return the stable reducer family name written to run metadata."""
        ...

    def reducer_settings(self) -> Mapping[str, JsonSetting]:
        """Return JSON-compatible settings that determine reducer behavior."""
        ...


@dataclass(frozen=True)
class ReductionResult:
    """Separate usable scalar statistics from reason-carrying unavailable metrics."""

    stats: Mapping[str, float]
    unavailable_stats: Mapping[str, str]
