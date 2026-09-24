"""Isolate reducer failures while retaining the observer's error policy."""

from collections.abc import Callable
from dataclasses import dataclass

import torch

from torchinstruments.measurement import TensorMeasurement
from torchinstruments.records import Absent, TensorRecord
from torchinstruments.reducers import (
    HistogramReducer,
    HistogramReductionResult,
    ReducedScalar,
    Reducer,
)
from torchinstruments.reducers.base import DescribedReducer


@dataclass(frozen=True)
class _GuardedScalar:
    """Apply failure policy to one reducer without discarding its successful siblings."""

    reducer: Reducer
    report: Callable[[Exception], None]

    def __call__(self, tensor: torch.Tensor) -> dict[str, ReducedScalar]:
        """Preserve known metric names as explicit absence after a reducer failure."""
        try:
            values = self.reducer(tensor)
            _validate_scalars(values)
            return values
        except Exception as error:
            self.report(error)
            reason = f"{type(error).__name__}: {error}"
            return {name: Absent(reason) for name in _metric_names(self.reducer)}


def _metric_names(reducer: Reducer) -> tuple[str, ...]:
    """Read known reducer output names without inventing identities for custom callables."""
    match reducer:
        case DescribedReducer():
            names = reducer.reducer_settings().get("metrics", ())
        case _:
            return ()
    match names:
        case tuple():
            return tuple(name for name in names if isinstance(name, str))
        case _:
            return ()


def _validate_scalars(values: dict[str, ReducedScalar]) -> None:
    """Reject malformed custom results before combining them with successful reducers."""
    for name, value in values.items():
        match value:
            case torch.Tensor():
                if value.numel() != 1:
                    raise ValueError(f"reducer metric {name!r} must be scalar")
            case float() | int() | Absent():
                continue
            case _:
                raise TypeError(f"unsupported scalar type for metric {name!r}")


@dataclass(frozen=True)
class _GuardedHistogram:
    """Apply failure policy without invalidating scalar history."""

    reducer: HistogramReducer
    report: Callable[[Exception], None]

    def __call__(self, tensor: torch.Tensor, *, sample_id: int) -> HistogramReductionResult:
        """Preserve a failed histogram as unavailable and report its exception."""
        try:
            return self.reducer(tensor, sample_id=sample_id)
        except Exception as error:
            self.report(error)
            return HistogramReductionResult(
                histograms={},
                unavailable_histograms={type(self.reducer).__qualname__: str(error)},
            )


class IsolatedMeasurement:
    """Wrap a fixed measurement plan with sample-local failure reporting."""

    def __init__(self, plan: TensorMeasurement, report: Callable[[Exception], None]) -> None:
        """Inject error handling into independent reducer adapters."""
        self._plan = TensorMeasurement(
            tuple(_GuardedScalar(reducer, report) for reducer in plan.reducers),
            tuple(_GuardedHistogram(reducer, report) for reducer in plan.histograms),
        )

    def __call__(self, tensor: torch.Tensor, *, sample_id: int) -> TensorRecord:
        """Collect all surviving scalar and histogram results."""
        return self._plan(tensor, sample_id=sample_id)
