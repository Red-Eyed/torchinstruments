"""Isolate reducer failures while retaining the observer's error policy."""

from collections.abc import Callable, Iterable
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
        values: dict[str, ReducedScalar] = {}
        try:
            values = self.reducer(tensor)
            return _validate_scalars(values)
        except Exception as error:
            self.report(error)
            reason = f"{type(error).__name__}: {error}"
            match values:
                case dict():
                    names = _string_names(values)
                case _:
                    names = ()
            names = names or _metric_names(self.reducer)
            return {name: Absent(reason) for name in names}


def _metric_names(reducer: Reducer) -> tuple[str, ...]:
    """Read known reducer output names without inventing identities for custom callables."""
    match reducer:
        case DescribedReducer():
            names = reducer.reducer_settings().get("metrics", ())
        case _:
            return ()
    match names:
        case tuple():
            return _string_names(names)
        case _:
            return ()


def _string_names(names: Iterable[object]) -> tuple[str, ...]:
    """Keep valid metric identities from potentially malformed custom reducer output."""
    valid: list[str] = []
    for name in names:
        match name:
            case str():
                valid.append(name)
    return tuple(valid)


def _validate_scalars(values: dict[str, ReducedScalar]) -> dict[str, ReducedScalar]:
    """Normalize host scalars and reject unsupported tensors before batched transfers."""
    validated: dict[str, ReducedScalar] = {}
    for name, value in values.items():
        match name:
            case str():
                validated[name] = _validate_scalar(name, value)
            case _:
                raise TypeError("reducer metric names must be strings")
    return validated


def _validate_scalar(name: str, value: ReducedScalar) -> ReducedScalar:
    """Keep conversion failures inside the owning reducer's error boundary."""
    match value:
        case torch.Tensor():
            if value.numel() != 1:
                raise ValueError(f"reducer metric {name!r} must be scalar")
            if (
                value.is_complex()
                or value.layout != torch.strided
                or value.is_quantized
                or value.device.type == "meta"
            ):
                raise TypeError(f"reducer metric {name!r} must be a materializable real tensor")
            return value.detach()
        case float() | int():
            return float(value)
        case Absent():
            return value
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
