"""Numerically safe built-in statistics and reducer composition."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, cast

import torch

from torchinstruments.reducers.base import ReducedScalar, Reducer, ReductionResult

_MetricName = Literal["mean", "std", "rms", "max_abs", "finite_fraction"]
_ALL_METRICS: tuple[_MetricName, ...] = (
    "mean",
    "std",
    "rms",
    "max_abs",
    "finite_fraction",
)


@dataclass(frozen=True)
class _StatisticReducer:
    """Compute a selected set of built-in metrics in one tensor pass."""

    metrics: tuple[_MetricName, ...]

    def __call__(self, tensor: torch.Tensor) -> Mapping[str, ReducedScalar]:
        """Return only the configured built-in statistics for ``tensor``."""
        statistics = _statistics(tensor, self.metrics)
        return {name: statistics[name] for name in self.metrics}


def default_reducers() -> tuple[Reducer, ...]:
    """Return the inexpensive diagnostic statistics enabled for every sample."""
    return (_StatisticReducer(_ALL_METRICS),)


def mean() -> Reducer:
    """Build a reducer for the mean of finite tensor values."""
    return _StatisticReducer(("mean",))


def std() -> Reducer:
    """Build a reducer for population standard deviation over finite values."""
    return _StatisticReducer(("std",))


def rms() -> Reducer:
    """Build a reducer for root-mean-square magnitude over finite values."""
    return _StatisticReducer(("rms",))


def max_abs() -> Reducer:
    """Build a reducer for maximum absolute finite magnitude."""
    return _StatisticReducer(("max_abs",))


def finite_fraction() -> Reducer:
    """Build a reducer for the fraction of original tensor values that are finite."""
    return _StatisticReducer(("finite_fraction",))


def combine(*reducers: Reducer) -> Reducer:
    """Compose reducers while rejecting duplicate metric names.

    Built-in statistics are fused so common tensor preparation and masking happen once.
    """
    if all(isinstance(reducer, _StatisticReducer) for reducer in reducers):
        statistic_reducers = cast(tuple[_StatisticReducer, ...], reducers)
        metrics = tuple(metric for reducer in statistic_reducers for metric in reducer.metrics)
        if len(metrics) != len(set(metrics)):
            raise ValueError("combined reducers contain duplicate metrics")
        return _StatisticReducer(metrics)

    def combined(tensor: torch.Tensor) -> Mapping[str, ReducedScalar]:
        """Evaluate arbitrary child reducers and merge their scalar mappings."""
        combined_values: dict[str, ReducedScalar] = {}
        for reducer in reducers:
            for name, value in reducer(tensor).items():
                if name in combined_values:
                    raise ValueError(f"duplicate reducer metric: {name}")
                combined_values[name] = value
        return combined_values

    return combined


def reduce_tensor(tensor: torch.Tensor, reducers: Sequence[Reducer]) -> ReductionResult:
    """Detach and reduce a tensor, batching scalar device-to-host transfers by device."""
    detached = tensor.detach()
    reduced: dict[str, ReducedScalar] = {}

    for reducer in reducers:
        for name, value in reducer(detached).items():
            if name in reduced:
                raise ValueError(f"duplicate reducer metric: {name}")
            reduced[name] = value

    return _materialize_scalars(reduced)


def _statistics(
    tensor: torch.Tensor,
    requested: tuple[_MetricName, ...],
) -> Mapping[str, ReducedScalar]:
    """Compute requested statistics on finite values in a numerically safe dtype."""
    values = _working_values(tensor)
    if values.numel() == 0:
        unavailable = torch.full((), float("nan"), device=values.device, dtype=values.dtype)
        return {name: unavailable for name in requested}

    finite_mask = torch.isfinite(values)
    statistics: dict[str, torch.Tensor] = {}
    if "finite_fraction" in requested:
        statistics["finite_fraction"] = finite_mask.to(dtype=values.dtype).mean()

    numerical_metrics = tuple(name for name in requested if name != "finite_fraction")
    if not numerical_metrics:
        return statistics

    finite_count = finite_mask.sum()
    denominator = finite_count.clamp_min(1)
    zeros = torch.zeros((), device=values.device, dtype=values.dtype)
    finite_values = torch.where(finite_mask, values, zeros)
    unavailable = torch.full((), float("nan"), device=values.device, dtype=values.dtype)
    has_finite_value = finite_count > 0

    mean_value = finite_values.sum() / denominator
    if "mean" in numerical_metrics:
        statistics["mean"] = torch.where(has_finite_value, mean_value, unavailable)
    if "std" in numerical_metrics:
        centered = torch.where(finite_mask, values - mean_value, zeros)
        variance = centered.square().sum() / denominator
        statistics["std"] = torch.where(has_finite_value, torch.sqrt(variance), unavailable)
    if "rms" in numerical_metrics:
        rms_value = torch.sqrt(finite_values.square().sum() / denominator)
        statistics["rms"] = torch.where(has_finite_value, rms_value, unavailable)
    if "max_abs" in numerical_metrics:
        negative_infinity = torch.full((), float("-inf"), device=values.device, dtype=values.dtype)
        maximum = torch.where(finite_mask, values.abs(), negative_infinity).max()
        statistics["max_abs"] = torch.where(has_finite_value, maximum, unavailable)

    return statistics


def _working_values(tensor: torch.Tensor) -> torch.Tensor:
    """Detach layout concerns and promote low-precision values for safe reductions."""
    if tensor.layout != torch.strided:
        raise TypeError(f"unsupported tensor layout: {tensor.layout}")
    if tensor.is_complex():
        raise TypeError("complex tensors are not supported")
    if tensor.dtype == torch.float64:
        return tensor
    return tensor.to(dtype=torch.float32)


def _materialize_scalars(values: Mapping[str, ReducedScalar]) -> ReductionResult:
    """Convert compact scalar tensors to Python floats with one transfer per device."""
    stats: dict[str, float] = {}
    unavailable_stats: dict[str, str] = {}
    tensor_groups: dict[torch.device, list[tuple[str, torch.Tensor]]] = {}

    for name, value in values.items():
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError(f"reducer metric {name!r} must be scalar")
            tensor_groups.setdefault(value.device, []).append((name, value.reshape(())))
            continue
        _store_scalar(name, float(value), stats, unavailable_stats)

    for group in tensor_groups.values():
        names = [name for name, _value in group]
        target_dtype = group[0][1].dtype
        for _name, value in group[1:]:
            target_dtype = torch.promote_types(target_dtype, value.dtype)
        compact = torch.stack([value.to(dtype=target_dtype) for _name, value in group])
        materialized = compact.cpu().tolist()
        for name, value in zip(names, materialized, strict=True):
            _store_scalar(name, float(value), stats, unavailable_stats)

    return ReductionResult(stats=stats, unavailable_stats=unavailable_stats)


def _store_scalar(
    name: str,
    value: float,
    stats: dict[str, float],
    unavailable_stats: dict[str, str],
) -> None:
    """Store a finite metric or preserve why its value is unavailable."""
    if math.isfinite(value):
        stats[name] = value
        return
    unavailable_stats[name] = "reducer produced a non-finite scalar"
