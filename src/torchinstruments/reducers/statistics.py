"""Numerically safe built-in statistics and reducer composition."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, cast

import torch

from torchinstruments.records import Absent, JsonSetting
from torchinstruments.reducers.base import ReducedScalar, Reducer, ReductionResult
from torchinstruments.reducers.quantiles import exact_quartiles

_MetricName = Literal[
    "mean",
    "std",
    "rms",
    "minimum",
    "maximum",
    "max_abs",
    "finite_fraction",
    "nonfinite_fraction",
    "zero_fraction",
    "p25",
    "median",
    "p75",
]


@dataclass(frozen=True)
class _StatisticReducer:
    """Compute a selected set of built-in metrics in one tensor pass."""

    metrics: tuple[_MetricName, ...]

    def __call__(self, tensor: torch.Tensor) -> dict[str, ReducedScalar]:
        """Return only the configured built-in statistics for ``tensor``."""
        statistics = _statistics(tensor, self.metrics)
        return {name: statistics[name] for name in self.metrics}

    def reducer_type(self) -> str:
        """Identify the fused built-in scalar-statistics reducer."""
        return "statistics"

    def reducer_settings(self) -> dict[str, JsonSetting]:
        """Record the exact scalar metric names produced by this reducer."""
        return {"metrics": self.metrics}


@dataclass(frozen=True)
class _CompactStatistics:
    """Expose a small descriptive profile with consistent percentile names."""

    def __call__(self, tensor: torch.Tensor) -> dict[str, ReducedScalar]:
        """Preserve invalid-value prevalence while reducing finite tensor entries."""
        source = _statistics(
            tensor,
            (
                "mean",
                "std",
                "minimum",
                "maximum",
                "zero_fraction",
                "nonfinite_fraction",
            ),
        )
        names = {"minimum": "min", "maximum": "max"}
        result: dict[str, ReducedScalar] = {
            names.get(name, name): value for name, value in source.items()
        }
        return result

    def reducer_type(self) -> str:
        """Identify the default finite-value profile."""
        return "statistics"

    def reducer_settings(self) -> dict[str, JsonSetting]:
        """Describe the public metric names in this profile."""
        return {
            "metrics": (
                "mean",
                "std",
                "min",
                "max",
                "zero_fraction",
                "nonfinite_fraction",
            )
        }


def default_reducers() -> tuple[Reducer, ...]:
    """Return a compact distribution profile for every sampled tensor."""
    return (_CompactStatistics(), _QuartileStatistics())


@dataclass(frozen=True)
class _QuartileStatistics:
    """Keep quartile failures independent of moments and invalid-value counts."""

    def __call__(self, tensor: torch.Tensor) -> dict[str, ReducedScalar]:
        """Return exact finite-entry quartiles using public percentile names."""
        values = _statistics(tensor, ("p25", "median", "p75"))
        return {"p25": values["p25"], "p50": values["median"], "p75": values["p75"]}

    def reducer_type(self) -> str:
        """Identify the independently recoverable quartile reducer."""
        return "quartiles"

    def reducer_settings(self) -> dict[str, JsonSetting]:
        """Expose expected names even when quartiles cannot be measured."""
        return {"metrics": ("p25", "p50", "p75")}


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

    def combined(tensor: torch.Tensor) -> dict[str, ReducedScalar]:
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
    tensor: torch.Tensor, requested: tuple[_MetricName, ...]
) -> dict[str, torch.Tensor]:
    """Measure finite entries while retaining invalid and zero prevalence over the full tensor."""
    values = _working_values(tensor)
    unavailable = torch.full((), float("nan"), device=values.device, dtype=values.dtype)
    if values.numel() == 0:
        return {name: unavailable for name in requested}
    finite_mask = torch.isfinite(values)
    statistics: dict[str, torch.Tensor] = {}
    if "finite_fraction" in requested:
        statistics["finite_fraction"] = finite_mask.to(values.dtype).mean()
    if "nonfinite_fraction" in requested:
        statistics["nonfinite_fraction"] = (~finite_mask).sum() / values.numel()
    if "zero_fraction" in requested:
        statistics["zero_fraction"] = (finite_mask & (values == 0)).to(values.dtype).mean()
    numerical = tuple(
        name
        for name in requested
        if name not in {"finite_fraction", "nonfinite_fraction", "zero_fraction"}
    )
    if not numerical:
        return statistics
    finite = values[finite_mask]
    if finite.numel() == 0:
        statistics.update((name, unavailable) for name in numerical)
        return statistics
    statistics.update(_finite_statistics(finite, numerical))
    return statistics


def _finite_statistics(
    values: torch.Tensor, requested: tuple[_MetricName, ...]
) -> dict[str, torch.Tensor]:
    """Compute only requested distribution measurements on a nonempty finite tensor."""
    result: dict[str, torch.Tensor] = {}
    for name in requested:
        match name:
            case "mean":
                result[name] = values.mean()
            case "std":
                result[name] = values.std(correction=0)
            case "rms":
                result[name] = values.square().mean().sqrt()
            case "minimum":
                result[name] = values.min()
            case "maximum":
                result[name] = values.max()
            case "max_abs":
                result[name] = values.abs().max()
            case "p25" | "median" | "p75":
                continue
            case _:
                raise ValueError(f"unsupported finite statistic: {name}")
    result.update(_quantiles(values, requested))
    return result


def _quantiles(values: torch.Tensor, requested: tuple[_MetricName, ...]) -> dict[str, torch.Tensor]:
    """Fuse the requested quartiles into one device-local operation."""
    probabilities = {"p25": 0.25, "median": 0.5, "p75": 0.75}
    names = [name for name in probabilities if name in requested]
    if not names:
        return {}
    if values.numel() > 2**24:
        quartiles = exact_quartiles(values)
        return {name: quartiles[name] for name in names}
    levels = torch.tensor(
        [probabilities[name] for name in names], device=values.device, dtype=values.dtype
    )
    quantiles = torch.quantile(values, levels)
    return dict(zip(names, quantiles, strict=True))


def _working_values(tensor: torch.Tensor) -> torch.Tensor:
    """Detach layout concerns and promote low-precision values for safe reductions."""
    if tensor.layout != torch.strided:
        raise TypeError(f"unsupported tensor layout: {tensor.layout}")
    if tensor.is_complex():
        raise TypeError("complex tensors are not supported")
    if tensor.dtype == torch.float64:
        return tensor
    return tensor.to(dtype=torch.float32)


def _materialize_scalars(values: dict[str, ReducedScalar]) -> ReductionResult:
    """Convert compact scalar tensors to Python floats with one transfer per device."""
    stats: dict[str, float] = {}
    unavailable_stats: dict[str, str] = {}
    tensor_groups: dict[torch.device, list[tuple[str, torch.Tensor]]] = {}

    for name, value in values.items():
        if isinstance(value, Absent):
            unavailable_stats[name] = value.reason
            continue
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
