"""Numerically safe built-in statistics and reducer composition."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, cast

import torch

from torchinstruments.records import JsonSetting
from torchinstruments.reducers.base import ReducedScalar, Reducer, ReductionResult

_MetricName = Literal[
    "mean",
    "std",
    "rms",
    "minimum",
    "maximum",
    "mean_abs",
    "l1_norm",
    "l2_norm",
    "max_abs",
    "finite_fraction",
    "zero_fraction",
    "negative_fraction",
    "positive_fraction",
    "skewness",
    "excess_kurtosis",
    "p01",
    "p05",
    "p25",
    "median",
    "p75",
    "p95",
    "p99",
    "p999",
    "p99_abs",
    "p999_abs",
    "interquartile_range",
    "central_98_range",
    "max_to_rms",
    "p99_abs_to_rms",
    "p999_abs_to_rms",
    "tail_fraction_beyond_3_std",
    "normalized_magnitude_entropy",
    "effective_magnitude_support_fraction",
]
_ALL_METRICS: tuple[_MetricName, ...] = (
    "mean",
    "std",
    "rms",
    "minimum",
    "maximum",
    "mean_abs",
    "l1_norm",
    "l2_norm",
    "max_abs",
    "finite_fraction",
    "zero_fraction",
    "negative_fraction",
    "positive_fraction",
    "skewness",
    "excess_kurtosis",
    "p01",
    "p05",
    "p25",
    "median",
    "p75",
    "p95",
    "p99",
    "p999",
    "p99_abs",
    "p999_abs",
    "interquartile_range",
    "central_98_range",
    "max_to_rms",
    "p99_abs_to_rms",
    "p999_abs_to_rms",
    "tail_fraction_beyond_3_std",
    "normalized_magnitude_entropy",
    "effective_magnitude_support_fraction",
)


@dataclass(frozen=True)
class _StatisticReducer:
    """Compute a selected set of built-in metrics in one tensor pass."""

    metrics: tuple[_MetricName, ...]

    def __call__(self, tensor: torch.Tensor) -> Mapping[str, ReducedScalar]:
        """Return only the configured built-in statistics for ``tensor``."""
        statistics = _statistics(tensor, self.metrics)
        return {name: statistics[name] for name in self.metrics}

    def reducer_type(self) -> str:
        """Identify the fused built-in scalar-statistics reducer."""
        return "statistics"

    def reducer_settings(self) -> Mapping[str, JsonSetting]:
        """Record the exact scalar metric names produced by this reducer."""
        return {"metrics": self.metrics}


def default_reducers() -> tuple[Reducer, ...]:
    """Return the rich point-in-time distribution profile enabled for every sample."""
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
    if "zero_fraction" in requested:
        statistics["zero_fraction"] = (finite_mask & (values == 0)).to(values.dtype).mean()
    if "negative_fraction" in requested:
        statistics["negative_fraction"] = (finite_mask & (values < 0)).to(values.dtype).mean()
    if "positive_fraction" in requested:
        statistics["positive_fraction"] = (finite_mask & (values > 0)).to(values.dtype).mean()

    fraction_metrics = {
        "finite_fraction",
        "zero_fraction",
        "negative_fraction",
        "positive_fraction",
    }
    numerical_metrics = tuple(name for name in requested if name not in fraction_metrics)
    if not numerical_metrics:
        return statistics

    finite_count = finite_mask.sum()
    denominator = finite_count.clamp_min(1)
    zeros = torch.zeros((), device=values.device, dtype=values.dtype)
    masked_values = torch.where(finite_mask, values, zeros)
    finite_values = values[finite_mask]
    unavailable = torch.full((), float("nan"), device=values.device, dtype=values.dtype)
    if finite_values.numel() == 0:
        for name in numerical_metrics:
            statistics[name] = unavailable
        return statistics
    has_finite_value = finite_count > 0

    sum_value = masked_values.sum()
    sum_abs = masked_values.abs().sum()
    sum_squares = masked_values.square().sum()
    mean_value = sum_value / denominator
    mean_abs_value = sum_abs / denominator
    rms_value = torch.sqrt(sum_squares / denominator)
    negative_infinity = torch.full((), float("-inf"), device=values.device, dtype=values.dtype)
    positive_infinity = torch.full((), float("inf"), device=values.device, dtype=values.dtype)
    minimum = torch.where(finite_mask, values, positive_infinity).min()
    maximum = torch.where(finite_mask, values, negative_infinity).max()
    maximum_absolute = torch.where(finite_mask, values.abs(), negative_infinity).max()

    if "mean" in numerical_metrics:
        statistics["mean"] = torch.where(has_finite_value, mean_value, unavailable)
    if "minimum" in numerical_metrics:
        statistics["minimum"] = torch.where(has_finite_value, minimum, unavailable)
    if "maximum" in numerical_metrics:
        statistics["maximum"] = torch.where(has_finite_value, maximum, unavailable)
    if "mean_abs" in numerical_metrics:
        statistics["mean_abs"] = torch.where(has_finite_value, mean_abs_value, unavailable)
    if "l1_norm" in numerical_metrics:
        statistics["l1_norm"] = torch.where(has_finite_value, sum_abs, unavailable)
    if "l2_norm" in numerical_metrics:
        statistics["l2_norm"] = torch.where(has_finite_value, torch.sqrt(sum_squares), unavailable)
    if "rms" in numerical_metrics:
        statistics["rms"] = torch.where(has_finite_value, rms_value, unavailable)
    if "max_abs" in numerical_metrics:
        statistics["max_abs"] = torch.where(has_finite_value, maximum_absolute, unavailable)

    centered = torch.where(finite_mask, values - mean_value, zeros)
    variance = centered.square().sum() / denominator
    standard_deviation = torch.sqrt(variance)
    if "std" in numerical_metrics:
        statistics["std"] = torch.where(has_finite_value, torch.sqrt(variance), unavailable)

    has_scale = variance > 0
    if "skewness" in numerical_metrics:
        third_moment = centered.pow(3).sum() / denominator
        skewness = third_moment / standard_deviation.pow(3)
        statistics["skewness"] = torch.where(has_scale, skewness, unavailable)
    if "excess_kurtosis" in numerical_metrics:
        fourth_moment = centered.pow(4).sum() / denominator
        excess_kurtosis = fourth_moment / variance.square() - 3
        statistics["excess_kurtosis"] = torch.where(has_scale, excess_kurtosis, unavailable)
    if "tail_fraction_beyond_3_std" in numerical_metrics:
        tail_fraction = (centered.abs() > 3 * standard_deviation).to(values.dtype).sum()
        tail_fraction = tail_fraction / denominator
        statistics["tail_fraction_beyond_3_std"] = torch.where(
            has_scale, tail_fraction, unavailable
        )

    quantile_names = ("p01", "p05", "p25", "median", "p75", "p95", "p99", "p999")
    needs_quantiles = any(name in numerical_metrics for name in quantile_names) or any(
        name in numerical_metrics for name in ("interquartile_range", "central_98_range")
    )
    if needs_quantiles:
        probabilities = torch.tensor(
            [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999],
            device=values.device,
            dtype=values.dtype,
        )
        quantiles = torch.quantile(finite_values, probabilities)
        quantile_values = dict(zip(quantile_names, quantiles, strict=True))
        for name in quantile_names:
            if name in numerical_metrics:
                statistics[name] = quantile_values[name]
        if "interquartile_range" in numerical_metrics:
            statistics["interquartile_range"] = quantile_values["p75"] - quantile_values["p25"]
        if "central_98_range" in numerical_metrics:
            statistics["central_98_range"] = quantile_values["p99"] - quantile_values["p01"]

    absolute_quantile_names = ("p99_abs", "p999_abs")
    needs_absolute_quantiles = any(
        name in numerical_metrics
        for name in (*absolute_quantile_names, "p99_abs_to_rms", "p999_abs_to_rms")
    )
    absolute_quantiles: dict[str, torch.Tensor] = {}
    if needs_absolute_quantiles:
        probabilities = torch.tensor([0.99, 0.999], device=values.device, dtype=values.dtype)
        values_at_quantiles = torch.quantile(finite_values.abs(), probabilities)
        absolute_quantiles = dict(zip(absolute_quantile_names, values_at_quantiles, strict=True))
        for name in absolute_quantile_names:
            if name in numerical_metrics:
                statistics[name] = absolute_quantiles[name]

    if "max_to_rms" in numerical_metrics:
        statistics["max_to_rms"] = torch.where(
            rms_value > 0, maximum_absolute / rms_value, unavailable
        )
    if "p99_abs_to_rms" in numerical_metrics:
        statistics["p99_abs_to_rms"] = torch.where(
            rms_value > 0, absolute_quantiles["p99_abs"] / rms_value, unavailable
        )
    if "p999_abs_to_rms" in numerical_metrics:
        statistics["p999_abs_to_rms"] = torch.where(
            rms_value > 0, absolute_quantiles["p999_abs"] / rms_value, unavailable
        )

    entropy_metrics = {
        "normalized_magnitude_entropy",
        "effective_magnitude_support_fraction",
    }
    if entropy_metrics & set(numerical_metrics):
        probabilities = finite_values.abs() / sum_abs
        log_probabilities = probabilities.clamp_min(torch.finfo(values.dtype).tiny).log()
        entropy = -(probabilities * log_probabilities).sum()
        normalized_entropy = entropy / math.log(max(finite_values.numel(), 2))
        has_magnitude = sum_abs > 0
        if "normalized_magnitude_entropy" in numerical_metrics:
            statistics["normalized_magnitude_entropy"] = torch.where(
                has_magnitude, normalized_entropy, unavailable
            )
        if "effective_magnitude_support_fraction" in numerical_metrics:
            effective_support = entropy.exp() / denominator
            statistics["effective_magnitude_support_fraction"] = torch.where(
                has_magnitude, effective_support, unavailable
            )

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
