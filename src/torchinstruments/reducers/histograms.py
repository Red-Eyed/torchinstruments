"""Configurable on-device histogram reduction for sampled tensor records."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, TypeAlias, TypeVar

import torch

from torchinstruments.records import HistogramRecord, JsonSetting


class HistogramRange(StrEnum):
    """Select a data-derived histogram range instead of fixed numeric bounds."""

    DYNAMIC = "dynamic"


HistogramValueRange: TypeAlias = tuple[float, float] | HistogramRange
_ResultValue = TypeVar("_ResultValue")


@dataclass(frozen=True)
class HistogramReductionResult:
    """Separate available histograms from reason-carrying unavailable results."""

    histograms: Mapping[str, HistogramRecord]
    unavailable_histograms: Mapping[str, str]


class HistogramReducer(Protocol):
    """Produce compact named histograms for one tensor and snapshot identifier."""

    def __call__(
        self,
        tensor: torch.Tensor,
        *,
        snapshot_id: int,
    ) -> HistogramReductionResult:
        """Reduce ``tensor`` when its independent snapshot policy is eligible."""
        ...


@dataclass(frozen=True)
class _ConfiguredHistogram:
    """Apply one validated histogram definition at a fixed snapshot cadence."""

    name: str
    bins: int
    value_range: HistogramValueRange
    every_n_snapshots: int

    def __call__(
        self,
        tensor: torch.Tensor,
        *,
        snapshot_id: int,
    ) -> HistogramReductionResult:
        """Return one histogram on eligible snapshots without copying raw values to CPU."""
        if snapshot_id % self.every_n_snapshots != 0:
            return HistogramReductionResult(histograms={}, unavailable_histograms={})

        try:
            record = _reduce_histogram(tensor.detach(), self.bins, self.value_range)
        except _HistogramUnavailable as error:
            return HistogramReductionResult(
                histograms={},
                unavailable_histograms={self.name: str(error)},
            )
        return HistogramReductionResult(
            histograms={self.name: record},
            unavailable_histograms={},
        )

    def reducer_type(self) -> str:
        """Identify the fixed-bin histogram reducer family."""
        return "histogram"

    def reducer_settings(self) -> Mapping[str, JsonSetting]:
        """Record the exact name, binning range, and independent cadence."""
        serialized_range: str | tuple[float, float]
        if isinstance(self.value_range, HistogramRange):
            serialized_range = self.value_range.value
        else:
            serialized_range = self.value_range
        return {
            "name": self.name,
            "bins": self.bins,
            "value_range": serialized_range,
            "every_n_snapshots": self.every_n_snapshots,
        }


class _HistogramUnavailable(RuntimeError):
    """Identify valid tensors for which no strict histogram can be represented."""


def histogram(
    *,
    name: str = "distribution",
    bins: int = 64,
    value_range: HistogramValueRange = HistogramRange.DYNAMIC,
    every_n_snapshots: int = 10,
) -> HistogramReducer:
    """Build an independently sampled finite-value histogram reducer.

    Snapshot zero is eligible, followed by every ``every_n_snapshots`` snapshot. A fixed
    ``(lower, upper)`` range makes bins comparable over time and records out-of-range values
    separately. ``HistogramRange.DYNAMIC`` covers each tensor's finite minimum and maximum.

    Args:
        name: Stable record name used in JSON and dashboard paths.
        bins: Number of regular bins, excluding underflow and overflow counts.
        value_range: Fixed finite bounds or a per-tensor dynamic range.
        every_n_snapshots: Independent cadence relative to sampled snapshot identifiers.

    Raises:
        ValueError: If the name, bin count, cadence, or fixed bounds are invalid.
    """
    normalized_name = name.strip()
    if not normalized_name:
        raise ValueError("histogram name must not be empty")
    if isinstance(bins, bool) or bins <= 0:
        raise ValueError("histogram bins must be greater than zero")
    if isinstance(every_n_snapshots, bool) or every_n_snapshots <= 0:
        raise ValueError("every_n_snapshots must be greater than zero")
    _validate_value_range(value_range)
    return _ConfiguredHistogram(
        name=normalized_name,
        bins=bins,
        value_range=value_range,
        every_n_snapshots=every_n_snapshots,
    )


def reduce_histograms(
    tensor: torch.Tensor,
    reducers: Sequence[HistogramReducer],
    *,
    snapshot_id: int,
) -> HistogramReductionResult:
    """Evaluate configured histogram reducers and reject duplicate record names."""
    histograms: dict[str, HistogramRecord] = {}
    unavailable: dict[str, str] = {}
    detached = tensor.detach()
    for reducer in reducers:
        result = reducer(detached, snapshot_id=snapshot_id)
        _merge_unique(histograms, result.histograms, "histogram")
        _merge_unique(unavailable, result.unavailable_histograms, "unavailable histogram")
        duplicate = histograms.keys() & unavailable.keys()
        if duplicate:
            name = sorted(duplicate)[0]
            raise ValueError(f"histogram {name!r} is both available and unavailable")
    return HistogramReductionResult(histograms=histograms, unavailable_histograms=unavailable)


def _validate_value_range(value_range: HistogramValueRange) -> None:
    """Reject ambiguous or non-finite fixed histogram bounds."""
    if isinstance(value_range, HistogramRange):
        return
    if len(value_range) != 2:
        raise ValueError("histogram value_range must contain lower and upper bounds")
    lower, upper = value_range
    if not math.isfinite(lower) or not math.isfinite(upper):
        raise ValueError("histogram value_range bounds must be finite")
    if lower >= upper:
        raise ValueError("histogram value_range lower bound must be less than upper bound")


def _reduce_histogram(
    tensor: torch.Tensor,
    bins: int,
    value_range: HistogramValueRange,
) -> HistogramRecord:
    """Compute one compact histogram and materialize it in a single CPU transfer."""
    values = _working_values(tensor)
    finite_mask = torch.isfinite(values)
    finite_values = values[finite_mask]
    if finite_values.numel() == 0:
        raise _HistogramUnavailable("tensor has no finite values")

    minimum = finite_values.min()
    maximum = finite_values.max()
    edges = _histogram_edges(minimum, maximum, bins, value_range)
    counts = torch.histogram(finite_values, bins=edges).hist
    lower = edges[0]
    upper = edges[-1]
    underflow = (finite_values < lower).sum()
    overflow = (finite_values > upper).sum()
    finite_count = finite_mask.sum()
    nonfinite_count = finite_mask.numel() - finite_count

    moments = finite_values.to(dtype=torch.float64)
    compact = torch.cat(
        (
            edges.to(dtype=torch.float64),
            counts.to(dtype=torch.float64),
            torch.stack(
                (
                    finite_count.to(dtype=torch.float64),
                    nonfinite_count.to(dtype=torch.float64),
                    underflow.to(dtype=torch.float64),
                    overflow.to(dtype=torch.float64),
                    minimum.to(dtype=torch.float64),
                    maximum.to(dtype=torch.float64),
                    moments.sum(),
                    moments.square().sum(),
                )
            ),
        )
    )
    materialized = compact.cpu().tolist()
    return _build_record(materialized, bins)


def _working_values(tensor: torch.Tensor) -> torch.Tensor:
    """Flatten supported tensors and promote low-precision values for histogram operations."""
    if tensor.layout != torch.strided:
        raise TypeError(f"unsupported tensor layout: {tensor.layout}")
    if tensor.is_complex():
        raise TypeError("complex tensors are not supported")
    values = tensor.reshape(-1)
    if values.dtype == torch.float64:
        return values
    return values.to(dtype=torch.float32)


def _histogram_edges(
    minimum: torch.Tensor,
    maximum: torch.Tensor,
    bins: int,
    value_range: HistogramValueRange,
) -> torch.Tensor:
    """Create fixed or data-derived edges on the tensor's device."""
    if not isinstance(value_range, HistogramRange):
        lower, upper = value_range
        return torch.linspace(
            lower,
            upper,
            bins + 1,
            device=minimum.device,
            dtype=minimum.dtype,
        )

    span = maximum - minimum
    scale = torch.maximum(torch.maximum(minimum.abs(), maximum.abs()), torch.ones_like(minimum))
    padding = scale * 0.01
    lower = torch.where(span > 0, minimum, minimum - padding)
    upper = torch.where(span > 0, maximum, maximum + padding)
    return torch.linspace(lower, upper, bins + 1, device=minimum.device, dtype=minimum.dtype)


def _build_record(materialized: list[float], bins: int) -> HistogramRecord:
    """Parse one compact transfer and enforce JSON-safe histogram aggregates."""
    edge_end = bins + 1
    count_end = edge_end + bins
    edges = tuple(materialized[:edge_end])
    counts = tuple(round(value) for value in materialized[edge_end:count_end])
    aggregates = materialized[count_end:]
    if not all(math.isfinite(value) for value in (*edges, *aggregates[4:])):
        raise _HistogramUnavailable("histogram aggregates are not finite")

    finite_count, nonfinite_count, underflow, overflow = (round(value) for value in aggregates[:4])
    if sum(counts) + underflow + overflow != finite_count:
        raise RuntimeError("histogram counts do not cover every finite tensor value")
    return HistogramRecord(
        bin_edges=edges,
        bin_counts=counts,
        finite_count=finite_count,
        nonfinite_count=nonfinite_count,
        underflow_count=underflow,
        overflow_count=overflow,
        minimum=aggregates[4],
        maximum=aggregates[5],
        sum=aggregates[6],
        sum_squares=aggregates[7],
    )


def _merge_unique(
    target: dict[str, _ResultValue],
    source: Mapping[str, _ResultValue],
    kind: str,
) -> None:
    """Merge dynamically named results while rejecting ambiguous duplicates."""
    for name, value in source.items():
        if name in target:
            raise ValueError(f"duplicate {kind}: {name}")
        target[name] = value
