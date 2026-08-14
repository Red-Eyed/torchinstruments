"""Public reducer protocols, factories, and materialization helpers."""

from torchinstruments.reducers.base import ReducedScalar, Reducer, ReductionResult
from torchinstruments.reducers.histograms import (
    HistogramRange,
    HistogramReducer,
    HistogramReductionResult,
    HistogramValueRange,
    histogram,
    reduce_histograms,
)
from torchinstruments.reducers.statistics import (
    combine,
    default_reducers,
    finite_fraction,
    max_abs,
    mean,
    reduce_tensor,
    rms,
    std,
)

__all__ = [
    "HistogramRange",
    "HistogramReducer",
    "HistogramReductionResult",
    "HistogramValueRange",
    "ReducedScalar",
    "Reducer",
    "ReductionResult",
    "combine",
    "default_reducers",
    "finite_fraction",
    "histogram",
    "max_abs",
    "mean",
    "reduce_histograms",
    "reduce_tensor",
    "rms",
    "std",
]
