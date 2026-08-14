"""Public reducer protocols, factories, and materialization helpers."""

from torchinstruments.reducers.base import ReducedScalar, Reducer, ReductionResult
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
    "ReducedScalar",
    "Reducer",
    "ReductionResult",
    "combine",
    "default_reducers",
    "finite_fraction",
    "max_abs",
    "mean",
    "reduce_tensor",
    "rms",
    "std",
]
