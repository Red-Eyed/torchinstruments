"""Passive per-layer PyTorch history with Polars summaries and TensorBoard histograms."""

from importlib.metadata import version

from torchinstruments.api import has_observer, inject_observer, remove_observer
from torchinstruments.distributed import RankPolicy
from torchinstruments.errors import ErrorPolicy, ObserverAlreadyAttachedError
from torchinstruments.history.summary import HistoryConfig
from torchinstruments.reducers import (
    HistogramRange,
    HistogramReducer,
    HistogramValueRange,
    Reducer,
    combine,
    default_reducers,
    finite_fraction,
    histogram,
    max_abs,
    mean,
    rms,
    std,
)
from torchinstruments.sampling import AlwaysSampler, EveryNForwardsSampler, TimedSampler
from torchinstruments.selectors import leaf_modules
from torchinstruments.sinks import (
    CompositeSink,
    DirectorySink,
    Sink,
    TensorBoardLogger,
    TensorBoardSink,
)

__version__ = version("torchinstruments")

__all__ = [
    "AlwaysSampler",
    "CompositeSink",
    "DirectorySink",
    "ErrorPolicy",
    "EveryNForwardsSampler",
    "HistogramRange",
    "HistogramReducer",
    "HistogramValueRange",
    "HistoryConfig",
    "ObserverAlreadyAttachedError",
    "RankPolicy",
    "Reducer",
    "Sink",
    "TensorBoardLogger",
    "TensorBoardSink",
    "TimedSampler",
    "__version__",
    "combine",
    "default_reducers",
    "finite_fraction",
    "has_observer",
    "histogram",
    "inject_observer",
    "leaf_modules",
    "max_abs",
    "mean",
    "remove_observer",
    "rms",
    "std",
]
