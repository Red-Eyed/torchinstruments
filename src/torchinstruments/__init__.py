"""Public API for passive PyTorch model telemetry."""

from importlib.metadata import version

from torchinstruments.aggregation import Aggregator, IndicatorConfig, LiveAggregator
from torchinstruments.api import has_observer, inject_observer, remove_observer
from torchinstruments.errors import ErrorPolicy, ObserverAlreadyAttachedError
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
    MetricLogger,
    MetricLoggerSink,
    Sink,
    TensorBoardLogger,
    TensorBoardSink,
)

__version__ = version("torchinstruments")

__all__ = [
    "Aggregator",
    "AlwaysSampler",
    "CompositeSink",
    "DirectorySink",
    "ErrorPolicy",
    "EveryNForwardsSampler",
    "HistogramRange",
    "HistogramReducer",
    "HistogramValueRange",
    "IndicatorConfig",
    "LiveAggregator",
    "MetricLogger",
    "MetricLoggerSink",
    "ObserverAlreadyAttachedError",
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
