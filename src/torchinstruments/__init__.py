"""Public API for passive PyTorch model telemetry."""

from importlib.metadata import version

from torchinstruments.aggregation import Aggregator, IndicatorConfig, LiveAggregator
from torchinstruments.api import has_observer, inject_observer, remove_observer
from torchinstruments.distributed import RankPolicy
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
from torchinstruments.reporting import ReportConfig
from torchinstruments.reporting.merge import merge_rank_reports
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
    "RankPolicy",
    "Reducer",
    "ReportConfig",
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
    "merge_rank_reports",
    "remove_observer",
    "rms",
    "std",
]
