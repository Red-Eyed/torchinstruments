"""Public API for passive PyTorch model telemetry."""

from importlib.metadata import version

from torchinstruments.api import has_observer, inject_observer, remove_observer
from torchinstruments.errors import ErrorPolicy, ObserverAlreadyAttachedError
from torchinstruments.reducers import (
    Reducer,
    combine,
    default_reducers,
    finite_fraction,
    max_abs,
    mean,
    rms,
    std,
)
from torchinstruments.sampling import AlwaysSampler, EveryNForwardsSampler, TimedSampler
from torchinstruments.selectors import leaf_modules
from torchinstruments.sinks import DirectorySink, Sink

__version__ = version("torchinstruments")

__all__ = [
    "AlwaysSampler",
    "DirectorySink",
    "ErrorPolicy",
    "EveryNForwardsSampler",
    "ObserverAlreadyAttachedError",
    "Reducer",
    "Sink",
    "TimedSampler",
    "__version__",
    "combine",
    "default_reducers",
    "finite_fraction",
    "has_observer",
    "inject_observer",
    "leaf_modules",
    "max_abs",
    "mean",
    "remove_observer",
    "rms",
    "std",
]
