"""Persistence protocols and concrete telemetry destinations."""

from torchinstruments.sinks.base import Sink
from torchinstruments.sinks.composite import CompositeSink
from torchinstruments.sinks.directory import DirectorySink
from torchinstruments.sinks.tensorboard import HistogramWriter, TensorBoardLogger, TensorBoardSink

__all__ = [
    "CompositeSink",
    "DirectorySink",
    "HistogramWriter",
    "Sink",
    "TensorBoardLogger",
    "TensorBoardSink",
]
