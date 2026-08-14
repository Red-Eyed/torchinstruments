"""Persistence protocols and built-in telemetry sinks."""

from torchinstruments.sinks.base import Sink
from torchinstruments.sinks.composite import CompositeSink
from torchinstruments.sinks.directory import DirectorySink
from torchinstruments.sinks.logger import MetricLogger, MetricLoggerSink

__all__ = [
    "CompositeSink",
    "DirectorySink",
    "MetricLogger",
    "MetricLoggerSink",
    "Sink",
]
