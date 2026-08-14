"""Persistence protocols and the built-in directory sink."""

from torchinstruments.sinks.base import Sink
from torchinstruments.sinks.directory import DirectorySink

__all__ = ["DirectorySink", "Sink"]
