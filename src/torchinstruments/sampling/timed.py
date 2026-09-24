"""Monotonic deadlines for root forwards and independently invoked children."""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import timedelta
from threading import Lock

from torchinstruments.records import JsonScalar
from torchinstruments.sampling.base import SamplingEvent


class TimedSampler:
    """Keep a separate interval deadline for each sampling scope."""

    def __init__(
        self,
        interval: timedelta = timedelta(minutes=1),
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Start a positive interval using an injectable monotonic clock."""
        interval_seconds = interval.total_seconds()
        if interval_seconds <= 0:
            raise ValueError("interval must be greater than zero")

        self._interval_seconds = interval_seconds
        self._next_due = {"": clock()}
        self._lock = Lock()

    def should_sample(self, event: SamplingEvent) -> bool:
        """Select an event at or after the deadline and schedule the next deadline."""
        with self._lock:
            deadline = self._next_due.get(event.module_name, event.monotonic_time)
            if event.monotonic_time < deadline:
                return False
            self._next_due[event.module_name] = event.monotonic_time + self._interval_seconds
            return True

    def sampling_type(self) -> str:
        """Return the stable run-metadata name for this policy."""
        return "timed"

    def sampling_settings(self) -> dict[str, JsonScalar]:
        """Return the sampling interval in unambiguous seconds."""
        return {"interval_seconds": self._interval_seconds}
