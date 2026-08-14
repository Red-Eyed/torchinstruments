"""Fan out normalized telemetry records to multiple independent sinks."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from torchinstruments.records import ModuleRecord, RunRecord, SnapshotRecord
from torchinstruments.sinks.base import Sink


class CompositeSink:
    """Forward each lifecycle event to every configured sink.

    Snapshot writes and closes are attempted for every sink before collected failures are raised
    as an ``ExceptionGroup``. Initialization stops at the first failure and closes sinks that were
    already initialized.
    """

    def __init__(self, *sinks: Sink) -> None:
        """Require at least one sink and preserve the caller's delivery order."""
        if not sinks:
            raise ValueError("CompositeSink requires at least one sink")
        self._sinks = sinks

    def initialize(self, run: RunRecord, modules: Mapping[str, ModuleRecord]) -> None:
        """Initialize every sink or roll back those initialized before a failure."""
        initialized: list[Sink] = []
        try:
            for sink in self._sinks:
                sink.initialize(run, modules)
                initialized.append(sink)
        except Exception as initialize_error:
            rollback_errors = _close_sinks(reversed(initialized))
            if rollback_errors:
                raise ExceptionGroup(
                    "sink initialization and rollback failed",
                    [initialize_error, *rollback_errors],
                ) from initialize_error
            raise

    def write_snapshot(self, snapshot: SnapshotRecord) -> None:
        """Deliver a snapshot to every sink and report all delivery failures together."""
        errors: list[Exception] = []
        for sink in self._sinks:
            try:
                sink.write_snapshot(snapshot)
            except Exception as error:
                errors.append(error)
        _raise_errors("snapshot delivery", errors)

    def close(self) -> None:
        """Close every sink in reverse initialization order."""
        errors = _close_sinks(reversed(self._sinks))
        _raise_errors("sink close", errors)


def _close_sinks(sinks: Iterable[Sink]) -> list[Exception]:
    """Close all supplied sinks and return failures without interrupting cleanup."""
    errors: list[Exception] = []
    for sink in sinks:
        try:
            sink.close()
        except Exception as error:
            errors.append(error)
    return errors


def _raise_errors(operation: str, errors: list[Exception]) -> None:
    """Raise collected independent sink failures when an operation was not clean."""
    if errors:
        raise ExceptionGroup(f"{operation} failed for one or more sinks", errors)
