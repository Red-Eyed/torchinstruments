"""Exceptions and collection error policies exposed by TorchInstruments."""

from __future__ import annotations

from enum import StrEnum


class TorchInstrumentsError(Exception):
    """Base exception for observer configuration and lifecycle failures."""


class ObserverAlreadyAttachedError(TorchInstrumentsError):
    """Raised when a model already has a TorchInstruments observer."""


class SinkAlreadyInitializedError(TorchInstrumentsError):
    """Raised when a directory already contains telemetry for another run."""


class ErrorPolicy(StrEnum):
    """Control whether instrumentation failures raise, warn, or remain silent."""

    RAISE = "raise"
    WARN = "warn"
    IGNORE = "ignore"


def parse_error_policy(value: ErrorPolicy | str) -> ErrorPolicy:
    """Parse a public error-policy value and report the valid choices on failure."""
    try:
        return ErrorPolicy(value)
    except ValueError as error:
        choices = ", ".join(policy.value for policy in ErrorPolicy)
        raise ValueError(f"error_policy must be one of: {choices}") from error
