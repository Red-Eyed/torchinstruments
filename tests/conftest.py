"""Shared pytest fixtures for observer behavior tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from torch import nn

from torchinstruments import DirectorySink


@pytest.fixture
def linear_model() -> nn.Linear:
    """Provide a fresh computational leaf module for each test."""
    return nn.Linear(4, 3)


@pytest.fixture
def telemetry_dir(tmp_path: Path) -> Path:
    """Provide an isolated telemetry directory that does not yet exist."""
    return tmp_path / "stats"


@pytest.fixture
def detailed_sink(telemetry_dir: Path) -> DirectorySink:
    """Provide explicit exhaustive JSON only for tests that inspect every tensor path."""
    return DirectorySink(telemetry_dir, write_full_details=True)
