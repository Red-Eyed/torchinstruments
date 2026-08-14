"""Shared pytest fixtures for observer behavior tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from torch import nn


@pytest.fixture
def linear_model() -> nn.Linear:
    """Provide a fresh computational leaf module for each test."""
    return nn.Linear(4, 3)


@pytest.fixture
def telemetry_dir(tmp_path: Path) -> Path:
    """Provide an isolated telemetry directory that does not yet exist."""
    return tmp_path / "stats"
