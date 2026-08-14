"""Smoke tests for runnable public examples."""

from __future__ import annotations

from pathlib import Path

from examples.basic_training import run_demo
from tests.json_records import read_stats


def test_basic_training_writes_forward_and_backward_telemetry(tmp_path: Path) -> None:
    """Prove the basic example updates one live record across all iterations."""
    output_dir = tmp_path / "example-stats"

    run_demo(output_dir)

    stats = read_stats(output_dir / "stats.json")
    assert stats["samples_observed"] == 3
    assert stats["backward_samples_observed"] == 3
    assert set(stats["layers"]) == {"0", "1", "2"}
    assert stats["layers"]["2"][0]["outputs"]["output"]["shape"] == [16, 1]
    gradients = stats["layers"]["2"][0]["output_gradients"]["grad_output"]
    assert "rms" in gradients["latest_statistics"]
