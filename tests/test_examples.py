"""Smoke tests for runnable public examples."""

from __future__ import annotations

from pathlib import Path

from examples.basic_training import run_demo
from tests.json_records import read_report


def test_basic_training_writes_forward_and_backward_telemetry(tmp_path: Path) -> None:
    """Prove the basic example updates one live record across all iterations."""
    output_dir = tmp_path / "example-stats"

    run_demo(output_dir)

    report = read_report(output_dir / "report.json")
    assert report["coverage"]["samples_observed"] == 3
    assert report["coverage"]["backward_samples_observed"] == 3
    assert report["coverage"]["selected_modules"] == 3
    assert report["coverage"]["tensor_paths"] == 6
    assert sorted(path.name for path in output_dir.iterdir()) == ["index.md", "report.json"]
