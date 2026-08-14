"""Smoke tests for runnable public examples."""

from __future__ import annotations

from pathlib import Path

from examples.basic_training import run_demo
from tests.json_records import read_snapshot


def test_basic_training_writes_forward_and_backward_telemetry(tmp_path: Path) -> None:
    """Prove the basic example produces one complete snapshot per iteration."""
    output_dir = tmp_path / "example-stats"

    run_demo(output_dir)

    snapshot_paths = sorted((output_dir / "snapshots").glob("*.json"))
    assert [path.name for path in snapshot_paths] == [
        "000000.json",
        "000001.json",
        "000002.json",
    ]

    snapshot = read_snapshot(snapshot_paths[-1])
    assert snapshot["state"] == "backward_observed"
    assert set(snapshot["modules"]) == {"0", "1", "2"}
    assert snapshot["modules"]["2"][0]["outputs"]["output"]["shape"] == [16, 1]
    assert "rms" in snapshot["modules"]["2"][0]["output_gradients"]["grad_output"]["stats"]
