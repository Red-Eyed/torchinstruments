"""Tests for the generated human and LLM telemetry run index."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from torchinstruments import AlwaysSampler, histogram, inject_observer, remove_observer


def test_index_exists_before_the_first_sample(telemetry_dir: Path) -> None:
    """Create an immediately useful file guide when the directory sink initializes."""
    model = nn.Identity()

    inject_observer(model, sampler=AlwaysSampler(), output_dir=telemetry_dir)

    index = (telemetry_dir / "index.md").read_text(encoding="utf-8")
    assert "# TorchInstruments run index" in index
    assert "- Snapshots written: `0`" in index
    assert "- Latest snapshot ID: `not available yet`" in index
    assert "- Configured signals: `module_outputs`, `module_output_gradients`" in index
    assert "`statistics` with settings" in index
    assert "- Histogram reducers: none configured" in index
    assert "run.json" in index
    assert "modules.json" in index
    assert "representative files under snapshots/" in index
    remove_observer(model)


def test_index_tracks_observed_scalars_histograms_and_backward(telemetry_dir: Path) -> None:
    """Update bounded field-name guidance without counting a backward rewrite twice."""
    model = nn.Identity()
    inject_observer(
        model,
        sampler=AlwaysSampler(),
        histograms=[
            histogram(bins=4, value_range=(-2.0, 2.0), every_n_snapshots=1),
        ],
        output_dir=telemetry_dir,
    )

    output = model(torch.tensor([-1.0, 0.0, 1.0], requires_grad=True))
    output.sum().backward()

    index = (telemetry_dir / "index.md").read_text(encoding="utf-8")
    assert "- Snapshots written: `1`" in index
    assert "- Latest snapshot ID: `0`" in index
    assert "- Signals: `module output gradients`, `module outputs`" in index
    assert "`finite_fraction`" in index
    assert "- Histogram records: `distribution`" in index
    assert '"every_n_snapshots": 1' in index
    assert '"value_range": [-2.0, 2.0]' in index
    assert "everything projected to TensorBoard" in index
    assert "does not observe loss values" in index
    remove_observer(model)


def test_index_remains_bounded_as_snapshots_accumulate(telemetry_dir: Path) -> None:
    """Report run progress without appending one Markdown section per snapshot."""
    model = nn.Identity()
    inject_observer(model, sampler=AlwaysSampler(), output_dir=telemetry_dir)

    for _forward_index in range(20):
        model(torch.ones(1))

    index = (telemetry_dir / "index.md").read_text(encoding="utf-8")
    assert "- Snapshots written: `20`" in index
    assert "- Latest snapshot ID: `19`" in index
    assert len(index) < 5_000
    remove_observer(model)
