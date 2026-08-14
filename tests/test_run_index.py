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
    assert "# TorchInstruments live run index" in index
    assert "- Sampled forwards observed: `0`" in index
    assert "- Correlated backwards observed: `0`" in index
    assert "- Invocation capture: `pytorch_hooks`" in index
    assert "stats.json" in index
    assert "No per-sample snapshot files are written" in index
    remove_observer(model)


def test_index_tracks_observed_scalars_histograms_and_backward(telemetry_dir: Path) -> None:
    """Update bounded field-name guidance without counting a backward rewrite twice."""
    model = nn.Identity()
    inject_observer(
        model,
        sampler=AlwaysSampler(),
        histograms=[
            histogram(bins=4, value_range=(-2.0, 2.0), every_n_samples=1),
        ],
        output_dir=telemetry_dir,
    )

    output = model(torch.tensor([-1.0, 0.0, 1.0], requires_grad=True))
    output.sum().backward()

    index = (telemetry_dir / "index.md").read_text(encoding="utf-8")
    assert "- Sampled forwards observed: `1`" in index
    assert "- Correlated backwards observed: `1`" in index
    assert "`finite_fraction`" in index
    assert "`cusum_change_score`" in index
    assert "does not observe loss" in index
    remove_observer(model)


def test_index_remains_bounded_as_live_samples_accumulate(telemetry_dir: Path) -> None:
    """Report run progress without appending one Markdown section per sample."""
    model = nn.Identity()
    inject_observer(model, sampler=AlwaysSampler(), output_dir=telemetry_dir)

    for _forward_index in range(20):
        model(torch.ones(1))

    index = (telemetry_dir / "index.md").read_text(encoding="utf-8")
    assert "- Sampled forwards observed: `20`" in index
    assert len(index) < 5_000
    remove_observer(model)
