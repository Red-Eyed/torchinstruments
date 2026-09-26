"""Specify supported checkpoint capture and visible reentrant coverage limitations."""

import copy
from pathlib import Path

import polars as pl
import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from typing_extensions import override

from torchinstruments import AlwaysSampler, inject_observer, remove_observer


class Checkpointed(nn.Module):
    """Reuse a checkpointed block across multiple outstanding forwards."""

    def __init__(self, reentrant: bool) -> None:
        """Keep the checkpoint mode explicit."""
        super().__init__()
        self.features = nn.Sequential(nn.Linear(2, 2), nn.Sigmoid())
        self.head = nn.Linear(2, 1)
        self.reentrant = reentrant
        self.checkpointed = True

    @override
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Checkpoint a shared block twice before projecting to a loss input."""
        if not self.checkpointed:
            output: object = self.head(self.features(self.features(inputs)))
        else:
            first = checkpoint(self.features, inputs, use_reentrant=self.reentrant)
            output = self.head(checkpoint(self.features, first, use_reentrant=self.reentrant))
        match output:
            case torch.Tensor():
                return output
            case _:
                pytest.fail("checkpointed model must return a tensor")


@pytest.mark.parametrize("reentrant", [False, True])
def test_checkpoint_capture_does_not_invent_recomputation_samples(
    telemetry_dir: Path,
    reentrant: bool,
) -> None:
    """Preserve model gradients and distinguish unrecorded reentrant internal graphs."""
    model = Checkpointed(reentrant)
    baseline = copy.deepcopy(model)
    baseline.checkpointed = False
    inputs = torch.ones(2, 2, requires_grad=True)
    reference_dir = telemetry_dir.parent / "reference"
    inject_observer(
        baseline, output_dir=reference_dir, sampler=AlwaysSampler(), error_policy="raise"
    )
    baseline(inputs).sum().backward()
    remove_observer(baseline)
    inject_observer(model, output_dir=telemetry_dir, sampler=AlwaysSampler(), error_policy="raise")
    first = model(inputs)
    second = model(inputs)
    second.sum().backward()
    first.sum().backward()
    remove_observer(model)
    for observed, reference in zip(model.parameters(), baseline.parameters(), strict=True):
        assert observed.grad is not None and reference.grad is not None
        torch.testing.assert_close(observed.grad, reference.grad * 2)
    history = pl.read_parquet(telemetry_dir / "history.parquet")
    assert history["sample_id"].n_unique() == 2
    outputs = history.filter((pl.col("layer") == "features.0") & (pl.col("signal") == "output"))
    assert outputs["call_index"].unique().sort().to_list() == [0, 1]
    gradients = history.filter(pl.col("signal") == "output_gradient")
    if reentrant:
        assert outputs["grad_enabled"].unique().to_list() == [False]
        assert set(gradients["layer"]) == {"", "head"}
    else:
        assert outputs["grad_enabled"].unique().to_list() == [True]
        assert set(gradients["layer"]) == {"", "features", "features.0", "features.1", "head"}
        columns = ["layer", "call_index", "signal", "metric", "value"]
        actual = history.filter(pl.col("sample_id") == 0).select(columns).sort(columns[:-1])
        reference = (
            pl.read_parquet(reference_dir / "history.parquet").select(columns).sort(columns[:-1])
        )
        assert actual.equals(reference)
