"""Capture Lightning child calls without requiring the observed root to execute."""

from __future__ import annotations

import copy
from datetime import timedelta
from enum import StrEnum
from typing import TYPE_CHECKING, assert_never

import lightning as L
import polars as pl
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from typing_extensions import override

from torchinstruments import AlwaysSampler, EveryNForwardsSampler, inject_observer, remove_observer

if TYPE_CHECKING:
    from pathlib import Path


class Invocation(StrEnum):
    """Represent common calls made inside a Lightning step."""

    ROOT = "root"
    FORWARD = "forward"
    CHILD = "child"


class ChildCallingModel(L.LightningModule):
    """Exercise the same network through different outer entry points."""

    def __init__(self, invocation: Invocation) -> None:
        """Build two independently selected layers."""
        super().__init__()
        self.network = nn.Sequential(nn.Linear(2, 2), nn.Sigmoid())
        self.invocation = invocation

    @override
    # Lightning types forward as Any varargs; remove when it supports typed model inputs.
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # pyrefly: ignore[bad-override]
        """Run the computational network."""
        output: object = self.network(inputs)
        match output:
            case torch.Tensor():
                return output
            case _:
                pytest.fail("network must return a tensor")

    def _compute(self, batch: object) -> torch.Tensor:
        """Invoke the network through the configured step entry point."""
        match batch:
            case [torch.Tensor() as inputs]:
                pass
            case _:
                raise TypeError("test batch must contain one input tensor")
        match self.invocation:
            case Invocation.ROOT:
                output: object = self(inputs)
            case Invocation.FORWARD:
                output = self.forward(inputs)
            case Invocation.CHILD:
                output = self.network(inputs)
            case _:
                assert_never(self.invocation)
        match output:
            case torch.Tensor():
                return output
            case _:
                pytest.fail("configured invocation must return a tensor")

    @override
    # Lightning dispatches a batch and index; remove when its hook signature expresses them.
    def training_step(self, batch: object, batch_idx: int) -> torch.Tensor:  # pyrefly: ignore[bad-override]
        """Produce a differentiable loss without changing model invocation semantics."""
        return self._compute(batch).square().mean()

    @override
    # Lightning dispatches a batch and index; remove when its hook signature expresses them.
    def validation_step(self, batch: object, batch_idx: int) -> None:  # pyrefly: ignore[bad-override]
        """Exercise standalone inference without a backward pass."""
        self._compute(batch)

    @override
    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Use a zero learning rate so captured outputs stay comparable."""
        return torch.optim.SGD(self.parameters(), lr=0.0)


@pytest.mark.parametrize("invocation", list(Invocation))
@pytest.mark.parametrize("training", [False, True])
def test_lightning_collects_without_root_dependency(
    telemetry_dir: Path, invocation: Invocation, training: bool
) -> None:
    """Publish all child layers before removal, including gradients from bypassed roots."""
    model = ChildCallingModel(invocation)
    reference = copy.deepcopy(model)
    inputs = torch.ones(2, 2)
    expected = reference(inputs)
    expected.square().mean().backward()
    inject_observer(
        model,
        interval=timedelta(seconds=1),
        output_dir=telemetry_dir,
        error_policy="raise",
    )
    loader = DataLoader(TensorDataset(inputs), batch_size=2)
    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        max_epochs=1,
        num_sanity_val_steps=0,
    )
    try:
        if training:
            trainer.fit(model, loader)
        else:
            trainer.validate(model, loader, verbose=False)
        history = pl.read_parquet(telemetry_dir / "history.parquet")
        outputs = history.filter(pl.col("signal") == "output")
        layers = {"network", "network.0", "network.1"}
        if invocation is not Invocation.CHILD:
            layers.add("")
        assert outputs.height == len(layers) * 9
        assert set(outputs["layer"]) == layers
        final_mean = outputs.filter(
            (pl.col("layer") == "network.1") & (pl.col("metric") == "mean")
        )["value"].item()
        assert final_mean == pytest.approx(expected.detach().mean().item())
        summary = pl.read_json(telemetry_dir / "result.json")
        observed_counts = summary.filter(pl.col("layer").is_in(layers))["tensors"].list.len()
        assert observed_counts.to_list() == [2 if training else 1] * len(layers)
        gradients = history.filter(pl.col("signal") == "output_gradient")
        assert gradients.height == (len(layers) * 9 if training else 0)
        if training:
            for actual, baseline in zip(model.parameters(), reference.parameters(), strict=True):
                torch.testing.assert_close(actual.grad, baseline.grad)
    finally:
        remove_observer(model)
    assert all("forward" not in module.__dict__ for module in model.modules())


def test_unsampled_root_does_not_trigger_independent_capture(telemetry_dir: Path) -> None:
    """Respect an explicit root sampling decision instead of falling back on its children."""
    model = ChildCallingModel(Invocation.ROOT)
    inject_observer(
        model, output_dir=telemetry_dir, sampler=EveryNForwardsSampler(2), error_policy="raise"
    )
    try:
        model(torch.ones(2, 2))
        assert pl.read_parquet(telemetry_dir / "history.parquet").is_empty()
        model(torch.ones(2, 2))
        history = pl.read_parquet(telemetry_dir / "history.parquet")
        assert history.height == 36
        assert history["sample_id"].n_unique() == 1
    finally:
        remove_observer(model)


def test_independent_calls_preserve_delayed_gradient_identity(telemetry_dir: Path) -> None:
    """Bind repeated child calls to their own forwards even with reversed backwards."""
    model = nn.Sequential(nn.Identity())
    inject_observer(model, output_dir=telemetry_dir, sampler=AlwaysSampler(), error_policy="raise")
    try:
        first = model[0](torch.tensor([2.0], requires_grad=True))
        second = model[0](torch.tensor([3.0], requires_grad=True))
        (second * 7).sum().backward()
        (first * 5).sum().backward()
        history = pl.read_parquet(telemetry_dir / "history.parquet")
        gradients = history.filter(
            (pl.col("signal") == "output_gradient") & (pl.col("metric") == "mean")
        ).sort("sample_id")
        assert gradients["sample_id"].to_list() == [0, 1]
        assert gradients["value"].to_list() == [5.0, 7.0]
    finally:
        remove_observer(model)
