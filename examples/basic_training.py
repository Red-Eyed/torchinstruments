"""Demonstrate passive telemetry around an ordinary PyTorch training loop."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import torch
from torch import nn

from torchinstruments import (
    AlwaysSampler,
    DirectorySink,
    default_reducers,
    inject_observer,
    leaf_modules,
    remove_observer,
)


def build_model() -> nn.Module:
    """Create a small regression model with several observable leaf modules."""
    return nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 1),
    )


def train(model: nn.Module, *, iterations: int = 3) -> None:
    """Train normally, without passing observer state through the loop."""
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    for _ in range(iterations):
        inputs = torch.randn(16, 4)
        targets = inputs.sum(dim=-1, keepdim=True)

        optimizer.zero_grad()
        predictions = model(inputs)
        loss = nn.functional.mse_loss(predictions, targets)
        loss.backward()
        optimizer.step()


def run_demo(output_dir: Path) -> None:
    """Run instrumented training and always detach hooks before returning.

    ``AlwaysSampler`` makes this short demonstration deterministic: each root forward creates a
    snapshot. Production training can omit ``sampler`` to use the one-minute default interval.
    """
    torch.manual_seed(7)
    model = build_model()
    inject_observer(
        model,
        sampler=AlwaysSampler(),
        selector=leaf_modules(),
        reducers=default_reducers(),
        sink=DirectorySink(output_dir),
        error_policy="warn",
    )
    try:
        train(model)
    finally:
        # Explicit removal closes the sink and releases every module and graph hook.
        remove_observer(model)


def main() -> None:
    """Write a collision-free demonstration run and report its location."""
    output_dir = Path("stats") / f"basic-training-{uuid4().hex[:8]}"
    run_demo(output_dir)
    print(f"Telemetry written to {output_dir}")


if __name__ == "__main__":
    main()
