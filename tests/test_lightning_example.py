"""Integration coverage for the real Lightning TensorBoard demonstration."""

from __future__ import annotations

from pathlib import Path

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.data import DataLoader, TensorDataset

from examples.lightning_mnist import MnistRunConfig, run_training
from tests.json_records import read_report


def test_lightning_example_writes_json_and_tensorboard(tmp_path: Path) -> None:
    """Verify one logger receives Lightning and TorchInstruments metric events."""
    telemetry_dir = tmp_path / "telemetry"
    config = MnistRunConfig(
        data_dir=tmp_path / "unused-data",
        telemetry_dir=telemetry_dir,
        log_dir=tmp_path / "lightning",
        train_batches=3,
        validation_batches=1,
        batch_size=16,
        sample_every_n_forwards=1,
        histogram_every_n_samples=1,
    )
    train_loader = _mnist_shaped_loader(samples=48)
    validation_loader = _mnist_shaped_loader(samples=16)

    tensorboard_dir = run_training(config, train_loader, validation_loader)

    report = read_report(telemetry_dir / "report.json")
    assert report["coverage"]["samples_observed"] == 4
    assert report["coverage"]["backward_samples_observed"] == 3
    assert report["coverage"]["histograms"] > 0

    events = EventAccumulator(
        str(tensorboard_dir),
        size_guidance={"histograms": 0},
    ).Reload()
    scalar_tags = _read_scalar_tags(events)
    output_rms = "torchinstruments/modules/0/call_0/output/rms"
    gradient_rms = "torchinstruments/modules/7/call_0/grad_output/rms"
    assert output_rms in scalar_tags
    assert gradient_rms in scalar_tags
    assert "task/validation_accuracy" in scalar_tags
    assert [event.step for event in events.Scalars(output_rms)] == [0, 1, 2, 3]
    assert [event.step for event in events.Scalars(gradient_rms)] == [0, 1, 2]
    histogram_tags = _read_histogram_tags(events)
    output_distribution = "torchinstruments/modules/0/call_0/output/histograms/distribution"
    gradient_distribution = "torchinstruments/modules/7/call_0/grad_output/histograms/distribution"
    assert output_distribution in histogram_tags
    assert gradient_distribution in histogram_tags
    assert [event.step for event in events.Histograms(output_distribution)] == [0, 1, 2, 3]
    assert [event.step for event in events.Histograms(gradient_distribution)] == [0, 1, 2]


def _mnist_shaped_loader(*, samples: int) -> DataLoader:
    """Provide labeled image batches without downloading data during tests."""
    images = torch.randn(samples, 1, 28, 28)
    targets = torch.arange(samples) % 10
    return DataLoader(TensorDataset(images, targets), batch_size=16)


def _read_scalar_tags(events: EventAccumulator) -> set[str]:
    """Validate TensorBoard's untyped tag payload at the test boundary."""
    raw_tags: object = events.Tags()["scalars"]
    if not isinstance(raw_tags, list):
        raise TypeError("TensorBoard scalar tags must be a list")

    tags: set[str] = set()
    for tag in raw_tags:
        if not isinstance(tag, str):
            raise TypeError("TensorBoard scalar tag must be a string")
        tags.add(tag)
    return tags


def _read_histogram_tags(events: EventAccumulator) -> set[str]:
    """Validate TensorBoard's untyped histogram-tag payload at the test boundary."""
    raw_tags: object = events.Tags()["histograms"]
    if not isinstance(raw_tags, list):
        raise TypeError("TensorBoard histogram tags must be a list")

    tags: set[str] = set()
    for tag in raw_tags:
        if not isinstance(tag, str):
            raise TypeError("TensorBoard histogram tag must be a string")
        tags.add(tag)
    return tags
