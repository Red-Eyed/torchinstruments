"""Real TensorBoard event and external-writer ownership checks."""

from pathlib import Path

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from torchinstruments import (
    AlwaysSampler,
    TensorBoardSink,
    histogram,
    inject_observer,
    remove_observer,
)


def test_external_writer_remains_usable_and_histograms_preserve_counts(tmp_path: Path) -> None:
    """Retain caller ownership and replay all finite values including outliers."""
    writer = SummaryWriter(str(tmp_path / "external"))
    model = nn.Identity()
    inject_observer(
        model,
        sampler=AlwaysSampler(),
        output_dir=tmp_path / "run",
        histograms=[histogram(bins=2, value_range=(-1, 1), every_n_samples=1)],
        sink=TensorBoardSink(writer),
        error_policy="raise",
    )
    model(torch.tensor([-2.0, -1, 0, 1, 2, float("nan")], requires_grad=True)).sum().backward()
    remove_observer(model)
    writer.add_scalar("caller/after_removal", 1, 1)
    writer.flush()
    writer.close()
    events = EventAccumulator(str(tmp_path / "external"), size_guidance={"histograms": 0}).Reload()
    tags = events.Tags()["scalars"]
    assert isinstance(tags, list)
    assert "caller/after_removal" in tags
    output = events.Histograms(
        "torchinstruments/train/grad_enabled_true/modules/@root/call_0/output/histograms/distribution"
    )[0]
    assert output.histogram_value.num == 5
    assert sum(output.histogram_value.bucket) == 5
    assert output.histogram_value.min == -2
    assert output.histogram_value.max == 2
    assert output.histogram_value.sum == 0
    assert output.histogram_value.sum_squares == 10
