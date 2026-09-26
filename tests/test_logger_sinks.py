"""Real TensorBoard event and external-writer ownership checks."""

from pathlib import Path

import polars as pl
import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from torchinstruments import (
    AlwaysSampler,
    TensorBoardSink,
    has_observer,
    histogram,
    inject_observer,
    remove_observer,
)
from torchinstruments.sinks import directory


@pytest.mark.parametrize("operation", ["add_histogram_raw", "flush"])
@pytest.mark.parametrize("policy", ["ignore", "warn", "raise"])
def test_dashboard_failure_keeps_live_summary(
    telemetry_dir: Path, monkeypatch: pytest.MonkeyPatch, operation: str, policy: str
) -> None:
    """Publish scalar evidence and failure detail before applying the observer error policy."""

    def fail(*args: object, **kwargs: object) -> None:
        """Simulate an unavailable dashboard destination."""
        raise OSError("dashboard unavailable")

    model = nn.Identity()
    inject_observer(model, sampler=AlwaysSampler(), output_dir=telemetry_dir, error_policy=policy)
    try:
        with monkeypatch.context() as patched:
            patched.setattr(SummaryWriter, operation, fail)
            inputs = torch.tensor([1.0, 3.0])
            match policy:
                case "raise":
                    with pytest.raises(ExceptionGroup, match="artifact publication failed"):
                        model(inputs)
                case "warn":
                    with pytest.warns(RuntimeWarning, match="artifact publication failed"):
                        assert model(inputs) is inputs
                case _:
                    assert model(inputs) is inputs
            history = pl.read_parquet(telemetry_dir / "history.parquet")
            assert history.filter(pl.col("metric") == "mean")["value"].item() == 2.0
            summary = (
                pl.read_json(telemetry_dir / "result.json")
                .explode("tensors", empty_as_null=True)
                .unnest("tensors")
                .select(pl.col("statistics").struct.field("mean").struct.field("mean"))
            )
            assert summary.item() == 2.0
            index = (telemetry_dir / "index.md").read_text()
            assert "tensorboard: OSError: dashboard unavailable" in index
            assert "Forward samples: 1" in index
    finally:
        remove_observer(model)


@pytest.fixture
def closed_writers(monkeypatch: pytest.MonkeyPatch) -> list[SummaryWriter]:
    """Track owned writer cleanup while still releasing actual event resources."""
    closed: list[SummaryWriter] = []
    original = SummaryWriter.close

    def close(writer: SummaryWriter) -> None:
        """Record resource release without replacing its effects."""
        closed.append(writer)
        original(writer)

    monkeypatch.setattr(SummaryWriter, "close", close)
    return closed


@pytest.mark.parametrize("fail_dashboard", [False, True])
def test_summary_failure_preserves_index_and_all_errors(
    telemetry_dir: Path, monkeypatch: pytest.MonkeyPatch, fail_dashboard: bool
) -> None:
    """Keep independent failure causes visible even when multiple artifacts cannot publish."""

    def fail_summary(*args: object, **kwargs: object) -> None:
        """Simulate an unwritable summary destination."""
        raise OSError("summary unavailable")

    def fail_histogram(*args: object, **kwargs: object) -> None:
        """Simulate an independent dashboard failure."""
        raise OSError("dashboard unavailable")

    model = nn.Identity()
    inject_observer(model, sampler=AlwaysSampler(), output_dir=telemetry_dir, error_policy="raise")
    try:
        with monkeypatch.context() as patched:
            patched.setattr(directory, "write_result", fail_summary)
            if fail_dashboard:
                patched.setattr(SummaryWriter, "add_histogram_raw", fail_histogram)
            with pytest.raises(ExceptionGroup) as caught:
                model(torch.tensor([1.0, 3.0]))
            causes = [str(error) for error in caught.value.exceptions]
            expected = (
                ["dashboard unavailable", "summary unavailable"]
                if fail_dashboard
                else ["summary unavailable"]
            )
            assert causes == expected
            index = (telemetry_dir / "index.md").read_text()
            assert "result.json: OSError: summary unavailable" in index
            assert "Forward samples: 1" in index
            assert pl.read_parquet(telemetry_dir / "history.parquet").height == 9
    finally:
        remove_observer(model)


@pytest.mark.parametrize("stage", ["dashboard", "summary"])
def test_partial_initialization_closes_owned_writer(
    telemetry_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    closed_writers: list[SummaryWriter],
    stage: str,
) -> None:
    """Release resources acquired before initialization fails and leave the model unattached."""

    def fail(*args: object, **kwargs: object) -> None:
        """Interrupt setup after the owned event writer has been created."""
        raise OSError("setup unavailable")

    if stage == "dashboard":
        monkeypatch.setattr(TensorBoardSink, "initialize", fail)
    else:
        monkeypatch.setattr(directory, "write_result", fail)
    model = nn.Identity()
    with pytest.raises(OSError, match="setup unavailable"):
        inject_observer(model, output_dir=telemetry_dir)
    assert len(closed_writers) == 1
    assert not has_observer(model)
    assert "forward" not in model.__dict__


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
