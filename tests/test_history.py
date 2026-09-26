"""End-to-end regression coverage for history, layer summaries, and focused histograms."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import polars as pl
import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch import nn
from typing_extensions import override

from torchinstruments import AlwaysSampler, HistoryConfig, inject_observer, remove_observer
from torchinstruments.history.parquet import ParquetHistory
from torchinstruments.history.records import Observation, Signal
from torchinstruments.history.summary import aggregate_history
from torchinstruments.records import Absent, ExecutionContext, ModuleMode

if TYPE_CHECKING:
    from pathlib import Path

    from torchinstruments.reducers import HistogramReductionResult


def summary_metrics(path: Path) -> pl.DataFrame:
    """Flatten a result through Polars without untyped JSON dictionaries."""
    return (
        pl.read_json(path)
        .explode("tensors", empty_as_null=True)
        .unnest("tensors")
        .select("signal", pl.col("statistics").struct.field("mean").alias("summary"))
        .unnest("summary")
    )


@pytest.mark.parametrize("value", [0, 0.0, Absent("not measured")])
def test_history_numeric_contract(tmp_path: Path, value: float | Absent) -> None:
    """Accept integer and float zeros without confusing either with unavailable data."""
    history = ParquetHistory(tmp_path / "history.parquet", buffer_rows=7)
    history.initialize()
    history.append(
        Observation(
            "layer",
            0,
            Signal.OUTPUT,
            "output",
            0,
            0,
            datetime.now(UTC),
            (1,),
            "float32",
            "mean",
            value,
            ExecutionContext(ModuleMode.TRAIN, True),
        )
    )
    history.close()
    row = pl.read_parquet(history.path)
    match value:
        case Absent(reason=reason):
            assert row["value"].item() is None
            assert row["unavailable_reason"].item() == reason
        case _:
            assert row["value"].item() == 0.0
            assert row["unavailable_reason"].item() == ""


def test_artifacts_refresh_before_removal(telemetry_dir: Path) -> None:
    """Expose complete history, summaries, and histograms without requiring removal."""
    model = nn.Identity()
    inject_observer(model, sampler=AlwaysSampler(), output_dir=telemetry_dir, error_policy="raise")
    try:
        assert pl.read_parquet(telemetry_dir / "history.parquet").is_empty()
        for count, value in enumerate([2.0, 4.0], start=1):
            output = model(torch.tensor([value], requires_grad=True))
            forward = summary_metrics(telemetry_dir / "result.json").filter(
                pl.col("signal") == "output"
            )
            assert forward["observations"].item() == count
            assert forward["mean"].item() == (2.0 if count == 1 else 3.0)
            assert pl.read_parquet(telemetry_dir / "history.parquet").height == (count * 2 - 1) * 9
            output.sum().backward()
            backward = summary_metrics(telemetry_dir / "result.json").filter(
                pl.col("signal") == "output_gradient"
            )
            assert backward["observations"].item() == count
            assert backward["mean"].item() == 1.0
            assert pl.read_parquet(telemetry_dir / "history.parquet").height == count * 18
            assert (
                f"Forward samples: {count}; backward samples: {count}."
                in (telemetry_dir / "index.md").read_text()
            )
        live_summary = (telemetry_dir / "result.json").read_text()
        live_history = pl.read_parquet(telemetry_dir / "history.parquet")
        events = EventAccumulator(
            str(telemetry_dir / "tensorboard"), size_guidance={"histograms": 0}
        ).Reload()
        tags = events.Tags()["histograms"]
        assert isinstance(tags, list)
        assert len(tags) == 2
        for tag in tags:
            assert [event.step for event in events.Histograms(tag)] == [0, 1]
    finally:
        remove_observer(model)
    assert (telemetry_dir / "result.json").read_text() == live_summary
    assert pl.read_parquet(telemetry_dir / "history.parquet").equals(live_history)


def test_all_artifacts_and_exact_per_layer_statistics(telemetry_dir: Path) -> None:
    """Require history, unscored JSON, a reading guide, and real TensorBoard events."""
    model = nn.Identity()
    inject_observer(
        model,
        sampler=AlwaysSampler(),
        output_dir=telemetry_dir,
        history_config=HistoryConfig(window=2),
        error_policy="raise",
    )
    for value in [1.0, 2.0, 3.0, 4.0]:
        model(torch.tensor([value - 1, value + 1], requires_grad=True)).sum().backward()
    live = pl.scan_parquet(telemetry_dir / "history.parts/*.parquet").select(pl.len()).collect()
    assert live.item() == 72
    remove_observer(model)
    assert {p.name for p in telemetry_dir.iterdir()} == {
        "history.parquet",
        "result.json",
        "index.md",
        "tensorboard",
    }
    history = pl.read_parquet(telemetry_dir / "history.parquet")
    assert history.height == 72
    assert history["timestamp"].dtype == pl.Datetime("us", "UTC")
    mean = summary_metrics(telemetry_dir / "result.json").filter(pl.col("signal") == "output")
    assert mean["observations"].item() == 4
    assert mean["unavailable"].item() == 0
    assert mean["mean"].item() == 2.5
    assert mean["std"].item() == pytest.approx(1.25**0.5)
    assert mean["min"].item() == 1
    assert mean["max"].item() == 4
    assert mean["p25"].item() == 1.75
    assert mean["p50"].item() == 2.5
    assert mean["p75"].item() == 3.25
    assert mean["previous_window_mean"].item() == 1.5
    assert mean["recent_window_mean"].item() == 3.5
    assert not {"first", "latest", "minimum", "maximum", "sample_id", "timestamp"} & set(
        mean.columns
    )
    assert pl.read_json(telemetry_dir / "result.json").columns == [
        "layer",
        "type",
        "aliases",
        "tensors",
    ]
    events = EventAccumulator(
        str(telemetry_dir / "tensorboard"), size_guidance={"histograms": 0}
    ).Reload()
    tags = events.Tags()["histograms"]
    assert isinstance(tags, list)
    assert len(tags) == 2
    assert not events.Tags()["scalars"]
    for tag in tags:
        assert [event.step for event in events.Histograms(tag)] == [0, 1, 2, 3]


@pytest.mark.parametrize("count", [0, 1, 2, 3, 4, 7])
def test_adjacent_windows_never_overlap(telemetry_dir: Path, count: int) -> None:
    """Compare windows using actual observation counts even before warm-up."""
    model = nn.Identity()
    inject_observer(
        model,
        sampler=AlwaysSampler(),
        output_dir=telemetry_dir,
        history_config=HistoryConfig(window=2),
        error_policy="raise",
    )
    for value in range(count):
        model(torch.tensor(float(value)))
    remove_observer(model)
    metrics = aggregate_history(telemetry_dir / "history.parquet", 2).collect()
    if count == 0:
        assert metrics.is_empty()
        assert pl.read_json(telemetry_dir / "result.json")["tensors"].item().is_empty()
        return
    mean = metrics.filter(pl.col("metric") == "mean")
    previous = list(range(count))[:-2][-2:]
    recent = list(range(count))[-2:]
    assert mean["previous_window_mean"].item() == (
        sum(previous) / len(previous) if previous else None
    )
    assert mean["recent_window_mean"].item() == sum(recent) / len(recent)


@pytest.mark.parametrize("values", [[], [float("nan"), float("inf")], [float("nan"), 2.0]])
def test_unavailable_values_remain_distinct_from_zero(
    telemetry_dir: Path, values: list[float]
) -> None:
    """Preserve finite measurements and explicit absence reasons without nonstandard JSON."""
    model = nn.Identity()
    inject_observer(model, sampler=AlwaysSampler(), output_dir=telemetry_dir, error_policy="raise")
    model(torch.tensor(values))
    remove_observer(model)
    history = pl.read_parquet(telemetry_dir / "history.parquet")
    mean = history.filter(pl.col("metric") == "mean")
    if 2.0 in values:
        assert mean["value"].item() == 2
        assert history.filter(pl.col("metric") == "nonfinite_fraction")["value"].item() == 0.5
    else:
        assert mean["value"].item() is None
        assert mean["unavailable_reason"].item()
        summary = summary_metrics(telemetry_dir / "result.json")
        assert summary["mean"].item() is None
        assert summary["unavailable"].item() == 1
        assert summary["unavailable_reason"].item()
    text = (telemetry_dir / "result.json").read_text()
    assert "NaN" not in text and "Infinity" not in text


def test_history_summary_excludes_missing_values_without_shifting_windows(
    telemetry_dir: Path,
) -> None:
    """Keep missing sample positions while computing aggregates from available values."""
    model = nn.Identity()
    inject_observer(
        model,
        sampler=AlwaysSampler(),
        output_dir=telemetry_dir,
        history_config=HistoryConfig(window=2),
        error_policy="raise",
    )
    for value in [1.0, float("nan"), 3.0, float("nan")]:
        model(torch.tensor(value))
    remove_observer(model)
    summary = summary_metrics(telemetry_dir / "result.json")
    assert summary["observations"].item() == 4
    assert summary["unavailable"].item() == 2
    assert summary["mean"].item() == 2
    assert summary["std"].item() == 1
    assert summary["p25"].item() == 1.5
    assert summary["p50"].item() == 2
    assert summary["p75"].item() == 2.5
    assert summary["previous_window_mean"].item() == 1
    assert summary["recent_window_mean"].item() == 3


class Shared(nn.Module):
    """Invoke an aliased module twice while retaining per-call identities."""

    def __init__(self) -> None:
        """Register two names for one module object."""
        super().__init__()
        self.first = nn.Linear(2, 2)
        self.second = self.first

    @override
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Use both aliases in one forward."""
        output: object = self.second(self.first(inputs))
        match output:
            case torch.Tensor():
                return output
            case _:
                pytest.fail("shared model must return a tensor")


def test_shared_calls_and_delayed_backwards(telemetry_dir: Path) -> None:
    """Preserve separate calls and order backwards by forward identity, not arrival."""
    model = Shared()
    inject_observer(model, sampler=AlwaysSampler(), output_dir=telemetry_dir, error_policy="raise")
    first = model(torch.ones(2, 2))
    second = model(torch.ones(5, 2))
    (second.sum() * 7).backward()
    (first.sum() * 3).backward()
    remove_observer(model)
    frame = pl.read_parquet(telemetry_dir / "history.parquet")
    assert set(frame["call_index"]) == {0, 1}
    gradients = frame.filter(
        (pl.col("signal") == "output_gradient")
        & (pl.col("call_index") == 1)
        & (pl.col("metric") == "mean")
    ).sort("sample_id")
    assert gradients["value"].to_list() == [3, 7]
    metrics = aggregate_history(telemetry_dir / "history.parquet", 1).collect()
    selected = metrics.filter(
        (pl.col("signal") == "output_gradient")
        & (pl.col("call_index") == 1)
        & (pl.col("metric") == "mean")
    )
    assert selected["previous_window_mean"].item() == 3
    assert selected["recent_window_mean"].item() == 7
    assert pl.read_json(telemetry_dir / "result.json").filter(pl.col("layer") == "first")[
        "aliases"
    ].item().to_list() == [
        "first",
        "second",
    ]


def test_thousands_of_layers_keep_history_and_limit_histogram_work(telemetry_dir: Path) -> None:
    """Retain every observed layer while restricting actual histogram reduction work."""
    model = nn.Sequential(*(nn.Identity() for _ in range(1000)))
    calls = 0
    from torchinstruments import histogram

    configured = histogram(every_n_samples=1)

    def counted(tensor: torch.Tensor, *, sample_id: int) -> HistogramReductionResult:
        """Count reductions to distinguish collection selection from tag filtering."""
        nonlocal calls
        calls += 1
        return configured(tensor, sample_id=sample_id)

    inject_observer(
        model,
        sampler=AlwaysSampler(),
        output_dir=telemetry_dir,
        histograms=[counted],
        max_histogram_modules=2,
        error_policy="raise",
    )
    model(torch.tensor([1.0, 2.0]))
    remove_observer(model)
    assert calls == 2
    assert pl.read_json(telemetry_dir / "result.json").height == 1001
    assert (
        pl.scan_parquet(telemetry_dir / "history.parquet")
        .select(pl.col("layer").n_unique())
        .collect()
        .item()
        == 1001
    )
    events = EventAccumulator(str(telemetry_dir / "tensorboard")).Reload()
    tags = events.Tags()["histograms"]
    assert isinstance(tags, list)
    assert len(tags) == 2


def test_parquet_chunks_enforce_buffer_bound(tmp_path: Path) -> None:
    """Prove each on-disk batch is bounded regardless of the total history length."""
    history = ParquetHistory(tmp_path / "history.parquet", buffer_rows=7)
    history.initialize()
    for sample in range(101):
        history.append(
            Observation(
                "layer",
                0,
                Signal.OUTPUT,
                "output",
                sample,
                sample,
                datetime.now(UTC),
                (2,),
                "float32",
                "mean",
                float(sample),
                ExecutionContext(ModuleMode.TRAIN, True),
            )
        )
    history.append(
        Observation(
            "layer",
            0,
            Signal.OUTPUT,
            "output",
            101,
            101,
            datetime.now(UTC),
            (0,),
            "float32",
            "mean",
            Absent("empty"),
            ExecutionContext(ModuleMode.TRAIN, True),
        )
    )
    history.flush()
    sizes = [
        pl.scan_parquet(path).select(pl.len()).collect().item()
        for path in history.parts.glob("*.parquet")
    ]
    assert max(sizes) <= 7
    history.close()
    assert pl.scan_parquet(history.path).select(pl.len()).collect().item() == 102
    assert not history.parts.exists()


def test_removal_detaches_pending_gradients(telemetry_dir: Path) -> None:
    """Avoid writing stale gradients after the observer has finalized its history."""
    model = nn.Linear(2, 1)
    inject_observer(model, sampler=AlwaysSampler(), output_dir=telemetry_dir, error_policy="raise")
    output = model(torch.ones(2, 2))
    remove_observer(model)
    before = (telemetry_dir / "history.parquet").read_bytes()
    output.sum().backward()
    remove_observer(model)
    assert (telemetry_dir / "history.parquet").read_bytes() == before
