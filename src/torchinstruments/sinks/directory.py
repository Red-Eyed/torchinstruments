"""Persist scalar history, descriptive layer summaries, and TensorBoard histograms."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from torchinstruments.distributed import RankInfo
from torchinstruments.errors import SinkAlreadyInitializedError
from torchinstruments.history.parquet import ParquetHistory
from torchinstruments.history.records import observations
from torchinstruments.history.summary import HistoryConfig, write_result
from torchinstruments.records import (
    ErrorRecord,
    ModuleCallRecord,
    ModuleRecord,
    RunRecord,
    SampleRecord,
    SampleState,
)
from torchinstruments.sinks.files import write_text_atomic
from torchinstruments.sinks.tensorboard import TensorBoardSink

_DEFAULT_HISTORY = HistoryConfig()
_LOCAL_RANK = RankInfo(0, 1)


class DirectorySink:
    """Own all run artifacts and keep history buffering independent of run duration."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        history_config: HistoryConfig = _DEFAULT_HISTORY,
        rank: RankInfo = _LOCAL_RANK,
        isolate_rank: bool = False,
    ) -> None:
        """Bind explicit limits and rank-private paths without opening any resources."""
        root = Path(output_dir)
        self._output_dir = root / f"rank-{rank.rank:03d}" if isolate_rank else root
        self._config = history_config
        self._rank = rank
        self._history = ParquetHistory(
            self._output_dir / "history.parquet",
            buffer_rows=history_config.buffer_rows,
        )
        self._initialized = False
        self._samples = self._backwards = self._errors_omitted = 0
        self._errors: list[ErrorRecord] = []
        self._run: RunRecord
        self._modules: dict[str, ModuleRecord]
        self._writer: SummaryWriter
        self._dashboard: TensorBoardSink

    def initialize(self, run: RunRecord, modules: dict[str, ModuleRecord]) -> None:
        """Create all destinations and reject reuse of a previous telemetry directory."""
        if self._output_dir.exists() and any(self._output_dir.iterdir()):
            raise SinkAlreadyInitializedError("telemetry directory already contains a run")
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._run = run
        self._modules = modules
        self._history.initialize()
        self._writer = SummaryWriter(log_dir=str(self._output_dir / "tensorboard"))
        self._dashboard = TensorBoardSink(self._writer)
        self._dashboard.initialize(run, modules)
        self._initialized = True
        try:
            self._publish()
        except BaseException:
            self._writer.close()
            self._initialized = False
            raise

    def observe(self, sample: SampleRecord) -> None:
        """Persist only new measurements and refresh live collection metadata."""
        if not self._initialized:
            raise RuntimeError("sink must be initialized before observing samples")
        if sample.state is SampleState.FORWARD_COMPLETE:
            self._samples += 1
        else:
            self._backwards += 1
        for observation in observations(sample):
            self._history.append(observation)
        self._history.flush()
        for error in sample.errors:
            self._remember_error(error)
        self._remember_unavailable_histograms(sample)
        self._dashboard.observe(sample)
        self._writer.flush()
        self._publish_index(closed=False)

    def _remember_error(self, error: ErrorRecord) -> None:
        """Bound retained failure detail independently of run length."""
        if len(self._errors) < 100:
            self._errors.append(replace(error, message=error.message[:500]))
        else:
            self._errors_omitted += 1

    def _remember_unavailable_histograms(self, sample: SampleRecord) -> None:
        """Keep numerical histogram gaps visible after transient records are discarded."""
        for name, calls in sample.modules.items():
            for call in calls:
                self._remember_call_histograms(name, call, sample)

    def _remember_call_histograms(
        self, name: str, call: ModuleCallRecord, sample: SampleRecord
    ) -> None:
        """Persist histogram absence only for the event's newly collected signal."""
        tensors = (
            call.outputs if sample.state is SampleState.FORWARD_COMPLETE else call.output_gradients
        )
        for path, tensor in tensors.items():
            for histogram, reason in tensor.unavailable_histograms.items():
                self._remember_error(
                    ErrorRecord(
                        sample.timestamp,
                        name,
                        f"{path}/histograms/{histogram}",
                        "HistogramUnavailable",
                        reason,
                    )
                )

    def close(self) -> None:
        """Finalize history with a streaming Polars pass and close the owned writer."""
        if not self._initialized:
            return
        try:
            self._history.close()
            self._publish(closed=True)
        finally:
            self._dashboard.close()
            self._writer.close()
            self._initialized = False

    def _publish(self, *, closed: bool = False) -> None:
        """Atomically refresh a descriptive result and its reading guide."""
        source = (
            self._output_dir / "history.parquet" if closed else self._history.parts / "*.parquet"
        )
        write_result(source, self._output_dir / "result.json", self._modules, self._config.window)
        self._publish_index(closed=closed)

    def _publish_index(self, *, closed: bool) -> None:
        """Refresh live counts and errors without rescanning historical measurements."""
        metadata = (
            f"\nSchema: {self._run.schema_version}; TorchInstruments: "
            f"{self._run.observer_version}; PyTorch: {self._run.torch_version}.\n"
            f"\n## Collection\n\nSampling: {self._run.sampling.type} "
            f"{self._run.sampling.settings}. "
            f"Rank: {self._rank.rank}/{self._rank.world_size}.\n"
            f"Forward samples: {self._samples}; backward samples: {self._backwards}.\n"
            f"Histogram layers: {self._run.collection.histogram_modules}.\n"
            f"Collection errors and unavailable histograms: "
            f"{len(self._errors) + self._errors_omitted}.\n"
        )
        for error in self._errors:
            metadata += (
                f"- {error.module}: {error.probe}: {error.exception_type}: {error.message}\n"
            )
        write_text_atomic(
            self._output_dir / "index.md", _index(closed, self._config.window) + metadata
        )


def _index(closed: bool, window: int) -> str:
    """Explain the physical history location and interpretation of each artifact."""
    source = "history.parquet" if closed else "history.parts/*.parquet"
    return f'''# Model measurements

Read [result.json](result.json) for per-layer summaries. It contains measurements,
not diagnoses, rankings, or categories. Every selected layer is listed, including
layers without observations. During training result.json is a catalog; removal finalizes
its history aggregation. No layers are ranked or removed to meet a report byte budget.

## History

History source: `{source}`. During training, completed chunks live in `history.parts/`.
Observer removal streams those chunks into [history.parquet](history.parquet).
After an interrupted run, query the remaining chunks directly.

Each Parquet row is one statistic for one tensor observation. Identity columns are
`layer`, `call_index`, `signal`, `tensor_path`, `mode`, and `grad_enabled`.
`mode` is the selected module's train/eval mode at invocation; `grad_enabled` records
autograd recording at that invocation. Delayed backwards retain this original context.
`sample_id` and `forward_index`
identify the originating forward, not an optimizer step. `timestamp` is its UTC time.
`shape` and `dtype` describe that observation. `metric` names the statistic and
`value` holds it. A missing value always has an `unavailable_reason`.
Sort by `sample_id`: delayed backwards may arrive out of order.

```python
import polars as pl

history = pl.scan_parquet("{source}")
trend = (
    history.filter(
        (pl.col("layer") == "encoder.blocks.0")
        & (pl.col("signal") == "output_gradient")
        & (pl.col("metric") == "std")
    )
    .select("sample_id", "timestamp", "call_index", "tensor_path", "value", "unavailable_reason")
    .sort("sample_id")
    .collect()
)
```

## Statistics and aggregation

Defaults: mean, population std, min, max, p25, p50 (median), p75,
zero_fraction, and nonfinite_fraction. Distribution statistics use finite entries;
fractions use the original tensor size. Empty tensors have unavailable statistics.
These are tensor-wide statistics, not per-channel statistics.

`result.json` is an array of layer records. Within each layer, tensors are separated
by call, signal, tensor path, module mode, and autograd recording context.
`statistics` maps metric names to whole-history aggregates: observations, unavailable
count, mean, population std, min, max, and linearly interpolated p25/p50/p75.
Each observation has equal weight; unavailable values are excluded from numeric aggregates.
These summarize sampled statistics: for example statistics.mean.p50 is the median of
sampled tensor means, NOT the median of pooled tensor entries.
previous_window_mean and recent_window_mean compare adjacent windows ordered by sample ID.
The recent window contains at most {window} observations; the previous window contains
up to {window} preceding observations. Missing values occupy window positions but are
excluded from their means. Null aggregates have an accompanying unavailable_reason;
no missing observation is replaced by zero. Use Parquet for exact samples, timestamps,
missing-value reasons, and window coverage. latest_shape and dtype describe the latest
observation; a history may contain different tensor shapes.


## TensorBoard

[tensorboard/](tensorboard/) holds histogram events for the configured focus layers.
Open with `tensorboard --logdir tensorboard`. Histogram steps are telemetry sample IDs.
Histogram selection limits dashboard size; scalar history still covers all selected layers.
Histograms describe finite entries; check nonfinite_fraction in the history as well.
Missing gradients can mean no backward, an unused path, disabled gradients, or disconnection.
Reentrant activation checkpointing's internal gradient capture is unsupported: its original
internal forwards have grad_enabled=false and recomputation is not assigned a guessed sample.
Non-reentrant checkpointing is covered by regression tests. No-grad output observations are
visible coverage limitations, never evidence that the model's parameter gradients are absent.
Only the first backward per sampled forward is recorded. AMP loss scaling also scales these
raw output gradients; the observer does not normalize them to optimizer or parameter gradients.
An observed output gradient is not a parameter gradient. These measurements alone cannot
establish a cause of task loss or accuracy changes. Use matched runs to test hypotheses.
'''
