# TorchInstruments

Record activation and output-gradient statistics from a PyTorch model without changing its
training loop. Keep the sampled history in Parquet, inspect per-layer summaries in JSON,
and investigate selected layers with TensorBoard histograms.

```python
from torchinstruments import inject_observer, remove_observer

inject_observer(model, output_dir="stats")
try:
    train(model)
finally:
    remove_observer(model)
```

Every run produces:

```text
stats/
    history.parquet    # All successfully collected scalar measurements
    result.json        # History aggregated by layer, call, signal, and tensor path
    index.md           # Schema, sampling information, and queries for an LLM
    tensorboard/       # Histograms for the configured focus layers
```

During training, history is available as completed files in `history.parts/`. Removal combines
these into `history.parquet` using Polars streaming and writes the final `result.json`.
Before removal, `result.json` contains the selected layer catalog with empty measurements.
If the process is interrupted, completed history chunks and flushed TensorBoard events remain
readable. There is no automatic interpretation, ranking, problem classification, or health score.

## What is measured

The default profile is **mean, population std, min, max, p25, p50, p75, zero fraction, and
nonfinite fraction**. Statistics describe all finite entries of each sampled output tensor.
Fractions use the original tensor size. Empty or entirely invalid tensors retain explicit
unavailability reasons. Output gradients are measured separately and matched to their forward.

By default, leaf modules are observed at the first forward after each 60-second interval.
Sampling units are root forwards, not optimizer steps. Inputs, parameter gradients, losses,
and optimizer updates are not collected.

Histograms are always enabled. By default they cover the first **eight observed modules** in
traversal order, on every sampled forward and its observed backward. Scalar history covers every
selected module. Focus the dashboard on layers you want to investigate:

```python
from torchinstruments import EveryNForwardsSampler, HistoryConfig, inject_observer

inject_observer(
    model,
    output_dir="stats",
    sampler=EveryNForwardsSampler(100),
    histogram_selector=lambda name, module: name.startswith("encoder.blocks.7."),
    max_histogram_modules=8,
    history_config=HistoryConfig(buffer_rows=8192, window=20),
)
```

The histogram selector must match at least one observed module. Its limit applies to modules;
shared calls and multiple tensor outputs have separate tags. Selection occurs before histogram
reduction, so unselected modules pay no histogram cost. Histograms contain finite entries;
inspect `nonfinite_fraction` to detect excluded invalid values.

## Read the history with Polars

Parquet uses a long table: one row per scalar measurement. A tensor's metrics share the same
layer, call index, signal, tensor path, and sample ID. A nullable value always has an
`unavailable_reason`; it is never silently replaced with zero.

```python
import polars as pl

trend = (
    pl.scan_parquet("stats/history.parquet")
    .filter(
        (pl.col("layer") == "encoder.blocks.7.proj")
        & (pl.col("signal") == "output_gradient")
        & (pl.col("metric") == "std")
    )
    .select("sample_id", "timestamp", "call_index", "tensor_path", "value")
    .sort("sample_id")
    .collect()
)
```

While training, scan `stats/history.parts/*.parquet` instead. Always order trends by `sample_id`:
backward events may arrive in a different order from their forwards.

`result.json` is an array of layer records. Each tensor contains metric summaries with observation
counts, whole-history mean, population std, min/max, and linearly interpolated quartiles.
`statistics` is keyed by metric name. Each metric also has previous/recent window means.
Aggregates give each available sample statistic equal weight; they are **not pooled tensor
distributions**. For example, `statistics.mean.p50` is the median of sampled tensor means.
Unavailable values are counted and excluded from numeric aggregates; null aggregates carry
an explanation. Exact samples, timestamps, and missing-value reasons stay in Parquet.
All selected layers are listed, including unexecuted layers; there is no byte-budget selection.

## Find a problem, then test a fix

The [debugging examples](examples/README.md) reproduce inactive ReLUs, sigmoid saturation,
growing scale, invalid arithmetic, and an accidentally detached branch. Each uses matched
baseline, broken, and fixed runs, prints a Polars evidence table, and creates comparison plots.
The fault starts after four healthy samples so its onset is visible.

```bash
uv run examples/find_problems.py
```

These examples demonstrate known interventions, not universal thresholds for unfamiliar models.
See [research workflows](docs/research-workflows.md) for evidence and interpretation limits.

## Integration and ownership

`rank_policy="rank0"` is the default and attaches nothing on nonzero ranks. With
`rank_policy="all"`, each rank writes all four artifacts under its own `rank-NNN/` directory.
Compare ranks explicitly; summaries never silently average measurements across ranks.

Use `capture_direct_forwards=True` for model code that calls `module.forward(...)` directly.
Removal restores observer-owned overrides. Capture does not replace model outputs or gradients,
and the observer never registers a parameter, buffer, or child module.

A supplied `TensorBoardSink(logger)` additionally writes histograms into an existing logger.
The logger remains caller-owned. The ordinary directory artifacts are still produced.
The [Lightning example](examples/lightning_mnist.py) demonstrates this with a real MNIST model.

Runtime dependencies are PyTorch, Polars, and TensorBoard. The ordinary training loop and
telemetry remain trainer-independent. CPU behavior is tested; CUDA performance and
`torch.compile` compatibility are not claimed.

## Migration from 0.6

`report.json`, ranked findings, `ReportConfig`, live indicator aggregation, scalar logger exports,
and rank-report merging have been removed. Use `history.parquet`, `result.json`, and `HistoryConfig`.
`DirectorySink` always produces all four artifacts; `write_full_details` is no longer supported.
Custom scalar reducers and additional sinks remain supported.

## Development

```bash
uv sync --dev
uv run ruff check src tests examples
uv run ruff format --check src tests examples
uv run pyrefly check src tests examples
uv run pytest
uv build --wheel
```

MIT licensed. Author: Vadym Stupakov <vadim.stupakov@gmail.com>.
