# Changelog

All notable changes to TorchInstruments are documented here.

## [Unreleased]

## [0.9.0] - 2026-09-24

### Backwards Incompatible Changes

- Add `mode` and `grad_enabled` to Parquet observations and JSON tensor identities (schema 7).
  Include both in queries and grouping keys. TensorBoard tags now include these contexts.
  Older histories require their matching reader or an explicit migration using known context;
  the new reader never infers training mode from absent gradients.
- Sample the first forward immediately under the default timed sampler, then apply the existing
  interval. Short evaluation runs now produce measurements without a custom sampler.

### Bug Fixes

- Bind output gradients before downstream in-place operations can change their gradient edge.
  Preserve original sample identity and publish only the first backward, including autograd.grad.
- Compute exact quartiles above torch.quantile's element limit using order-statistic selection.
- Keep successful scalar and histogram measurements when another reducer fails under warn/ignore;
  preserve reasons for unavailable quartiles and histograms. Explicit raise still propagates errors.
- Support native MPS histogram collection without float64 operations; retain integer counts and
  use float32 moments on that backend.
- Count nonfinite entries directly so rare invalid values do not disappear through subtraction
  from a rounded finite fraction.

### Documentation

- Simplify the README and ordinary example around interval and output directory; preserve the
  existing API and imports. Reduce Lightning telemetry settings to one interval and the benchmark
  CLI to three fixed workloads without dimension-tuning flags.
- Attach telemetry inside the Lightning model's `on_fit_start()` using `self.trainer.logger`,
  with cleanup in `on_fit_end()` and a failure cleanup fallback.
- State the tested non-reentrant checkpoint behavior and unsupported reentrant internal gradient
  capture. Persist the original no-grad context instead of guessing recomputation ownership.
- Document raw loss-scaled gradient semantics, train/eval separation, and schema migration.
- Add a reproducible benchmark CLI and local CPU/MPS measurements, including large activations,
  1,000 modules, and 900,000-row histories. Every-forward overhead and native memory growth remain
  limitations; no CUDA performance claim is made.

## [0.8.0] - 2026-09-24

### Backwards Incompatible Changes

#### JSON history summaries

Replace each tensor's `statistics` list with an object keyed by metric name. Each metric
contains observation and unavailable counts, whole-history mean, population standard deviation,
min/max, linearly interpolated quartiles, and previous/recent window means. Remove embedded
first/latest/extreme sample records; exact samples and timestamps remain in `history.parquet`.

JSON consumers should look up `tensor.statistics.mean` instead of searching the statistics list
for `metric == "mean"`. Replace `recent_window.mean` with `recent_window_mean` and query Parquet
for individual observations. Aggregates give each available sample statistic equal weight:
`statistics.mean.p50` is the median of sampled tensor means, not a pooled tensor median.
Missing samples retain their window positions and are excluded from numeric aggregates.

### Improvements

- Write `result.json` with two-space indentation for direct inspection.

### Documentation

- Update the generated `index.md` and analysis guides to explain history aggregates,
  adjacent window means, missing measurements, and when to query Parquet.

## [0.7.0] - 2026-09-24

### Highlights

- Keep sampled per-layer measurement history in Parquet, aggregate it with Polars, and write
  descriptive per-layer JSON without scores, rankings, or problem categories.
- Produce all four run artifacts: `history.parquet`, `result.json`, an LLM reading guide in
  `index.md`, and focused TensorBoard histograms.

### Backwards Incompatible Changes

#### Artifacts and finalization

Replace ranked `report.json` and optional `details.json` with `result.json` and Parquet history.
The JSON is an array of layer records, including selected layers with no observations. Each
observed tensor contains metric counts, endpoints, extrema, and adjacent-window averages.
There is no report byte budget or layer ranking.

During training, query completed `history.parts/*.parquet` chunks and TensorBoard events.
`result.json` initially contains the layer catalog. Call `remove_observer(model)` in a `finally`
block to finalize `history.parquet` and its JSON aggregation. Completed chunks remain readable
after interrupted execution. Window averages describe sample statistics, not pooled tensors.

#### Configuration and extension points

Remove `ReportConfig`, `IndicatorConfig`, `Aggregator`, `LiveAggregator`, `MetricLogger`,
`MetricLoggerSink`, and `merge_rank_reports()`. Remove `DirectorySink` finding-rule,
aggregator-factory, report-config, and `write_full_details` arguments. Replace report configuration
with `HistoryConfig(buffer_rows=..., window=...)` through `history_config`.

Custom sinks supplied to `inject_observer()` now receive events alongside required directory
outputs; a supplied `DirectorySink` owns the destination itself. `TensorBoardSink` writes
histograms only and accepts a writer directly or a logger exposing `.experiment`. External
writers remain caller-owned. Rank-private directories remain supported; compare their data
explicitly instead of calling the removed ranking merger.

#### Default measurements and dependencies

Reduce the default profile to mean, population standard deviation, min, max, p25, p50, p75,
zero fraction, and nonfinite fraction. Default metric names use `min`, `max`, `p50`, and
`nonfinite_fraction`; update consumers of the previous names and richer default profile.
Custom scalar reducers remain supported.

Make Polars and TensorBoard runtime dependencies. Histograms run on every sampled forward and
observed backward for the first eight selected modules by default. Configure
`histogram_selector` and `max_histogram_modules` to choose the focus before reduction.
At least one scalar reducer and one histogram reducer are required. Transient telemetry schema
advances to version 6 and records the histogram module selection.

### Documentation

- Add `examples/find_problems.py` with controlled inactive-ReLU, sigmoid-saturation, growing-gain,
  nonfinite-value, and detached-branch experiments. Each generates baseline/broken/fixed artifacts,
  Polars evidence tables, comparison plots, and an explanation of the intervention.
- Rewrite the LLM guide and research workflows around querying measurements, distinguishing
  missing evidence, and testing a proposed cause through a controlled comparison.

### Developers

- Validate all-layer coverage with 1,000 modules while restricting actual histogram collection
  to the configured focus. Cover buffer bounds, delayed backwards, shared calls, invalid values,
  writer ownership, and unchanged model outputs and gradients.
- Use explicit record types and exhaustive pattern-match fallbacks. Package the release as a
  portable pure-Python wheel; CUDA performance and `torch.compile` compatibility remain unclaimed.

## [0.6.0] - 2026-08-14

### Highlights

- Replace the default exhaustive live-state file with byte-bounded `report.json` and `index.md`
  artifacts containing category-ranked research findings and exact supporting evidence.
- Add multiprocess-safe distributed ownership without a database, shared writer, lock, or barrier.

### Backwards Incompatible Changes

#### Default storage

`DirectorySink` no longer writes `stats.json` by default. Consumers read the typed `report.json`
schema, which contains selected top findings rather than every live series. Complete live state is
available only through explicit `DirectorySink(..., write_full_details=True)` as `details.json`.

#### Distributed default

`inject_observer()` now defaults to `rank_policy="rank0"`. A detected nonzero rank registers no
hooks and writes no files. Use `rank_policy="all"` to instrument every worker in isolated rank
directories.

### New Features

- Add `ReportConfig` with an exact default 256,000-byte JSON budget and per-category top-K limits.
- Add independent rules for activation-scale drift, gradient-scale change, heavy-tail growth,
  non-finite values, zero-fraction growth, volatility, oscillation, and regime changes.
- Preserve first, latest, minimum, and maximum points plus named evidence in every typed finding.
- Add `merge_rank_reports()` to stream rank-local reports into bounded `global-report.json` and
  `global-index.md` while recording rank completeness and source truncation.

### Performance

- Prevent large model telemetry from becoming an unbounded persisted document or costly default
  LLM input.
- Avoid all observer hook and reduction overhead on nonzero ranks under the default rank policy.

### Documentation

- Rewrite the README, LLM guide, design, research workflows, and examples around ranked outcomes,
  coverage limits, controlled experiments, and distributed analysis.

### Developers

- Define persisted report records with frozen dataclasses and NamedTuples and parse external JSON
  through TypedDict boundaries.
- Add regressions for exact byte budgets, deterministic evidence, rank isolation, rank-zero
  disabling, and bounded global merging.

## [0.5.0] - 2026-08-14

### Highlights

- Replace per-sample JSON files with one atomically updated `stats.json` containing live per-layer
  distribution and temporal indicators.
- Distinguish skew, heavy tails, monotonic drift, oscillation, volatility, and regime changes that
  mean and standard deviation alone cannot reveal.

### Backwards Incompatible Changes

#### Storage and sink lifecycle

Versions through 0.4 wrote `run.json`, `modules.json`, and `snapshots/*.json`. Version 0.5 writes
one canonical `stats.json` plus a derived `index.md`; consumers must read live layer summaries
instead of enumerating sample files. Custom sinks now implement `observe(SampleRecord)` instead of
`write_snapshot(SnapshotRecord)`, and transient identifiers are named sample IDs.

#### Histogram cadence

Rename `histogram(every_n_snapshots=...)` to `histogram(every_n_samples=...)`. Fixed-bin histograms
are merged in live JSON; dynamic histograms retain their latest value and report why an aggregate
becomes unavailable after edges change.

### New Features

- Add quantiles, skewness, excess kurtosis, sign and zero prevalence, norms, tail ratios, entropy,
  and effective magnitude support to the default sampled tensor profile.
- Add configurable fast/slow EMAs, momentum horizons, slope and `R²`, exponentially weighted
  volatility, z-scores, drawdown/runup, directional balance, CUSUM, autocorrelation, oscillation,
  and directional-run indicators.
- Add `IndicatorConfig`, `Aggregator`, and `LiveAggregator` as public bounded-analysis extension
  points.
- Preserve sample locations for first, latest, minimum, and maximum values without retaining the
  complete time series.

### Performance

- Bound recent windows, metric series, tensor paths, module-call positions, histogram identities,
  and distinct error summaries independently of training duration; expose drop counts in live
  telemetry.

### Documentation

- Rewrite the README, LLM guide, research workflow, examples, and generated `index.md` around
  single-file live analysis and evidence-constrained interpretation.

### Developers

- Add regressions proving skew detection, linear-drift indicators, oscillation detection, exact
  fixed-histogram merging, live forward/backward updates, and bounded dynamic-series state.

## [0.4.0] - 2026-08-14

### Highlights

- Observe models that mix normal `module(...)` dispatch with literal `module.forward(...)` calls.

### New Features

- Add opt-in `capture_direct_forwards=True` capture for the root and recursively selected modules.
- Record the active invocation-capture strategy in `run.json` and the generated `index.md`.

### Compatibility

- Keep native PyTorch hooks as the default and restore prior instance-level forward overrides when
  direct-forward capture is removed.
- Advance the telemetry schema to version 3 for invocation-capture metadata.

### Developers

- Add a root/leaf invocation matrix proving output, gradient, state, single-record, and cleanup
  equivalence across `module(...)` and `module.forward(...)`.

## [0.3.0] - 2026-08-14

### Highlights

- Add opt-in activation and output-gradient histograms that remain fully reproducible from the
  canonical JSON telemetry.
- Generate a bounded `index.md` in every run directory so humans and LLMs can understand and
  analyze a run without prior knowledge of the schema.

### New Features

- Add configurable fixed-range or dynamic histograms with independent every-N-snapshots sampling.
- Record bin edges and counts, underflow and overflow, finite prevalence, extrema, and compact
  moments without persisting raw tensors.
- Add `TensorBoardSink` for scalar and pre-aggregated histogram projection through externally owned
  Lightning-compatible loggers.
- Record monitored signals and built-in reducer configuration in `run.json`, while identifying
  opaque custom reducers by type.

### Documentation

- Rewrite the README around diagnostic outcomes, concrete next experiments, and the simple
  attach-train-analyze workflow.
- Extend the real MNIST example with JSON-backed TensorBoard histograms.

### Developers

- Add unit and real TensorBoard event coverage proving histogram events are derived from the same
  normalized records written to JSON.

## [0.2.0] - 2026-08-14

### Highlights

- Add optional scalar-logger output alongside canonical JSON for live research dashboards.

### New Features

- Add `MetricLoggerSink` for Lightning-compatible and custom `log_metrics()` implementations.
- Add `CompositeSink` for lossless JSON and scalar logger fan-out from the same snapshots.

### Documentation

- Add a tested Lightning MNIST example with task accuracy, `TensorBoardLogger`, and structured
  telemetry from a real dataset workflow.
- Add research-diagnosis and LLM-analysis workflows that separate observations, hypotheses,
  missing evidence, and next experiments.

### Developers

- Add Lightning, TensorBoard, and torchvision as development-only dependencies for real
  integration coverage.

## [0.1.1] - 2026-08-14

### Documentation

- Add a runnable training example that produces forward and output-gradient snapshots.
- Add a quick start, concrete use cases, monitoring boundaries, configuration recipes, and
  telemetry interpretation guidance to the project README.

### Developers

- Include runnable examples in Ruff and Pyrefly validation.

## [0.1.0] - 2026-08-14

### Highlights

- Add passive, trainer-agnostic telemetry that requires no training-loop callbacks.

### New Features

- Add time-based, periodic-forward, and always-on sampling policies.
- Add leaf-module output and correlated output-gradient statistics.
- Add strict, atomically updated JSON run metadata and snapshots.
- Add explicit observer detection, duplicate-injection errors, and cleanup.

### Documentation

- Document lifecycle semantics, extension boundaries, limitations, and the project roadmap.
- Publish citation guidance and the MIT License.

### Developers

- Support and test Python 3.11 through 3.14.
- Add Ruff, Pyrefly, pytest, and declarative `dirty-equals` assertions.
