# AGENTS.md

This file provides guidance to Codex when working with code in this repository.

## Project Overview

TorchInstruments passively collects compact telemetry from arbitrary PyTorch models. A user
injects an observer once, runs an otherwise unchanged training or inference loop, and receives
one bounded research report without trainer-specific integration.

Key terms:

- **Root forward**: one invocation of the model passed to `inject_observer()`. It is the sampling
  unit; it is deliberately not called a training step.
- **Sample event**: a transient forward or backward record delivered to sinks and then discarded.
- **Live statistics**: bounded in-memory distributions and temporal indicators accumulated across
  sample events before local diagnostic ranking.
- **Finding rule**: an independent callable that turns one typed temporal series into either no
  candidate or one category-specific candidate with exact evidence.
- **Report configuration**: limits category top-K, errors, Markdown detail, and the exact UTF-8
  byte size of the JSON report.
- **Module call**: one invocation position of a selected module. One module may have
  several calls because modules can be shared or reused.
- **Sampling policy**: a callable object that decides whether a root forward is sampled.
- **Module selector**: a predicate evaluated during injection to choose which module objects get
  collection callbacks.
- **Call capture**: the replaceable boundary that uses native hooks by default or reversible
  forward wrappers when direct `.forward(...)` calls must be observed.
- **Reducer**: a callable that detaches one tensor and returns named compact scalar diagnostics.
- **Histogram reducer**: an opt-in callable with an independent sample cadence that emits a
  complete pre-aggregated distribution record.
- **Aggregator**: bounded state that derives live temporal indicators from transient samples.
- **Sink**: the boundary that consumes sample events and owns any side effects.
- **Metric logger sink**: a lossy scalar projection that uses sample IDs as logger steps and
  leaves ownership of the supplied logger with the caller.
- **TensorBoard sink**: a scalar and histogram projection derived from normalized records; it
  never receives source tensors and never owns the supplied logger.
- **Composite sink**: an ordered fan-out that sends the same lifecycle records to multiple sinks.
- **Absent**: a typed record carrying the reason a canonical value is unavailable; it replaces
  unexplained nulls in telemetry records.

The canonical artifacts are strict UTF-8 `report.json` and a compact `index.md`. They contain
ranked findings, exact evidence, and coverage within a configured byte budget. Both use atomic
filesystem replacement; JSON rejects NaN or infinity. `details.json` is an explicitly enabled,
potentially large debugging artifact, never the default LLM input. No sampled-forward files are
persisted.

## Development Commands

Use `uv` for every Python command.

```bash
uv sync --dev                         # Create/update the project environment.
uv run pytest                         # Run the complete test suite.
uv run pytest tests/test_api.py       # Run one test module.
uv run pytest tests/test_api.py::test_forward_output_is_bit_identical  # Run one test.
uv run ruff check .                   # Lint all Python files.
uv run ruff format .                  # Format all Python files in place.
uv run pyrefly check src tests examples # Type-check package code, tests, and examples.
uv build --wheel                      # Build the portable pure-Python wheel.
uv version <version> --no-sync        # Update the package version through uv.
```

Run Ruff check, Ruff format, Pyrefly, and pytest before committing. Re-read files after Ruff
format because it rewrites them. Only the user publishes package artifacts.

## Architecture

`inject_observer()` resolves convenience arguments into replaceable components and constructs the
imperative `Observer` shell. The observer selects modules once, initializes the sink, and attaches
one `CallCapture` strategy. A context-local sample builder lets selected-module callbacks make
one cheap context lookup on unsampled execution and isolates nested or concurrent root forwards.

During a sampled forward, tensor-tree traversal finds supported tensor leaves and reducers perform
device-local reductions. Only compact scalar tensors are transferred to CPU. The root capture
boundary emits forward measurements and registers one graph-local multi-gradient hook. That hook
binds output gradients to the exact forward context; a module-global backward hook cannot safely
provide that correlation when several forwards are outstanding.

Core abstractions are structural `Protocol` types:

- `SamplingPolicy` in `sampling/base.py` decides at root-forward boundaries.
- `CallCapture` in `capture.py` attaches native hooks or reversible forward wrappers.
- `Aggregator` in `aggregation/base.py` folds transient events into bounded live records.
- `LiveAggregator` in `aggregation/live.py` calculates distribution and temporal indicators.
- `FindingRule` in `reporting/rules.py` evaluates one series independently.
- `build_report()` in `reporting/builder.py` performs deterministic category ranking under an
  exact byte budget.
- Frozen records and NamedTuples in `reporting/records.py` define the persisted report schema.
- `ModuleSelector` in `selectors/base.py` selects module objects during attachment.
- `Reducer` in `reducers/base.py` produces named scalar reductions without inheritance.
- `HistogramReducer` in `reducers/histograms.py` produces independently sampled distributions.
- `Sink` in `sinks/base.py` consumes transient events and can be replaced independently.
- `MetricLoggerSink` projects only scalar statistics; `TensorBoardSink` derives scalars and
  histograms from the same records that `DirectorySink` serializes.
- Frozen dataclasses in `records.py` define transient and live schemas before serialization.

`rank_policy="rank0"` attaches no hooks on nonzero ranks. `rank_policy="all"` writes only to
rank-private directories. `merge_rank_reports()` streams bounded rank reports without sharing a
writer, lock, database, or distributed barrier.

The observer is intentionally attached as a plain private Python attribute. Never register it as
an `nn.Module`, parameter, or buffer, because injection must leave `state_dict()` unchanged.

## Coding Conventions

- Keep component APIs trainer-agnostic. Training steps, optimizer updates, and Transformer layout
  assumptions do not belong in the generic observer.
- Prefer a callable or small protocol for extension points. Add an interface only when there is a
  real substitution boundary.
- Keep raw tensors inside the sampled hook call or graph-local registration boundary. Builders and
  persisted records hold only compact CPU-native values.
- Use `datetime` for timestamps and monotonic floating-point seconds only for interval decisions.
  Convert UTC timestamps to ISO text solely in the JSON adapter.
- Known record shapes are frozen dataclasses, NamedTuples, or TypedDicts, never bare dictionaries.
  Typed homogeneous lookup tables are allowed when keys are genuinely dynamic, such as
  extension-defined reducer names.
- Preserve unexplained absence as an `Absent` value or an `unavailable_stats` reason. Do not emit
  bare JSON nulls or non-standard NaN/Infinity tokens.
- Built-in statistics operate on finite values and report `finite_fraction` against the original
  tensor. Standard deviation, skewness, and kurtosis use population moments.
- Temporal windows, metric series, tensor paths, call positions, histogram identities, and error
  identities must remain explicitly bounded.
- Default persisted reports must honor their exact serialized UTF-8 byte budget. Keep findings
  ranked independently by category and never invent a cross-category health score.
- Distributed workers must not share writers, locks, temporary files, or databases. Preserve rank
  identity and report incomplete rank coverage explicitly.
- Histogram JSON must contain every bin, outlier count, and compact moment required to replay the
  TensorBoard event. A dashboard adapter must never recompute from or receive raw tensors.
- Capture callbacks must return without replacing module output or gradients. Forward wrappers
  must restore only attributes they still own. The `raise` error policy is the only mode allowed
  to break a valid model execution.
- Metric logger steps are telemetry sample IDs, never inferred optimizer steps. Externally
  supplied loggers remain caller-owned and must not be finalized by observer removal.
- Tests use injected pytest fixtures for reusable resources and parametrization for repeated cases.
  Include regression coverage for shared modules and multiple outstanding forwards when changing
  lifecycle logic.
- Comments document ownership, lifecycle, or external constraints. Simplify code instead of adding
  comments that narrate control flow.
- Every Python module, class, method, and function requires a concise docstring, including private
  helpers and test helpers. Ruff's `D` rules enforce this repository-wide.
