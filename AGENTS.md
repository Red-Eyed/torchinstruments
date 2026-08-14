# AGENTS.md

This file provides guidance to Codex when working with code in this repository.

## Project Overview

TorchInstruments passively collects compact telemetry from arbitrary PyTorch models. A user
injects an observer once, runs an otherwise unchanged training or inference loop, and receives
versioned JSON snapshots without trainer-specific integration.

Key terms:

- **Root forward**: one invocation of the model passed to `inject_observer()`. It is the sampling
  unit; it is deliberately not called a training step.
- **Snapshot**: all measurements correlated with one sampled root forward. It is written first as
  `forward_complete` and atomically enriched to `backward_observed` if gradients arrive.
- **Module call**: one invocation of a selected module inside a snapshot. One module may have
  several calls because modules can be shared or reused.
- **Sampling policy**: a callable object that decides whether a root forward starts a snapshot.
- **Module selector**: a predicate evaluated during injection to choose which module objects get
  collection callbacks.
- **Call capture**: the replaceable boundary that uses native hooks by default or reversible
  forward wrappers when direct `.forward(...)` calls must be observed.
- **Reducer**: a callable that detaches one tensor and returns named compact scalar diagnostics.
- **Histogram reducer**: an opt-in callable with an independent snapshot cadence that emits a
  complete pre-aggregated distribution record.
- **Sink**: the side-effect boundary that initializes run metadata and persists snapshots.
- **Metric logger sink**: a lossy scalar projection that uses snapshot IDs as logger steps and
  leaves ownership of the supplied logger with the caller.
- **TensorBoard sink**: a scalar and histogram projection derived from normalized records; it
  never receives source tensors and never owns the supplied logger.
- **Composite sink**: an ordered fan-out that sends the same lifecycle records to multiple sinks.
- **Absent**: a typed record carrying the reason a canonical value is unavailable; it replaces
  unexplained nulls in telemetry records.

The canonical telemetry artifact is strict UTF-8 JSON. A run directory also contains a derived,
bounded `index.md` for human and LLM discovery, plus `run.json`, `modules.json`, and monotonically
numbered files under `snapshots/`. Snapshot and index rewrites use atomic filesystem replacement;
JSON rejects NaN or infinity during serialization.

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
one `CallCapture` strategy. A context-local snapshot builder lets selected-module callbacks make
one cheap context lookup on unsampled execution and isolates nested or concurrent root forwards.

During a sampled forward, tensor-tree traversal finds supported tensor leaves and reducers perform
device-local reductions. Only compact scalar tensors are transferred to CPU. The root capture
boundary writes the forward snapshot and registers one graph-local multi-gradient hook. That hook
binds output gradients to the exact forward context; a module-global backward hook cannot safely
provide that correlation when several forwards are outstanding.

Core abstractions are structural `Protocol` types:

- `SamplingPolicy` in `sampling/base.py` decides at root-forward boundaries.
- `CallCapture` in `capture.py` attaches native hooks or reversible forward wrappers.
- `ModuleSelector` in `selectors/base.py` selects module objects during attachment.
- `Reducer` in `reducers/base.py` produces named scalar reductions without inheritance.
- `HistogramReducer` in `reducers/histograms.py` produces independently sampled distributions.
- `Sink` in `sinks/base.py` owns persistence and can later be replaced by an asynchronous sink.
- `MetricLoggerSink` projects only scalar statistics; `TensorBoardSink` derives scalars and
  histograms from the same records that `DirectorySink` serializes.
- Frozen dataclasses in `records.py` define the normalized schema before serialization.

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
- Known record shapes are frozen dataclasses or typed structures, not bare dictionaries. Dynamic
  metric maps are allowed because reducer names are extension-defined.
- Preserve unexplained absence as an `Absent` value or an `unavailable_stats` reason. Do not emit
  bare JSON nulls or non-standard NaN/Infinity tokens.
- Built-in statistics operate on finite values and report `finite_fraction` against the original
  tensor. Standard deviation is population standard deviation with correction zero.
- Histogram JSON must contain every bin, outlier count, and compact moment required to replay the
  TensorBoard event. A dashboard adapter must never recompute from or receive raw tensors.
- Capture callbacks must return without replacing module output or gradients. Forward wrappers
  must restore only attributes they still own. The `raise` error policy is the only mode allowed
  to break a valid model execution.
- Metric logger steps are telemetry snapshot IDs, never inferred optimizer steps. Externally
  supplied loggers remain caller-owned and must not be finalized by observer removal.
- Tests use injected pytest fixtures for reusable resources and parametrization for repeated cases.
  Include regression coverage for shared modules and multiple outstanding forwards when changing
  lifecycle logic.
- Comments document ownership, lifecycle, or external constraints. Simplify code instead of adding
  comments that narrate control flow.
- Every Python module, class, method, and function requires a concise docstring, including private
  helpers and test helpers. Ruff's `D` rules enforce this repository-wide.
