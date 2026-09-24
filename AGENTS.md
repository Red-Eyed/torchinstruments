# AGENTS.md

## Project Overview

TorchInstruments collects activation and output-gradient measurements from PyTorch models without
changing training loops. Every directory run produces history.parquet, result.json, index.md,
and focused TensorBoard histograms. JSON contains per-layer data, never diagnostic categories.
Problem definitions and controlled interventions belong only in examples and documentation.

## Inversion First

Before changes, identify how missing evidence, incomplete capture, or aggregation could mislead.
Turn material failure modes into regression tests. Do not present a measurement signature as proof
of a cause or task quality. Preserve missing values with reasons and distinguish sample IDs from
optimizer steps.

## Architecture

- `api.py` selects modules and binds per-module `TensorMeasurement` plans at attachment.
- `observer.py` owns root-forward contexts; `gradient_capture.py` binds tensor hooks before
  in-place mutation and uses an engine completion callback for first-backward publication. Shared calls,
  nested outputs, and multiple outstanding forwards must remain distinct.
- Selected children called outside a root context create independent samples. Timed deadlines
  are keyed by module path; an explicitly unsampled root must never fall back to child sampling.
  Backward engine recomputation must not create independent forward samples.
- `capture.py` provides native hooks and reversible direct-forward wrappers.
- `reducers/` computes finite-value statistics and independently sampled histograms on-device.
- `history/records.py` defines typed scalar observations. `history/parquet.py` buffers compact
  rows into completed chunks using Polars and publishes history.parquet after each sampled event.
- `history/summary.py` uses Polars expressions for history distributions and adjacent window means,
  then groups them by layer and exports result.json with the Polars serializer.
- Module mode and autograd recording context are captured at invocation and retained through
  delayed backward. History identities and dashboard tags separate these contexts.
- `isolated_measurement.py` wraps reducers with the observer's error policy so failures in one
  reducer do not discard successful siblings. Exact quartiles use selection above the size limit.
- `DirectorySink` owns Parquet, JSON, the LLM guide, and its TensorBoard writer. Extra supplied
  sinks receive events as well. Externally supplied loggers remain caller-owned.
- `index.md` explains the live chunk location, final files, schema, counts, errors, and queries.
  result.json refreshes its history aggregation after every sampled forward or backward.
- `rank_policy="rank0"` attaches nothing on nonzero ranks; `"all"` gives each rank private files.
  There are no shared writers, global scores, or rank-summary merging.

The observer is a private Python attribute, never a registered module, parameter, or buffer.

## Development Commands

Use uv for every Python command. Run before committing:

```bash
uv run ruff check src tests examples
uv run ruff format src tests examples
uv run pyrefly check src tests examples
uv run pytest
uv build --wheel
```

Re-read files after formatting. Only the user publishes package artifacts.

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
  extension-defined reducer names. Do not use Mapping annotations; use precise record types.
- Preserve unexplained absence as an `Absent` value or an `unavailable_stats` reason. Nullable serialized cells must carry an absence reason; never emit non-standard NaN/Infinity tokens.
- Built-in statistics operate on finite values and report `nonfinite_fraction` against the original
  tensor. Standard deviation uses population moments.
- Buffer Parquet writes with explicit row limits. Use Polars for table construction, Parquet IO,
  queries, aggregation, and JSON export. Never collect the raw full history into Python memory.
- JSON contains per-layer measurements only: no categories, ranks, scores, or diagnoses.
- Histogram focus is explicit and applied before reduction, not just before event writing.
- Distributed workers must not share writers, locks, temporary files, or databases. Preserve rank
  identity and report incomplete rank coverage explicitly.
- Histogram records retain every bin, outlier count, and compact moment needed by TensorBoard.
  The event adapter never receives source tensors.
- Capture callbacks must return without replacing module output or gradients. Forward wrappers
  must restore only attributes they still own. The `raise` error policy is the only mode allowed
  to break a valid model execution.
- TensorBoard steps are telemetry sample IDs, never inferred optimizer steps. Externally
  supplied loggers remain caller-owned and must not be finalized by observer removal.
- Tests use injected pytest fixtures for reusable resources and parametrization for repeated cases.
  Include regression coverage for shared modules and multiple outstanding forwards when changing
  lifecycle logic.
- Comments document ownership, lifecycle, or external constraints. Simplify code instead of adding
  comments that narrate control flow.
- Every Python module, class, method, and function requires a concise docstring, including private
  helpers and test helpers. Ruff's `D` rules enforce this repository-wide.
