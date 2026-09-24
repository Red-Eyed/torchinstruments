# Reliable model telemetry

Author: Vadym Stupakov <vadim.stupakov@gmail.com>
Created: 2026-09-24
Status: Implemented for 0.9.0 with documented checkpoint and performance boundaries
Authoritative URL: https://github.com/Red-Eyed/torchinstruments
Baseline: v0.8.0; see [current design](design.md).

## Objective and scope

Make the existing measurements trustworthy across common model execution patterns before
expanding the diagnostic surface or claiming production-scale performance.

The review reproduced four problems despite all 66 existing tests passing: incorrect upstream
gradient statistics after an in-place ReLU; default quantile failure on 16,777,217 finite values;
float64 histogram operations rejected by MPS; and missing internal gradient telemetry with
reentrant activation checkpointing. The review also identified mixed train/eval histories,
delayed first sampling, and unmeasured collection/finalization costs.

Preserve the attach/run/remove workflow, all four required artifacts, Polars for table processing,
per-layer JSON, explicit absence, and histogram selection before collection. Keep source tensors
out of persistence. Preserve model outputs, gradients, RNG state, and checkpoint contents.

Do not add scores, diagnoses, inferred optimizer steps, automatic histogram-layer selection,
or per-channel metrics in this work. Do not silently subsample tensors or change exact quantiles
into approximations. Compiler compatibility remains outside the supported contract until tested.

## Design checks

- **S:** Keep capture identity, numerical reduction, storage, and presentation separate. A
  histogram failure must not invalidate successful scalar measurements.
- **O:** Put alternative quantile algorithms and backend-specific numerical handling in focused
  units behind existing reducer boundaries. Avoid threading storage or reporting flags through
  capture callbacks. Adding required measurement identity is an explicit schema change.
- **L:** Preserve reducer and sink contracts, including error policies and resource ownership.
- **I:** Reuse existing small protocols; add no interface solely for one implementation.
- **D:** Keep clocks and sinks injectable. Backend choices belong at the numerical boundary,
  not in observer orchestration or the JSON renderer.

Repeat these checks against the chosen implementation before editing runtime code. Each stage
starts with a failing regression demonstrating its failure mode.

## Stage 1 — Correct and independently preserve measurements

### 1.1 Bind gradients before downstream in-place mutation

Files: `observer.py`, with a focused capture helper if needed; new
`tests/test_gradient_capture.py`; existing shared-call and lifecycle tests.

Prototype gradient registration at the selected module's output boundary, retaining the exact
sample, module call, and tensor path. Rework completion bookkeeping so partial/unused branches
and delayed backwards do not duplicate or misassign observations. Do not clone or replace model
outputs to make observation easier. Preserve the documented first-backward-only behavior.

Acceptance: identity Linear followed by ReLU reports an upstream gradient mean of 0.5 for input
[-1, 1], both in-place and out-of-place. Match independent reference gradients for branches,
shared modules, aliases, multiple outstanding forwards, and unused outputs. Removal detaches
pending hooks; abandoned graphs are reclaimable; outputs and parameter gradients are unchanged.

### 1.2 Isolate scalar and histogram failures

Files: `measurement.py`, `reducers/statistics.py`, `reducers/histograms.py`, `records.py`,
`observer.py`, and reducer/capture tests.

Keep successful measurements when another measurement fails. Represent unavailable measurements
with reasons and preserve their collection errors in the reading guide. Honor warn/ignore/raise
consistently; do not convert unexpected exceptions into plausible numbers. Explicit raise still
propagates the failure. Ensure gradient binding does not depend on all output reducers succeeding.

Acceptance: a failing histogram leaves scalar history intact under warn/ignore; a failing quantile
does not discard mean, range, or nonfinite prevalence. Empty/all-invalid tensors remain distinct
from zero-valued tensors. Failure reasons survive through final artifacts.

### 1.3 Support large tensors and MPS numerics

Files: focused reducer helpers under `reducers/`, existing reducer wiring, and reducer tests.

Evaluate an exact order-statistic selection implementation for tensors beyond the installed
quantile limit, preserving linear interpolation and finite-value semantics. Compare its memory
and latency with the existing implementation before choosing a threshold. Do not average chunk
quantiles: that would change the statistic.

Remove mandatory device float64 operations from MPS histograms. Preserve integer counts and
compute moments through backend-supported operations with documented numerical tolerances;
transfer only compact results to CPU. Make unrepresentable moments explicitly unavailable.

Acceptance: tensors above 2^24 elements produce the default metric set; quantiles match exact
references for structured distributions, ties, constants, and nonfinite inputs. Native MPS
collection writes scalar history and replayable histogram events. Counts include every finite
entry, and ordinary CUDA/CPU behavior remains covered. Test rare nonfinite values and extreme
magnitudes to expose rounding and overflow, not just small random tensors.

## Stage 2 — Make coverage and comparison semantics explicit

### 2.1 Establish activation-checkpointing coverage

Files: `observer.py`, capture helpers as necessary, `tests/test_checkpointing.py`, and guides.

Exercise reentrant and non-reentrant checkpointing, including recomputation, shared calls, and
multiple outstanding forwards. Prototype a way to correlate recomputation with its originating
sample without guessing from arrival order. Record original forward statistics once.

Acceptance: every supported checkpoint mode matches an uncheckpointed reference for activation
and output-gradient statistics without duplicate samples or changed parameter gradients. If a
mode cannot be reliably correlated through the available boundaries, document it as unsupported
and expose detectable coverage gaps. Do not claim missing gradients prove disconnection, or
claim general checkpoint support on the strength of the non-reentrant case alone.

### 2.2 Separate training and evaluation

Files: `records.py`, `observer.py`, `history/records.py`, `history/parquet.py`,
`history/summary.py`, `sinks/directory.py`, `sinks/tensorboard.py`, and schema tests.

Capture a typed train/eval mode at each selected module invocation, plus whether gradient
recording is enabled. Bind this context to delayed backward observations. Use the context in
history and summary identities and TensorBoard tags; preserve the per-layer JSON organization.
Do not infer mode from the presence of gradients: eval can run with gradients, and training
can run under no_grad. Account for mixed-mode models such as frozen BatchNorm modules.

Acceptance: train -> eval -> train yields separate summaries/tags; eval-with-grad and
train-without-grad are represented accurately; later mode changes do not relabel earlier
backwards. Old histories lacking context must never be silently classified as training.

### 2.3 Make short runs observable

Files: `sampling/timed.py`, `tests/test_sampling.py`, and sampling documentation.

Proposed behavior: sample the first eligible root forward immediately, then apply the configured
interval. Preserve explicit EveryNForwardsSampler and AlwaysSampler behavior. Test with injected
clocks, including interval boundaries, rather than sleeps.

Acceptance: a short default evaluation produces measurements; later samples respect the interval.
Document this as a sampling behavior change.

## Stage 3 — Measure cost, then optimize the demonstrated bottleneck

Add a reproducible benchmark entry point under `benchmarks/` using project CLI conventions.
Compare observer-free execution, attached-but-unsampled execution, and sampled execution on
representative convolutional and sequence models. Include large activations, low precision,
thousands of modules, and long synthetic scalar histories. Use explicit workload sizes and
record package versions, hardware, configuration, and raw benchmark results.

Measure forward/backward latency, throughput, peak device memory, peak process RSS, Parquet
chunk count/bytes, and removal latency. Synchronize accelerator timings and separate warm-up
from measurement. Run CPU and available MPS locally; CUDA results require a CUDA machine.
Do not interpret skipped accelerator checks as passes.

Use the results to choose changes in `measurement.py`, reducers, `history/parquet.py`,
`history/summary.py`, or `sinks/directory.py`. Candidates include repeated tensor preparation,
per-tensor transfers, small synchronous writes, and final aggregation memory. Do not introduce
background queues before measurements justify their ownership, backpressure, and failure costs.

Acceptance: publish baseline and changed measurements on identical workloads. Verify that
completed chunks remain queryable, interrupted runs retain published chunks, and finalization
preserves all rows and aggregates. Exact history quartiles and per-series ordering must be
included in memory measurements; a streaming Polars setting is not itself a memory guarantee.

## Decisions and delivery gates

Stages 1–2 and these behavior changes were approved for implementation:

- Keep exact tensor and history quantiles by default. If exact large-tensor processing is too
  expensive, bring measured alternatives and their accuracy tradeoffs back for a decision.
- Add train/eval and gradient-recording context to the persisted identities; this requires a
  schema migration and a minor version release.
- Make the first timed sample immediate. Keep the configured interval afterward.

Checkpoint support remains an implementation feasibility gate. Hardware-specific performance
budgets must be set against measured representative workloads, not invented in advance.

Implementation outcome: stage 1 fixes and stage 2 context/sampling changes are implemented.
Non-reentrant checkpointing matches an uncheckpointed reference; reentrant internal gradient
capture remains explicitly unsupported because reliable origin correlation is unavailable at
the existing boundaries. Original no-grad context is persisted and documented. Stage 3 provides
the benchmark harness and local CPU/MPS/long-history evidence; no CUDA machine was available.
The measured overhead and native memory growth remain performance limitations, not solved
problems. No approximations or asynchronous queues were introduced.

Deliver stage 1 independently when its regressions pass. Complete stage 2 with updated schema
documentation, index.md, examples, and migration notes. Stage 3 produces benchmark evidence
before any performance claim or optimization release.

For each release, run Ruff check and format, Pyrefly, the complete pytest suite, and the relevant
device checks; build and verify a portable wheel. Update AGENTS.md's stale summary description,
README.md, the design, LLM guide, and changelog where affected. Keep versions synchronized on
main before committing. Commit/tag/push only when requested; package uploads remain the user's.
