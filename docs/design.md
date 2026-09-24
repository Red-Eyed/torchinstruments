# Per-layer PyTorch measurement history

Author: Vadym Stupakov <vadim.stupakov@gmail.com>
Created: 2026-08-14; revised: 2026-09-24
Status: Implemented and validated
Authoritative URL: https://github.com/Red-Eyed/torchinstruments

## Objective and background

Collect measurements across many model layers and retain enough history for a researcher to
investigate trends. The earlier ranked-report design replaced measurements with selected findings
and exposed too much diagnostic machinery. A histogram-only design would expose thousands of
TensorBoard series. This version combines full scalar history, per-layer summaries, and focused
histograms. All three outputs and their reading guide are required.

## Goals

- Keep the existing attach/train/remove workflow and preserve model execution.
- Record every successfully collected scalar observation and every selected layer identity.
- Use Polars for Parquet IO, table queries, aggregation, and JSON export.
- Give the LLM index.md as an entry point explaining data, sampling, and uncertainty.
- Restrict expensive histogram collection before reduction with an explicit module budget.
- Demonstrate diagnosis through controlled baseline/broken/fixed examples.

No automatic diagnoses, ranking, cross-layer health scores, causal conclusions, or inferred
optimizer steps are part of telemetry. The observer does not measure task loss or accuracy.

## Data flow

```mermaid
flowchart TD
    Capture[Sampled forward and graph-local backward] --> Measurement[Per-module measurement plan]
    Measurement --> History[Buffered Polars Parquet chunks]
    Measurement --> TensorBoard[Histograms for selected modules]
    History --> Final[Streaming history.parquet finalization]
    Final --> Summary[Polars per-layer aggregation]
    Summary --> JSON[result.json]
    Guide[index.md] --> Final
    Guide --> JSON
    Guide --> TensorBoard
```

## Storage and semantics

History is a long table with one row per statistic. Layer, call index, signal, tensor path,
sample ID, forward index, UTC timestamp, shape, and dtype establish its meaning. Values may be
nullable only alongside an unavailable reason. Default measurements are mean, population std,
min, max, p25, p50, p75, zero fraction, and nonfinite fraction.

Completed chunks can be queried during training. On removal, Polars streams chunks into one
atomic history.parquet and writes result.json. This avoids rewriting a growing monolithic file
on every event. Inference and empty runs also finalize valid artifacts. An interrupted run keeps
completed chunks. Summary finalization is synchronous and adds close-time IO and aggregation
cost; its output size grows with observed layer/call/path identities, not with run duration.

JSON is an array of layer records. Each contains tensor identities and metric summaries:
counts, first/latest, finite extrema, and two adjacent observation windows. Delayed backwards
are ordered by originating sample ID. Window means average sample statistics; they are not
pooled tensor distribution statistics. Selected but unexecuted modules have empty tensor lists.

All layer summaries are retained. Consequently, JSON is not subject to the old byte budget.
Python buffering is bounded by configured rows. Polars performs final aggregation in its
streaming engine; native query working memory still depends on grouping and ordering.

## Histogram and writer ownership

At attachment, API wiring builds one measurement plan per selected module. Default plans use
histograms on the first eight selected modules; explicit selection and limits are available.
Call capture knows only the resulting plans. Scalar history covers every selected module.
Histogram reducers transfer compact bin counts and moments, never raw tensors, to the event writer.

DirectorySink owns and closes its writer. Additional TensorBoardSink instances can wrap external
writers or Lightning loggers without closing them. Rank-zero policy avoids all work elsewhere;
all-rank policy isolates directories and preserves rank identity in each guide.

## Alternatives

- Ranked JSON: removed because interpreting evidence belongs to the researcher or LLM.
- All-layer TensorBoard histograms: too many series and unnecessary collection cost.
- Growing JSON history: repetitive and inefficient to query and rewrite.
- PyArrow Parquet implementation: Polars is the required table/storage/aggregation dependency.
- Continuous whole-history aggregation: avoided to prevent rescanning all past measurements on
  each event. Live users query Parquet chunks; final JSON is produced on close.

## Validation

Check model output/gradient invariance, shared modules, nested outputs, delayed/unused backwards,
empty and nonfinite tensors, histogram focus before reduction, buffer bounds, rank isolation,
writer ownership, and real TensorBoard replay. Verify each example's fault signature against its
fixed run. CPU tests do not establish CUDA performance or torch.compile compatibility.
