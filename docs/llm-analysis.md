# Reading a run with an LLM

Start with the run's `index.md`. It identifies the history location, sampling policy, rank,
histogram focus, collection errors, and summary semantics. Read `result.json` as an array of
layer records, not a list of diagnosed problems.

1. Check observation counts and unavailable reasons. An unexecuted layer or missing backward is
   not a healthy measurement. During training the summary is only a catalog; query live chunks.
2. Inspect per-layer first/latest values, extrema, and previous/recent windows. Compare like
   signals, tensor paths, shapes, and call indices. Keep distributed ranks distinct.
3. Use Polars to filter `history.parquet` to the relevant measurements. During training or after
   a crash, query `history.parts/*.parquet`. Sort by sample ID, not arrival order.
4. Inspect selected TensorBoard histograms when quartiles and scale summaries leave the
   distribution unclear. Re-run with a focused histogram selector if needed.
5. State a hypothesis and test one change with matched initialization and batches. Task metrics
   must come from the caller's training evaluation; the observer does not know them.

Default distribution metrics use finite tensor entries. Check nonfinite_fraction alongside them.
Window means average the individual sample statistics and are not pooled statistics. Whole-tensor
summaries can hide channel-specific effects; they do not establish that every unit is healthy.

Example request:

> Read index.md and result.json. Identify measurements worth investigating and cite the layer,
> call, tensor path, signal, sample IDs, and values. Query the underlying Parquet history before
> describing a trend. Distinguish missing gradients from measured zero gradients. Explain what
> remains unknown and propose a controlled comparison. Do not invent optimizer-step alignment,
> parameter-gradient measurements, or task metrics.
