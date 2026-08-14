# Research workflows

TorchInstruments turns internal model behavior into ranked, testable hypotheses. It does not know
whether an architecture, dataset, loss, or optimizer is correct.

```text
research question
    → bounded local measurements
    → category-ranked findings
    → plausible causes
    → missing evidence
    → smallest discriminating experiment
```

## Start with a falsifiable question

Prefer “Why did validation accuracy stop improving after the normalization change?” over “What is
wrong with the model?” Record the validation metric, model revision, seed, and intervention outside
TorchInstruments; those facts cannot be inferred from module telemetry.

## Collect comparable runs

Use the same selector, reducers, sampler, precision, data slice, indicator configuration, finding
rules, and report limits:

```python
inject_observer(baseline, output_dir="stats/baseline")
inject_observer(candidate, output_dir="stats/candidate")
```

Compare `baseline/report.json` with `candidate/report.json`. Sampling IDs count sampled root
forwards, not optimizer steps or epochs. Time-based sampling can observe different batches, so a
fixed evaluation batch is preferable when the question requires aligned distributions.

## Translate findings into experiments

| Finding category | Plausible causes | Missing evidence | Small next experiment |
| --- | --- | --- | --- |
| Activation-scale drift | Distribution shift, normalization mismatch, residual accumulation | Inputs, batch identity, normalization state | Repeat on a fixed evaluation batch and inspect normalization |
| Gradient-scale change | Saturation, detached path, weak residual scaling, excessive depth | `grad_input`, parameter gradients, task loss | Change normalization or residual scale at one boundary |
| Heavy-tail growth | Rare outliers, saturation, a small set of dominant channels | Inputs, per-channel statistics | Add fixed-bin histograms or per-channel reducers at that layer |
| Non-finite values | Overflow, invalid arithmetic, non-finite upstream values | Inputs and operation-level tracing | Instrument the preceding boundary and enable anomaly detection |
| Zero-fraction growth | Dead activations, gating collapse, excessive sparsity | Inputs and activation-function state | Compare one activation or initialization change |
| High volatility | Alternating batch regimes, unstable normalization, feedback instability | Batch identity, task loss, running statistics | Compare fixed batches or freeze normalization statistics |
| Oscillation | Alternating regimes or update overshoot | Optimizer updates and learning-rate events | Lower the learning rate in one controlled rerun |
| Regime change | Scheduler event, data phase shift, abrupt instability | External training timeline | Align the reported sample with scheduler and data events |

## Interpret rankings conservatively

- Rank compares candidates only within the same category.
- Ranking scores from different categories use different semantics and are not comparable.
- Exact evidence is more important than the ordering score.
- `warmup_complete=false` marks immature temporal evidence.
- Correlated indicators derived from the same series are not independent confirmation.
- A finding suggests where to test; it does not identify a cause.

## Treat missing findings as unknown

The report keeps category top-K candidates under an exact byte limit. Check
`report_truncated_by_byte_budget`, `findings_omitted`, dropped-observation counters, and errors
before concluding that an unlisted layer was healthy. Increase the report budget or enable
`details.json` for a targeted offline investigation rather than making exhaustive output the
default.

## Distributed experiments

Use `rank_policy="rank0"` when rank zero is representative and minimum overhead matters. Use
`rank_policy="all"` when data shards may behave differently, then call `merge_rank_reports()` and
inspect the generated global report.

Check rank completeness before generalizing. The merger preserves each finding's source rank and
ranks rank-local evidence globally; it does not average activations across processes.

## Keep experiments controlled

Change one research variable at a time. Preserve seed, data order, precision, schedule, and
evaluation. A useful report contains exact measurements and one small experiment, not a list of
unrelated optimization advice.

## Respect evidence boundaries

The current observer measures selected-module outputs and output gradients. It does not measure
loss, labels, optimizer updates, inputs, parameters, or parameter gradients. “Gradient scale falls
after layer 7” can be supported; “the optimizer made a bad update at layer 7” cannot.
