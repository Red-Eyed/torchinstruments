# Research workflows

TorchInstruments helps turn an underperforming model into testable hypotheses. It observes internal
training signals; it does not infer task quality, inspect data, or know which architectural change
will improve validation accuracy.

The reliable research loop is:

```text
research question
    → comparable telemetry
    → observed signature
    → plausible causes
    → missing evidence
    → smallest discriminating experiment
```

## Start with a falsifiable question

Prefer a specific question such as “Why did validation accuracy stop improving after the learning
rate change?” over “What is wrong with the model?” Record the validation metric, checkpoint or
training interval, and the exact model revision outside TorchInstruments. The observer does not
collect those task-level facts automatically.

## Collect comparable runs

Use the same selector, reducers, sampler, data slice, precision, and snapshot schedule for a
baseline and a candidate. Change one research variable at a time. Separate output directories keep
the evidence independent:

```python
inject_observer(baseline, output_dir="stats/baseline")
inject_observer(candidate, output_dir="stats/candidate")
```

When two runs use different module names or sampling schedules, comparisons may still inspire a
hypothesis, but they are not controlled evidence.

## Translate signatures into experiments

| Observed telemetry signature | Plausible causes | Evidence still missing | Useful next experiment |
| --- | --- | --- | --- |
| Gradient RMS drops sharply after one module | Saturation, detached path, poor residual scaling, excessive depth | Optimizer settings, input gradients, parameter gradients | Inspect the boundary module and test one normalization or residual-scaling change |
| Gradient RMS grows rapidly with depth | Unstable initialization, amplification through residual paths, learning rate too high | Parameter scale, optimizer updates, loss-scale behavior | Lower the learning rate or change initialization while holding data fixed |
| Activation RMS drifts across snapshots | Distribution shift, normalization mismatch, unstable residual accumulation | Batch composition, running statistics, input scale | Compare fixed evaluation batches and inspect normalization state |
| `finite_fraction` first drops below `1.0` at one output | Overflow, invalid arithmetic, non-finite upstream input | Module inputs and operation-specific state | Reproduce with a sampled batch and instrument or inspect the preceding boundary |
| `max_abs` is large relative to RMS | Rare outliers dominate dynamic range | Quantiles and per-channel ranges | Add targeted quantile/per-channel reducers when available or inspect that layer offline |
| Telemetry is stable while accuracy remains poor | The bottleneck may be outside observed optimization dynamics | Labels, data quality, loss suitability, capacity, evaluation protocol | Audit task-level evidence instead of changing internal scale blindly |

These rows are hypothesis generators, not universal diagnoses. A signature can have several causes,
and absence of a signature does not prove the model or data is correct.

## Compare observations, not filenames

Snapshot IDs count sampled root forwards. They are not optimizer steps, epochs, or dataset indices.
Compare snapshots only when the sampling conditions make them meaningfully aligned. With a
Lightning logger, TensorBoard uses the same snapshot IDs for telemetry series; Lightning's own
optimizer-step metrics may use a different step domain.

## Report conclusions with evidence levels

A useful research note separates four items:

1. **Observation** — a value directly present in telemetry.
2. **Hypothesis** — a mechanism that could explain the observation and accuracy behavior.
3. **Missing evidence** — data required to distinguish that hypothesis from alternatives.
4. **Next experiment** — the smallest controlled change or measurement that could falsify it.

This structure works for human review and for LLM-assisted analysis. See
[LLM-assisted analysis](llm-analysis.md) for a bounded file set and prompt template.
