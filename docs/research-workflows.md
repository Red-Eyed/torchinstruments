# Research workflows

TorchInstruments turns internal model behavior into testable hypotheses. It does not know whether
an architecture, dataset, loss, or optimizer is correct.

```text
research question
    → comparable live indicators
    → observed signature
    → plausible causes
    → missing evidence
    → smallest discriminating experiment
```

## Start with a falsifiable question

Prefer “Why did validation accuracy stop improving after the normalization change?” over “What is
wrong with the model?” Record the validation metric, model revision, seed, and intervention outside
TorchInstruments; those facts cannot be inferred from module telemetry.

## Collect comparable runs

Use the same selector, reducers, sampler, precision, data slice, and indicator configuration:

```python
inject_observer(baseline, output_dir="stats/baseline")
inject_observer(candidate, output_dir="stats/candidate")
```

Sampling IDs count sampled root forwards, not optimizer steps or epochs. Time-based sampling can
observe different batches in different runs, so a fixed evaluation batch is preferable when the
research question requires tightly aligned distributions.

## Translate signatures into experiments

| Observed signature | Plausible causes | Missing evidence | Small next experiment |
| --- | --- | --- | --- |
| Gradient RMS has negative momentum and a large drawdown after one module | Saturation, detached path, weak residual scaling, excessive depth | Inputs, `grad_input`, parameter gradients | Test one normalization or residual-scale change at that boundary |
| Output RMS fast EMA exceeds slow EMA with a high-`R²` positive slope | Distribution shift, normalization mismatch, residual accumulation | Batch composition, module inputs, normalization state | Repeat on a fixed evaluation batch and inspect normalization |
| RMS has low slope but high volatility and oscillation fraction | Alternating batch regimes, unstable normalization, feedback instability | Batch identity, task loss, running statistics | Compare odd/even samples or freeze normalization statistics |
| CUSUM rises while lifetime mean remains ordinary | A recent regime change hidden by older observations | Learning-rate and schedule events | Align the change sample with external training events |
| Skewness, kurtosis, and `p999_abs_to_rms` grow | Rare outliers or saturation dominate scale | Input distribution, per-channel statistics | Add fixed-bin histograms or per-channel reducers at that layer |
| `finite_fraction` falls below `1.0` | Overflow, invalid arithmetic, non-finite upstream values | Inputs and operation-level tracing | Instrument the preceding boundary and rerun with anomaly detection |
| No meaningful internal difference | Data, labels, loss, capacity, or evaluation may dominate | Task-level and data evidence | Redirect the experiment instead of tuning an arbitrary layer |

## Interpret indicators conservatively

- A large fast/slow EMA gap means recent behavior differs from longer-term behavior; it does not
  explain why.
- High momentum means persistent directional change over one configured horizon.
- High volatility with low slope suggests instability rather than drift.
- Lag-one autocorrelation near `1` indicates persistence; near `-1` indicates alternation.
- CUSUM is a change score, not a calibrated probability.
- `warmup_complete=false` means the configured history is not mature.
- Several indicators derive from the same values and are correlated evidence, not independent
  confirmations.

## Keep experiments controlled

Change one research variable at a time. Preserve seed, data order, precision, schedule, and
evaluation. A useful report contains exact measurements and one small experiment, not a list of
unrelated optimization advice.

## Respect evidence boundaries

The current observer measures selected-module outputs and output gradients. It does not measure
loss, labels, optimizer updates, inputs, parameters, or parameter gradients. A result such as
“gradient scale falls after layer 7” is supported; “the optimizer made a bad update at layer 7” is
not.
