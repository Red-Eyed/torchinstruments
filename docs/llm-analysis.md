# LLM-assisted analysis

TorchInstruments maintains one self-describing `stats.json` so an LLM can inspect a complete live
summary without selecting or parsing thousands of per-sample files. The LLM is an analysis
assistant, not an oracle: indicators can rank suspicious signatures and motivate experiments, but
they cannot establish why task accuracy is low.

## Provide the evidence

For one run, provide:

- `index.md` and `stats.json`;
- the research question and task-level outcome, such as validation accuracy;
- the exact intervention when comparing model revisions.

The JSON includes run configuration, module identities, current tensor distributions, forward and
backward temporal indicators, extrema locations, histogram aggregates, and instrumentation errors.
It does not include raw tensors, examples, labels, loss, or optimizer updates.

Module names and parameter counts can still reveal proprietary architecture details. Review the
file before sending it to an external service.

## Use an evidence-constrained prompt

```text
You are analyzing TorchInstruments live telemetry from a PyTorch research run.

Research question:
<state the accuracy problem or model comparison>

Task-level evidence:
<validation metric, baseline, candidate, intervention, and relevant training interval>

Read index.md and stats.json. Analyze only measurements present in those files.

1. Summarize sampling, capture, observed modules, forward samples, backward samples,
   dropped series, and indicator warm-up.
2. Rank modules showing the strongest evidence of:
   - activation or gradient scale drift;
   - persistent momentum or a fast/slow EMA divergence;
   - high volatility or oscillation;
   - a CUSUM regime change;
   - growing skewness, kurtosis, or tail-to-RMS ratios;
   - non-finite or zero-fraction deterioration;
   - collapse from a previous maximum or growth from a previous minimum.
3. For every finding provide:
   - Observed fact: exact module, call index, tensor path, metric, indicator, and value.
   - Interpretation: what the measurement directly means.
   - Hypothesis: mechanisms consistent with the measurement.
   - Missing evidence: what is needed to distinguish those mechanisms.
   - Next experiment: the smallest controlled test that could falsify the hypothesis.
4. Treat indicators with warmup_complete=false as weak evidence.
5. State explicitly when telemetry does not explain the task-level result.

Do not claim that an optimizer update, loss, input, parameter gradient, data-quality issue,
or causal effect was observed unless it was supplied separately.
```

## Read the hierarchy correctly

The useful path is:

```text
layers
  → module name
  → call index
  → outputs or output_gradients
  → tensor path
  → latest_statistics or statistics
  → one metric and its temporal indicators
```

`latest_statistics` describes the current sampled tensor distribution. `statistics` contains
bounded temporal summaries for selected diagnostic metrics. Do not confuse standard deviation
inside the latest tensor with temporal standard deviation across sampled forwards.

Every temporal series includes first, latest, minimum, maximum, count, and warm-up state. Extreme
points retain the sample ID and timestamp where they occurred, even though the complete historical
series is not stored.

## Compare two runs

Give each file a stable label such as `baseline` and `candidate`. Ask the LLM to match modules,
call indices, tensor paths, and statistic names before comparing values:

```text
Compare baseline/stats.json with candidate/stats.json.

Rank the largest matched changes in:
- first-to-latest RMS and gradient RMS;
- temporal slope and slope R²;
- fast/slow EMA relative gap;
- volatility, oscillation, and CUSUM scores;
- skewness, excess kurtosis, p999_abs, and max_to_rms;
- finite_fraction and zero_fraction.

Separate module/path mismatches from measured differences. For every difference, state what it
suggests, what it cannot prove, and the smallest controlled experiment that tests its relevance
to validation accuracy.
```

Use the same selector, reducers, sampling policy, precision, data slice, and indicator
configuration when making controlled comparisons.

## JSON and TensorBoard have different retention

TensorBoard receives live per-sample scalar and histogram events for visual exploration.
`stats.json` retains current distributions and bounded online indicators instead of replaying the
complete event history. Both originate from the same normalized transient measurements, but the
final live JSON intentionally cannot reconstruct every historical dashboard point.
