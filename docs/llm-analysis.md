# LLM-assisted analysis

TorchInstruments emits descriptive JSON so an LLM can inspect telemetry without TensorBoard event
parsing or knowledge of the Python process that created it. The LLM is an analysis assistant, not
an oracle: it can rank suspicious signatures and propose discriminating experiments, but telemetry
alone cannot establish why validation accuracy is low.

## Provide a bounded evidence set

For one run, provide:

- `run.json` for schema, package version, and sampling semantics;
- `modules.json` for module types, aliases, and parameter counts;
- a small set of relevant snapshot files;
- the research question and task-level outcome, such as validation accuracy;
- the exact intervention when comparing model revisions.

For a short run, all snapshots may fit. For a long run, begin with a few meaningfully chosen points,
such as early, middle, late, and the first snapshot near a known accuracy or loss change. Do not
upload thousands of snapshots blindly. Automatic cross-snapshot summaries are roadmap work; until
they exist, snapshot selection is part of the research design.

Module names and parameter counts can reveal proprietary architecture details even though raw
tensors are never stored. Review telemetry before sending it to an external service.

## Use an evidence-constrained prompt

Copy the following prompt and attach the selected JSON files:

```text
You are analyzing TorchInstruments telemetry from a PyTorch research run.

Research question:
<state the accuracy problem or model comparison>

Task-level evidence:
<validation metric, baseline, candidate, intervention, and relevant training interval>

Analyze only claims supported by the attached run.json, modules.json, and snapshots.

1. Summarize the sampling policy and which signals were actually observed.
2. Identify the earliest modules and snapshots with suspicious activation RMS,
   output-gradient RMS, max_abs relative to RMS, or finite_fraction.
3. Separate every conclusion into:
   - Observed fact: exact module, snapshot ID, metric, and value.
   - Hypothesis: one or more mechanisms consistent with that fact.
   - Missing evidence: what telemetry or task information is needed to distinguish them.
   - Next experiment: the smallest controlled test that could falsify the hypothesis.
4. Rank findings by likely impact and confidence.
5. State explicitly when the evidence does not explain the accuracy gap.

Do not claim that an optimizer update was observed. Do not infer module inputs,
parameter gradients, loss values, data quality, or causal effects unless those were
provided separately.
```

## Compare two runs

Give each run a stable label such as `baseline` and `candidate`, and attach both run/module catalogs
plus aligned snapshots. Ask for differences using exact metric paths and values. A useful comparison
request is:

```text
For matched modules and snapshot positions, rank the largest changes in activation RMS,
output-gradient RMS, max_abs/RMS, and finite_fraction. Distinguish module-name mismatches
from measured changes. For each difference, explain what it suggests, what it does not prove,
and the next experiment that would test its relevance to validation accuracy.
```

The current built-in reducers do not emit `max_abs/RMS` directly. An LLM may calculate that ratio
from `max_abs` and `rms` values in the same tensor record, but it should preserve the source values
in its report so the calculation can be checked.

## Use JSON and TensorBoard for different jobs

TensorBoard is useful for visual trend discovery. Canonical JSON is better for LLM analysis because
it preserves module calls, shapes, dtypes, unavailable-stat reasons, errors, and snapshot lifecycle
state. When both are needed, use `CompositeSink(DirectorySink(...), MetricLoggerSink(...))` rather
than replacing structured telemetry with a scalar-only logger.
