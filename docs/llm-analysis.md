# LLM-assisted analysis

TorchInstruments does the expensive triage locally. It ranks bounded sufficient statistics with
deterministic rules and writes a compact `report.json` plus a human-readable `index.md`. An LLM can
therefore reason from the strongest evidence instead of reading a potentially enormous dump of
every layer statistic.

The LLM is an analysis assistant, not an oracle. Findings identify suspicious signatures and
motivate experiments; they cannot establish why task accuracy is low.

## Provide the evidence

For one run, provide:

- `index.md` and `report.json`;
- the research question and task-level outcome, such as validation accuracy;
- the exact intervention when comparing model revisions.

The report includes module identities, ranked diagnostic categories, exact first/latest/extreme
measurements, rule-specific evidence, warm-up state, sampling coverage, omissions, and
instrumentation errors. It does not include raw tensors, examples, labels, loss, optimizer updates,
or the exhaustive internal state.

Module names can reveal proprietary architecture details. Review the files before sending them to
an external service.

## Start with the generated prompt

Every `index.md` contains a ready-to-use prompt. A more specific version is:

```text
You are analyzing a TorchInstruments research report from a PyTorch run.

Research question:
<state the accuracy problem or model comparison>

Task-level evidence:
<validation metric, baseline, candidate, intervention, and relevant training interval>

Read index.md and report.json. Analyze only evidence present in those files.

1. Check coverage first: samples, backward samples, warm-up, dropped observations,
   report truncation, omitted findings, and instrumentation errors.
2. Within each diagnostic category, inspect the highest-ranked findings.
3. For every material finding provide:
   - Observed fact: exact module, call index, signal, tensor path, metric, and values.
   - Interpretation: what the measurement directly means.
   - Hypothesis: mechanisms consistent with the measurement.
   - Missing evidence: what would distinguish those mechanisms.
   - Next experiment: the smallest controlled test that could falsify the hypothesis.
4. Treat warmup_complete=false as weak evidence.
5. Do not compare ranking_score across different categories.
6. State explicitly when the report cannot explain the task-level result.

Do not claim that an optimizer update, loss, input, parameter gradient, data-quality issue,
or causal effect was observed unless it was supplied separately.
```

## Read the hierarchy correctly

The useful path is:

```text
findings
  → diagnostic category
  → category-local rank
  → module and call index
  → output or output-gradient signal
  → tensor path and metric
  → exact points and named evidence
```

`ranking_score` is a deterministic ordering key within one category. Activation drift and
non-finite prevalence use different formulas and units, so a score of `2.0` in one category is not
twice—or even necessarily stronger than—a score of `1.0` in another. Never add scores into a
single health value.

The `first`, `latest`, `minimum`, and `maximum` points retain sample IDs and timestamps. `evidence`
names the quantities used by that category's rule. `warmup_complete` reports whether the temporal
indicator had enough observations to mature.

## Check coverage before interpreting findings

The report is intentionally selective. Its `coverage` section distinguishes:

- measurements observed from findings returned;
- findings omitted by category top-K or byte budget;
- errors returned from errors omitted;
- structural observations dropped by bounded live-state limits;
- a report that reached its exact byte budget.

No returned finding is fabricated, but absence from a truncated report does not prove a layer was
healthy. Increase `ReportConfig.max_bytes` or opt into `details.json` for a targeted offline
investigation when coverage is insufficient.

## Compare two runs

Give each report a stable label such as `baseline` and `candidate`:

```text
Compare baseline/report.json with candidate/report.json.

Match diagnostic category, module, call index, signal, tensor path, and metric.
For each matched finding, compare exact first/latest/extreme values and named evidence.
Separate unmatched paths from measured differences. Treat findings absent from a truncated
or top-K-limited report as unknown, not unchanged.

For every difference, state what it suggests, what it cannot prove, and the smallest controlled
experiment that tests its relevance to validation accuracy.
```

Use the same selector, reducers, sampling policy, precision, data slice, indicator configuration,
finding rules, and report limits for controlled comparisons.

## Analyze distributed runs

With `rank_policy="all"`, analyze `global-index.md` and `global-report.json` after calling
`merge_rank_reports()`. Check `expected_ranks`, `ranks_present`, `rank_coverage_complete`, and
`source_reports_truncated` before generalizing across workers. Each finding preserves its source
rank.

The merger ranks already-reduced rank-local findings. It does not average tensor statistics across
ranks and does not pretend missing workers completed.

## JSON and TensorBoard have different retention

TensorBoard receives live per-sample scalar and histogram events for visual exploration.
`report.json` retains ranked conclusions and exact supporting measurements inside a strict byte
budget. The two outputs share normalized source measurements, but the final report intentionally
cannot reconstruct every dashboard point.
