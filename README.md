# TorchInstruments

**Turn “accuracy stalled” into a small, evidence-backed list of model problems to investigate.**

Loss and accuracy curves say that a run changed. TorchInstruments observes selected-module
activations and output gradients, ranks suspicious internal behavior locally, and writes a bounded
research report while training continues normally.

```text
inject once  →  train normally  →  read a bounded report  →  run a narrower experiment
```

## Quick start

```python
from torchinstruments import inject_observer

inject_observer(model, output_dir="stats")
train(model)
```

There is no observer call inside the training loop. The default output is deliberately small:

```text
stats/
    index.md       # Human-readable findings and analysis prompt
    report.json    # Typed LLM input, at most 256 KB by default
```

No database, binary event format, raw tensor, per-sample file, or exhaustive 200 MB JSON document
is part of the default workflow.

## What you get

Suppose a model modification hurts validation accuracy. The report can provide evidence such as:

> **Gradient scale change #1:** `encoder.blocks.7.proj`, call 0, `grad_output.rms` fell from
> `0.0081` to `0.0002`. Relative movement, EMA divergence, momentum, and drawdown all rank this
> path above the other observed gradients. The series has completed warm-up.
>
> **Measured interpretation:** gradient scale weakened at this observed boundary.
>
> **Next experiment:** restore the previous normalization or residual scale at block 7 only,
> while preserving seed, data order, and precision.

TorchInstruments narrows the hypothesis space. It does not claim that correlation proves why the
task metric changed.

## The report is ranked before the LLM sees it

Sending a 200 MB telemetry file to an LLM can cost tens of millions of tokens. TorchInstruments
therefore performs deterministic searching and ranking in Python. Independent categories include:

- activation-scale drift;
- output-gradient scale change;
- heavy-tail and outlier growth;
- non-finite values;
- zero-fraction growth;
- relative volatility;
- oscillation;
- CUSUM regime-change evidence.

Each finding contains the exact module, call index, signal, tensor path, metric, first/latest and
extreme measurements, warm-up status, category-specific ranking basis, and supporting indicators.
There is no opaque combined health score.

Coverage fields report how many modules, tensor paths, temporal series, and histograms were
observed; how many findings were returned or omitted; and whether byte or collection limits removed
evidence.

## Analyze with an LLM

Give the LLM only `stats/index.md` and `stats/report.json`:

```text
Analyze the ranked TorchInstruments findings in report.json. For every material finding,
cite the exact category, module, call index, signal, tensor path, metric, values, and evidence.
Separate measured interpretation from plausible mechanisms. State missing evidence and propose
the smallest controlled experiment. Treat warmup_complete=false as weak temporal evidence.
Do not infer losses, labels, optimizer updates, inputs, or parameter gradients that were not
observed.
```

The generated `index.md` contains this prompt and a compact human rendering of the strongest
findings, so the report remains useful without an LLM.

## What is measured

| Boundary | Default behavior |
| --- | --- |
| Sampling | First root forward after each 60-second monotonic interval |
| Modules | Leaf modules, avoiding redundant container outputs |
| Forward | Tensor leaves in selected-module outputs |
| Backward | Gradients with respect to differentiable selected-module outputs |
| Distribution | Scale, quantiles, skewness, kurtosis, tails, signs, zeros, and entropy |
| Temporal behavior | EMA, momentum, slope, volatility, extrema, CUSUM, and oscillation |
| Persistence | Bounded UTF-8 JSON and Markdown reports |
| Errors | Warn and retain a bounded diagnostic summary |

The current release does not measure module inputs, `grad_input`, parameters, parameter gradients,
losses, optimizer state, or optimizer updates.

## Configure the report budget

```python
from torchinstruments import ReportConfig, inject_observer

inject_observer(
    model,
    output_dir="stats",
    report_config=ReportConfig(
        max_bytes=128_000,
        top_k_per_category=10,
    ),
)
```

The byte limit is enforced against the exact indented UTF-8 JSON written to disk. Findings are
selected round-robin across categories so one diagnostic question cannot consume the entire
budget. Omitted counts remain visible.

## Distributed training

The default `rank_policy="rank0"` instruments and writes only rank zero. Nonzero ranks register no
hooks and perform no telemetry reductions or filesystem writes.

When per-rank anomalies matter:

```python
inject_observer(model, output_dir="stats", rank_policy="all")
```

Every rank owns human- and LLM-readable files under an isolated directory:

```text
stats/
    rank-000/index.md
    rank-000/report.json
    rank-001/index.md
    rank-001/report.json
```

There are no shared writers, databases, or file locks. After rank reports exist, merge them without
loading all reports at once:

```python
from torchinstruments import merge_rank_reports

merge_rank_reports("stats")
```

This writes bounded `global-report.json` and `global-index.md`. The merged report states which
ranks were present and whether any source report was truncated. The merger does not introduce a
distributed barrier or assume every worker has finished.

## Direct `forward()` calls

Normal `module(...)` execution uses native PyTorch hooks. If model code literally calls
`module.forward(...)`, enable reversible direct-forward capture:

```python
inject_observer(model, capture_direct_forwards=True)
```

The root and recursively selected modules are observed exactly once across mixed invocation
styles. `remove_observer(model)` restores previous instance attributes.

## Histograms, TensorBoard, and custom loggers

Histograms remain opt-in because they are more expensive than scalar reductions:

```python
from torchinstruments import histogram, inject_observer

inject_observer(
    model,
    histograms=[
        histogram(
            bins=64,
            value_range=(-8.0, 8.0),
            every_n_samples=10,
        ),
    ],
)
```

`TensorBoardSink` and `MetricLoggerSink` project transient measurements to externally owned
loggers. The tested Lightning MNIST example uses the same logger for task metrics and internal
telemetry. Logger ownership remains with the caller.

Exhaustive live details are intentionally not written by default. Researchers who explicitly need
every current tensor path for local debugging can construct
`DirectorySink("stats", write_full_details=True)`. This creates `details.json`, can become very
large, and should not be sent wholesale to an LLM.

## Examples and research workflow

```bash
git clone https://github.com/Red-Eyed/torchinstruments.git
cd torchinstruments
uv sync --dev
uv run examples/basic_training.py
```

The [examples](https://github.com/Red-Eyed/torchinstruments/tree/main/examples) include an ordinary
training loop and a real Lightning MNIST workflow with TensorBoard. See the
[LLM analysis guide](https://github.com/Red-Eyed/torchinstruments/blob/main/docs/llm-analysis.md)
and [research workflows](https://github.com/Red-Eyed/torchinstruments/blob/main/docs/research-workflows.md)
for controlled baseline-versus-candidate investigations.

## Safety and compatibility

- Injection adds no parameters, buffers, or modules; `state_dict()` remains unchanged.
- Outputs and gradients remain bit-identical in the test suite.
- Unsampled callbacks perform only a cheap context lookup.
- Raw activations and gradients are never persisted.
- Report size, finding count, errors, temporal series, tensor paths, calls, and histograms have
  explicit limits.
- Python 3.11–3.14 and PyTorch 2.0+ are declared.
- CUDA-performance, Accelerate, and `torch.compile` support remain unclaimed until dedicated tests
  exist.

The core wheel depends only on PyTorch and the Python standard library. Lightning, TensorBoard,
torchvision, Dirty Equals, Ruff, Pyrefly, and pytest are development/example dependencies.

## License

TorchInstruments is released under the
[MIT License](https://github.com/Red-Eyed/torchinstruments/blob/main/LICENSE).

## Citation

If TorchInstruments supports your research or engineering work, cite it as:

```bibtex
@software{stupakov_2026_torchinstruments,
  author  = {Vadym Stupakov},
  title   = {TorchInstruments: Passive PyTorch Model Telemetry},
  year    = {2026},
  version = {0.6.0},
  url     = {https://github.com/Red-Eyed/torchinstruments}
}
```
