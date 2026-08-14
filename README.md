# TorchInstruments

**Turn “accuracy stalled” into evidence about what changed inside the model.**

A loss curve says that a run is underperforming. It does not say whether gradients started
collapsing at one layer, activations developed a heavy tail, scale began drifting, or a module
entered a new unstable regime. TorchInstruments follows those internal signals while training
continues normally.

```text
inject once  →  train normally  →  stats.json updates live  →  ask a narrower question
```

## Quick start

```python
from torchinstruments import inject_observer

inject_observer(model, output_dir="stats")
train(model)
```

There is no telemetry call inside the training loop. Open `stats/index.md`, or tell a
filesystem-capable LLM:

```text
Read stats/index.md and stats/stats.json. Find the strongest evidence for activation drift,
gradient collapse, heavy-tail growth, oscillation, or a regime change. For each finding, give
the exact layer and indicators, plausible mechanisms, missing evidence, and the smallest
controlled experiment.
```

TorchInstruments supports Python 3.11 and newer. Run the complete local example with:

```bash
git clone https://github.com/Red-Eyed/torchinstruments.git
cd torchinstruments
uv sync --dev
uv run examples/basic_training.py
```

## What the result looks like

Suppose a modified model stalls below its baseline. TorchInstruments can support a result such as:

> **Observed:** `encoder.blocks.7.proj` output RMS rose from `0.82` to `1.71`. Its fast/slow EMA
> gap reached `0.083`, linear slope is positive with `R²=0.94`, `p999_abs_to_rms` doubled, and
> output-gradient RMS fell from `0.0081` to `0.0002` with a large drawdown.
>
> **What it suggests:** scale and tail growth coincide with weakening gradient flow at block 7.
> Normalization, residual scaling, or saturation there is more plausible than a model-wide
> optimizer failure.
>
> **Next experiment:** restore the previous normalization or residual scale at block 7 only,
> while holding the seed and data order fixed.

The evidence narrows the hypothesis space. It does not pretend that correlation proves cause.

## More than mean and standard deviation

Two tensor distributions can have the same mean and standard deviation while having completely
different tails. The default sampled profile therefore includes:

- location and scale: mean, standard deviation, RMS, extrema, mean absolute value, L1/L2 norms;
- shape: median, quartiles, `p01`–`p999`, skewness, excess kurtosis, and central ranges;
- tails: `p99_abs`, `p999_abs`, max/RMS ratios, and mass beyond three standard deviations;
- prevalence: finite, zero, positive, and negative fractions;
- concentration: normalized magnitude entropy and effective support;
- optional fixed-bin histograms with explicit underflow, overflow, and non-finite counts.

For important metrics such as RMS, finite fraction, skewness, kurtosis, and tail ratios,
TorchInstruments maintains technical-analysis-like indicators over sampled forwards:

- fast and slow EMAs plus their absolute and relative gap;
- momentum over several horizons;
- linear slope and `R²`;
- exponentially weighted change and volatility;
- z-score against prior behavior;
- drawdown, runup, and historical-range position;
- directional up/down balance;
- CUSUM regime-change scores;
- lag-one autocorrelation, oscillation fraction, and consecutive directional runs.

Every series records its observation count and warm-up state so an LLM can distinguish mature
evidence from a three-sample coincidence.

## One live file, not thousands of samples

```text
stats/
    index.md
    stats.json
```

`stats.json` is the single canonical telemetry record. It contains run metadata, the module
catalog, forward and backward layer summaries, current distributions, temporal indicators,
mergeable histograms, observer overhead, and bounded error summaries. It is atomically replaced
after sampled forward and backward observations, so readers never see a partial file.

No per-sample files or raw tensors are persisted. First/latest/extreme values retain their sample
IDs and timestamps. Memory used by temporal windows, metric series, and error identities is
explicitly bounded.

## What is monitored by default

| Boundary | Default behavior |
| --- | --- |
| Sampling | First root forward after each 60-second monotonic interval |
| Modules | Leaf modules, without redundant container outputs |
| Forward | Every tensor in selected-module outputs, including nested structures |
| Backward | Gradient with respect to every differentiable selected-module output |
| Distribution | Rich finite-value shape, tail, prevalence, and concentration statistics |
| Temporal analysis | Bounded trend, momentum, volatility, regime, and stability indicators |
| Histograms | Disabled; fixed ranges are recommended for exact live aggregation |
| Persistence | One strict, atomically updated `stats.json` |
| Failures | Warn by default and aggregate instrumentation errors in telemetry |

The current release does not monitor module inputs, `grad_input`, parameters, parameter gradients,
losses, optimizer state, or optimizer updates. It never calls a sampled parameter change an
optimizer update.

## Direct `forward()` calls

Normal PyTorch `module(...)` dispatch uses native hooks. If model code literally invokes
`module.forward(...)`, enable reversible direct-forward capture:

```python
inject_observer(model, output_dir="stats", capture_direct_forwards=True)
```

The root and recursively selected modules are wrapped once, so mixed `module(...)` and
`module.forward(...)` execution is observed exactly once. `remove_observer(model)` restores the
previous instance attributes.

## Histograms

Histograms are opt-in because they cost more than scalar reductions. Fixed bins are exactly
mergeable across the run:

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

Dynamic-bin histograms retain their latest distribution, but cannot be merged after their edges
change. The live JSON records that limitation explicitly.

## Lightning and TensorBoard

Use the same externally owned Lightning logger for task metrics and model telemetry:

```python
from lightning.pytorch.loggers import TensorBoardLogger

from torchinstruments import CompositeSink, DirectorySink, TensorBoardSink, inject_observer

logger = TensorBoardLogger(save_dir="logs", name="experiment")
inject_observer(
    model.network,
    sink=CompositeSink(
        DirectorySink("stats"),
        TensorBoardSink(logger),
    ),
)
```

TorchInstruments never closes the caller's logger. TensorBoard receives live per-sample events;
`stats.json` retains bounded online indicators rather than the complete dashboard history. The
tested [MNIST example](https://github.com/Red-Eyed/torchinstruments/blob/main/examples/lightning_mnist.py)
demonstrates the complete integration.

## Configure indicator windows

The default path needs no configuration. Research-specific horizons remain explicit at the sink
boundary:

```python
from torchinstruments import DirectorySink, IndicatorConfig, LiveAggregator, inject_observer

config = IndicatorConfig(
    momentum_horizons=(1, 10, 100),
    recent_window=100,
    warmup_observations=100,
)
sink = DirectorySink("stats", aggregator_factory=lambda: LiveAggregator(config))
inject_observer(model, sink=sink)
```

`max_series`, `max_tensor_paths`, `max_module_calls`, `max_histograms`, and
`max_error_summaries` provide hard bounds for dynamic model structure and error messages. The
live record counts observations omitted by each structural limit.

## Research and LLM use cases

TorchInstruments helps investigate:

- where gradient signal first weakens or amplifies;
- whether activation scale is drifting or merely oscillating;
- whether skew, kurtosis, or extreme-to-RMS ratios are growing;
- which module first emits non-finite values;
- whether an architecture change creates a new internal regime;
- whether rare outliers make a layer quantization-hostile;
- whether internal behavior is stable enough to redirect investigation toward data, loss, or
  evaluation.

See the [LLM analysis guide](https://github.com/Red-Eyed/torchinstruments/blob/main/docs/llm-analysis.md)
and [research workflows](https://github.com/Red-Eyed/torchinstruments/blob/main/docs/research-workflows.md)
for evidence-constrained prompts and baseline-versus-candidate experiments.

## Safety and compatibility

- Injection adds no parameters, buffers, or modules; `state_dict()` remains unchanged.
- Outputs and gradients remain bit-identical in the test suite.
- Unsampled callbacks perform only a cheap context lookup.
- Sampled reductions happen on the tensor device; only compact results move to CPU.
- Raw activations and gradients are never written to disk.
- Duplicate injection raises `ObserverAlreadyAttachedError`.
- Python 3.11–3.14 and PyTorch 2.0+ are declared; CUDA, Accelerate, distributed output, and
  `torch.compile` remain unclaimed until dedicated compatibility tests exist.

The core wheel depends only on PyTorch and the Python standard library. Lightning, TensorBoard,
and torchvision are development/example dependencies.

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
  version = {0.5.0},
  url     = {https://github.com/Red-Eyed/torchinstruments}
}
```
