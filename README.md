# TorchInstruments

**Turn “validation accuracy stalled” into a concrete next experiment.**

A loss curve tells you that a run is underperforming. It rarely tells you whether gradients vanish
at one block, activations drift over time, values become non-finite before the loss does, or a
distribution is dominated by rare outliers. TorchInstruments records that internal evidence while
the model trains normally.

```text
inject_observer(model)  →  train normally  →  inspect evidence  →  test a narrower hypothesis
```

## The whole workflow

```python
from torchinstruments import inject_observer

inject_observer(model, output_dir="stats")
train(model)
```

Training remains unchanged. After the run, tell an LLM:

```text
Read stats/index.md and analyze why validation accuracy may have stalled.
Give me measured evidence, plausible hypotheses, missing evidence, and the next experiment.
```

`index.md` tells the LLM what else to read and what the telemetry can and cannot support. You do
not need to explain the snapshot schema by hand. If TensorBoard output is enabled, the same run also
shows layer curves and optional distributions for visual exploration.

## What a useful result looks like

Suppose a modified model stalls below its baseline. Instead of “try another learning rate,” the run
can support a result like this:

> **Observed:** gradient RMS remains comparable through blocks 1–6, then drops by orders of
> magnitude at `encoder.blocks.7.proj`. The collapse repeats in later sampled backward passes.
>
> **What it suggests:** the optimization path first becomes weak at block 7; normalization,
> residual scaling, or saturation there is more plausible than a model-wide optimizer failure.
>
> **Next experiment:** restore the previous normalization or residual scale at block 7 only, keep
> the seed and data order fixed, and test whether both gradient flow and validation accuracy recover.

Other runs may instead reveal drifting activation distributions, non-finite values appearing at a
specific module, or rare outliers that make one layer difficult to quantize. Finding no meaningful
internal difference is useful too: it redirects the investigation toward data, labels, loss,
capacity, or evaluation.

The measurements narrow the hypothesis space; they do not pretend that correlation proves cause.

## Quick start

TorchInstruments is an alpha package supporting Python 3.11 and newer. Run the complete example:

```bash
git clone https://github.com/Red-Eyed/torchinstruments.git
cd torchinstruments
uv sync --dev
uv run examples/basic_training.py
```

The script prints a new telemetry directory. Open its `index.md`, or give that path directly to a
filesystem-capable LLM. The [examples](https://github.com/Red-Eyed/torchinstruments/tree/main/examples)
also include a real MNIST classifier with Lightning and TensorBoard.

Instrument an existing model with one call before training:

```python
from datetime import timedelta

from torchinstruments import inject_observer, remove_observer

inject_observer(
    model,
    interval=timedelta(minutes=1),
    output_dir="stats",
)
try:
    train(model)
finally:
    remove_observer(model)
```

There is no observer call inside the training loop. The first eligible model forward after each
interval becomes a snapshot, and its backward is correlated automatically.

If model code calls `forward` methods directly, enable direct-forward capture once:

```python
inject_observer(model, output_dir="stats", capture_direct_forwards=True)
```

This mode recursively wraps the root and selected leaf modules, so both `module(x)` and
`module.forward(x)` are recorded. `remove_observer(model)` restores the prior `forward`
attributes. Normal `module(x)` dispatch remains the recommended PyTorch style and uses native
hooks by default; direct-forward capture is explicit because method wrapping is more invasive.

For distribution evidence and TensorBoard output, see the
[Lightning MNIST example](https://github.com/Red-Eyed/torchinstruments/blob/main/examples/lightning_mnist.py).
It uses an explicit 64-bin range and a less frequent histogram cadence so cost and comparison
semantics remain visible rather than hidden in defaults.

## Use cases

TorchInstruments helps answer focused research questions:

- Where does gradient signal first vanish or explode?
- Which layer first emits non-finite values?
- Do activation scales or distributions drift as training progresses?
- Are rare outliers dominating a layer's quantization range?
- Did an architecture change alter internal behavior even when loss curves look similar?
- Are model internals stable enough that the accuracy investigation should move to data, loss, or
  evaluation?

It attaches to the model rather than the trainer, so the same workflow applies to custom loops and
Lightning without telemetry calls inside training. The
[research workflow guide](https://github.com/Red-Eyed/torchinstruments/blob/main/docs/research-workflows.md)
covers controlled baseline-versus-candidate experiments in more depth.

## Analyze telemetry with an LLM

Canonical JSON is designed so an LLM can inspect module names, tensor paths, exact values, missing-
value reasons, and snapshot state without parsing TensorBoard files. For a filesystem-capable LLM,
point it at the generated `stats/index.md`; that file describes the run, every artifact, observed
fields, evidence limits, and a ready-to-use prompt. For an upload-based LLM, provide `index.md`,
`run.json`, `modules.json`, a small set of relevant snapshots, the validation result, and the exact
research intervention.

Ask the LLM to return four separate items for every finding:

1. **Observed fact** — exact module, snapshot, metric, and value.
2. **Hypothesis** — mechanisms consistent with the observation.
3. **Missing evidence** — information needed to distinguish those mechanisms.
4. **Next experiment** — the smallest controlled test that could falsify the hypothesis.

The
[LLM analysis guide](https://github.com/Red-Eyed/torchinstruments/blob/main/docs/llm-analysis.md)
provides a ready-to-use prompt, a two-run comparison prompt, file-selection guidance, and privacy
boundaries. It explicitly prevents claims about losses, optimizer updates, inputs, parameter
gradients, or data quality that were never observed.

## What is monitored by default

The default `inject_observer(model)` configuration has a deliberately bounded scope:

| Boundary | Default behavior |
| --- | --- |
| Sampling unit | One root-model forward and its associated backward |
| Sampling schedule | First root forward after each 60-second monotonic interval |
| Selected modules | Leaf modules: modules with no registered child modules |
| Invocation capture | Native PyTorch hooks; opt into wrappers for literal `.forward(...)` calls |
| Forward signals | Every tensor found in selected-module outputs, including nested lists, tuples, and dictionaries |
| Backward signals | The backpropagated gradient with respect to each differentiable selected-module output |
| Tensor metadata | `shape`, `dtype`, `device`, and `numel` |
| Scalar statistics | `mean`, population `std`, `rms`, `max_abs`, and `finite_fraction` |
| Histograms | Disabled; opt in with `histogram(...)` and an explicit independent cadence |
| Persistence | Strict, human-readable JSON under `stats/` |
| Collection failures | Preserve the error in telemetry and emit a warning |

Statistics operate on finite values in a numerically safer working dtype. Unavailable statistics
carry an explicit reason instead of producing non-standard JSON `NaN` or `Infinity`. Raw tensors
are never persisted.

The current release does **not** monitor module inputs, `grad_input`, parameters, parameter
gradients, losses, optimizer state, or optimizer updates. It also does not yet calculate
quantiles, zero fractions, per-channel metrics, or cross-snapshot summaries. These are planned
capabilities rather than implied behavior.

## Add histograms

Histograms are opt-in because they cost more than scalar statistics. Fixed bounds make the same
bins comparable throughout training:

```python
from torchinstruments import histogram, inject_observer

inject_observer(
    model,
    histograms=[
        histogram(
            bins=64,
            value_range=(-8.0, 8.0),
            every_n_snapshots=10,
        ),
    ],
)
```

Out-of-range values remain visible through underflow and overflow counts. The complete histogram
is stored in JSON even when it is also displayed in TensorBoard.

## Lightning and TensorBoard

Use the same Lightning logger for task metrics and internal model telemetry:

```python
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger

from torchinstruments import (
    CompositeSink,
    DirectorySink,
    TensorBoardSink,
    histogram,
    inject_observer,
    remove_observer,
)

logger = TensorBoardLogger(save_dir="logs", name="experiment")
sink = CompositeSink(
    DirectorySink("stats"),
    TensorBoardSink(logger),
)
model = MyLightningModule()
inject_observer(
    model.network,
    histograms=[histogram(value_range=(-8.0, 8.0))],
    sink=sink,
)

trainer = Trainer(logger=logger)
try:
    trainer.fit(model)
finally:
    remove_observer(model.network)
```

The Trainer still owns the logger. TorchInstruments never closes it. For another dashboard or a
custom `log_metrics(metrics, step)` implementation, use `MetricLoggerSink`. Sampling, module
selection, custom reducers, error policies, and sink protocols are covered in the
[design document](https://github.com/Red-Eyed/torchinstruments/blob/main/docs/design.md).

## Output layout

```text
stats/
    index.md
    run.json
    modules.json
    snapshots/
        000000.json
        000001.json
```

- `index.md` explains the run, observed evidence, file layout, limitations, and LLM workflow.
- `run.json` records versions, sampling, monitored signals, and built-in reducer configuration.
- `modules.json` records selected module types, aliases, and parameter counts once per run.
- Each numbered snapshot records one sampled forward and, when observed, its backward gradients.

A forward-only run remains useful: its snapshot is written as `forward_complete`. If backward later
uses its graph, the same file is atomically replaced with `backward_observed`. Multiple outstanding
forwards remain separate, and reused modules retain a distinct `call_index` for every invocation.

## Safety and performance boundaries

- Injection adds no parameters, buffers, or modules, so `model.state_dict()` remains unchanged.
- Native hooks or opt-in forward wrappers observe tensors without replacing model outputs or
  gradients.
- Unsampled capture callbacks take a cheap inactive path and do not call reducers.
- Sampled reductions run on the tensor's device; only compact scalar and histogram results move to
  CPU.
- Raw activations and gradients are never copied to disk.
- Snapshot files are strict JSON and use atomic replacement, so readers do not see partial writes.
- Duplicate injection raises `ObserverAlreadyAttachedError` instead of silently adding hooks.
- `remove_observer(model)` removes capture behavior and pending graph hooks, restores wrapped
  forwards, and closes the sink.

Collection does add reduction and device-synchronization cost on sampled passes. Every snapshot
records `collection_duration_ms` so telemetry cost remains visible rather than hidden.

## Compatibility and current scope

TorchInstruments requires Python 3.11 or newer and PyTorch 2.0 or newer. The core runtime depends
only on PyTorch and the Python standard library.

The current alpha supports CPU tensors and floating-point FP32, FP16, BF16, and FP64 diagnostics.
The test suite covers nested outputs, shared modules, multiple forwards combined into one
backward, inference-only execution, isolated reducer errors, lossless histogram projection,
generated LLM run indexes, mixed `module(...)`/`module.forward(...)` execution, composite output,
and a real Lightning `TensorBoardLogger`. Lightning, TensorBoard, and torchvision remain optional
development/example dependencies; they are not core wheel dependencies. CUDA, Accelerate, and
`torch.compile` behavior remain roadmap items and are not claimed as supported until they receive
dedicated compatibility tests.

Next development phases add input and parameter probes, richer reducer policies, aggregate
summaries, evidence-backed comparison and insight reports, quantization-oriented metrics,
distributed output policies, and domain-specific recipes through the existing extension
boundaries. See the
[design document](https://github.com/Red-Eyed/torchinstruments/blob/main/docs/design.md) for
lifecycle semantics and the complete roadmap.

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
  version = {0.4.0},
  url     = {https://github.com/Red-Eyed/torchinstruments}
}
```
