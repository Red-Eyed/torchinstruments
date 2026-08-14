# TorchInstruments

**See what is happening inside a PyTorch model before a bad loss curve becomes a failed run.**

TorchInstruments adds passive, trainer-agnostic telemetry to existing PyTorch models. Attach it
once, keep the training loop unchanged, and receive compact JSON snapshots of activation and
output-gradient behavior from the modules that matter.

It is useful when a scalar loss or validation score says that a research run underperformed but
cannot tell you where: an activation scale may be drifting with depth, gradients may disappear at
one layer, or a single outlier may be making a model difficult to quantize. TorchInstruments
preserves the evidence needed to investigate those questions without storing raw tensors or
adopting a particular trainer or dashboard.

```text
inject_observer(model)  ->  train normally  ->  inspect structured telemetry
```

## Quick start

TorchInstruments is currently an alpha available from its public GitHub repository. Run the
complete demonstration with Python 3.11 or newer and [`uv`](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/Red-Eyed/torchinstruments.git
cd torchinstruments
uv sync --dev
uv run examples/basic_training.py
```

The script prints a newly created telemetry directory containing three sampled training
iterations. See the
[example source](https://github.com/Red-Eyed/torchinstruments/blob/main/examples/basic_training.py)
and its [walkthrough](https://github.com/Red-Eyed/torchinstruments/tree/main/examples).

To instrument an existing model, add one call before training:

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

There is no `observer.step()`, loss wrapper, trainer callback, or logging call inside `train()`.
The first eligible model forward after each interval becomes a sampled snapshot, and its backward
is correlated with that same snapshot automatically.

To see the same telemetry in TensorBoard through a real Lightning logger:

```bash
uv run examples/lightning_mnist.py
```

The
[Lightning MNIST example](https://github.com/Red-Eyed/torchinstruments/blob/main/examples/lightning_mnist.py)
downloads the real dataset, trains a small CNN, and keeps validation accuracy, canonical JSON, and
TensorBoard telemetry from the same bounded research run.

## What the telemetry reveals

A snapshot associates measurements with the model's original module names:

```json
{
  "snapshot_id": 2,
  "state": "backward_observed",
  "modules": {
    "encoder.projection": [
      {
        "call_index": 0,
        "outputs": {
          "output": {
            "shape": [16, 512],
            "dtype": "bfloat16",
            "stats": {
              "rms": 0.83,
              "max_abs": 7.1,
              "finite_fraction": 1.0
            }
          }
        },
        "output_gradients": {
          "grad_output": {
            "stats": {
              "rms": 0.0042,
              "max_abs": 0.091,
              "finite_fraction": 1.0
            }
          }
        }
      }
    ]
  }
}
```

In practical terms, `output.rms` measures the typical activation scale, `output.max_abs` exposes
outliers relative to that scale, and `grad_output.rms` measures the backpropagated signal at the
module output. Comparing these values across layers and snapshots helps localize the first point
where training behavior changes.

## Use cases

### Investigate why accuracy stopped improving

Suppose a baseline reaches 78% validation accuracy while a modified model stalls at 75%. If the
candidate's gradient RMS collapses after one encoder layer while the baseline remains stable, that
is evidence for an optimization-path problem and motivates a controlled normalization, residual
scaling, initialization, or learning-rate experiment. If internal scales remain comparable, the
telemetry does not explain the gap and attention should move to missing evidence such as data,
labels, loss design, capacity, or evaluation.

TorchInstruments should narrow the hypothesis space; it should not invent a causal answer that the
observed tensors cannot support.

### Find exploding or vanishing gradients

Compare output-gradient RMS across adjacent modules. A sharp drop identifies where gradient signal
is disappearing; a sudden increase identifies where it begins to amplify. Because snapshots bind
forward and backward from the same model pass, the comparison is not assembled from unrelated
iterations.

### Catch non-finite values before the loss becomes NaN

Track `finite_fraction` through module order. The first module below `1.0` narrows the investigation
to the operation that produced non-finite values instead of the later loss computation that merely
reported them.

### Diagnose activation-scale drift

Compare activation RMS across snapshots after a learning-rate change, architecture modification,
or long training interval. Stable loss can hide a growing internal scale that later destabilizes
training or reduces numerical headroom.

### Assess quantization risk

Compare `max_abs` with `rms` to find layers whose ranges are dominated by rare outliers. Those
layers are candidates for clipping, alternate calibration, or architectural investigation before
running an expensive quantization evaluation.

### Compare model revisions

Module names and metric field names are deterministic, making snapshots suitable for direct JSON
diffs or LLM-assisted analysis. Compare the same layers before and after changing normalization,
initialization, residual connections, or precision.

### Add observability to any trainer

Hooks attach to the model rather than a training framework. Custom loops need no integration, and
Lightning, Accelerate, or other trainers do not become core package dependencies. The repository
contains tested Lightning/TensorBoard integration; other trainers still require dedicated
compatibility validation.

## Research diagnosis workflow

Useful analysis follows a disciplined chain:

```text
observed fact → plausible causes → missing evidence → next experiment
```

| Observation | What it may suggest | What to test next |
| --- | --- | --- |
| Gradient RMS drops sharply after one module | Saturation, detached paths, or poor residual scaling | Inspect that boundary and change one normalization or residual factor |
| Gradient RMS grows rapidly with depth | Unstable initialization, amplification, or excessive learning rate | Compare a lower learning rate or alternate initialization on the same data |
| Activation RMS drifts between snapshots | Distribution shift or normalization/residual instability | Compare fixed batches and inspect normalization state |
| `finite_fraction` first falls below `1.0` | Overflow or invalid arithmetic near that module | Reproduce the batch and inspect the preceding operation |
| Internal telemetry remains stable | The accuracy bottleneck may be data, labels, loss, capacity, or evaluation | Audit task-level evidence rather than changing scale blindly |

The
[research workflow guide](https://github.com/Red-Eyed/torchinstruments/blob/main/docs/research-workflows.md)
explains controlled baseline-versus-candidate collection, evidence levels, and how to convert a
suspicious signature into the smallest experiment that can falsify it.

## Analyze telemetry with an LLM

Canonical JSON is designed so an LLM can inspect module names, tensor paths, exact values, missing-
value reasons, and snapshot state without parsing TensorBoard files. For a bounded analysis, provide
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
| Forward signals | Every tensor found in selected-module outputs, including nested lists, tuples, and dictionaries |
| Backward signals | The backpropagated gradient with respect to each differentiable selected-module output |
| Tensor metadata | `shape`, `dtype`, `device`, and `numel` |
| Scalar statistics | `mean`, population `std`, `rms`, `max_abs`, and `finite_fraction` |
| Persistence | Strict, human-readable JSON under `stats/` |
| Collection failures | Preserve the error in telemetry and emit a warning |

Statistics operate on finite values in a numerically safer working dtype. Unavailable statistics
carry an explicit reason instead of producing non-standard JSON `NaN` or `Infinity`. Raw tensors
are never persisted.

The current release does **not** monitor module inputs, `grad_input`, parameters, parameter
gradients, losses, optimizer state, or optimizer updates. It also does not yet calculate
quantiles, histograms, zero fractions, per-channel metrics, or cross-snapshot summaries. These are
planned capabilities rather than implied behavior.

## Configuration

The short API resolves the same replaceable components exposed by the explicit API:

```python
from torchinstruments import (
    AlwaysSampler,
    CompositeSink,
    DirectorySink,
    MetricLoggerSink,
    default_reducers,
    inject_observer,
    leaf_modules,
)

inject_observer(
    model,
    sampler=AlwaysSampler(),
    selector=leaf_modules(),
    reducers=default_reducers(),
    sink=DirectorySink("stats"),
    error_policy="warn",
)
```

| Setting | Purpose | Built-in choices |
| --- | --- | --- |
| `interval` | Configure time-based sampling through the convenience API | Any positive `datetime.timedelta`; default is one minute |
| `output_dir` | Select the directory used by the convenience sink | `"stats"` by default |
| `sampler` | Decide which root forwards become snapshots | `TimedSampler`, `EveryNForwardsSampler`, `AlwaysSampler` |
| `selector` | Decide which named modules receive collection hooks | `leaf_modules()` or any compatible predicate |
| `reducers` | Convert detached tensors into compact named scalars | `mean()`, `std()`, `rms()`, `max_abs()`, `finite_fraction()`, `combine(...)` |
| `sink` | Persist or project normalized records | `DirectorySink`, `MetricLoggerSink`, `CompositeSink`, or any compatible sink |
| `error_policy` | Control collection failure behavior | `"warn"`, `"ignore"`, or `"raise"` |

`interval` cannot be combined with a custom `sampler`, and `output_dir` cannot be combined with a
custom `sink`; conflicting configuration raises immediately instead of silently ignoring one
value.

Sampling every 100 root forwards requires no trainer step counter:

```python
from torchinstruments import EveryNForwardsSampler, inject_observer

inject_observer(
    model,
    sampler=EveryNForwardsSampler(100),
    output_dir="stats",
)
```

Module selection is an ordinary typed callable, so domain-specific policy stays outside the core:

```python
from torch import nn

from torchinstruments import AlwaysSampler, inject_observer


def select_linear_layers(name: str, module: nn.Module) -> bool:
    """Select named linear transformations for one diagnostic run."""
    return bool(name) and isinstance(module, nn.Linear)


inject_observer(
    model,
    sampler=AlwaysSampler(),
    selector=select_linear_layers,
    output_dir="linear-stats",
)
```

Reducers and sinks use the same callable/protocol style, so custom diagnostics and persistence do
not require inheriting from framework base classes.

### Lightning, TensorBoard, and custom metric loggers

`MetricLoggerSink` accepts any object with `log_metrics(metrics, step)`, including Lightning's
`TensorBoardLogger`. `CompositeSink` sends each lifecycle event to multiple destinations:

```python
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger

from torchinstruments import (
    CompositeSink,
    DirectorySink,
    MetricLoggerSink,
    inject_observer,
    remove_observer,
)

logger = TensorBoardLogger(save_dir="logs", name="experiment")
sink = CompositeSink(
    DirectorySink("stats"),
    MetricLoggerSink(logger),
)
model = MyLightningModule()
inject_observer(model.network, sink=sink)

trainer = Trainer(logger=logger)
try:
    trainer.fit(model)
finally:
    remove_observer(model.network)
```

The metric logger receives paths such as
`torchinstruments/modules/encoder.projection/call_0/output/rms`. Logger steps are snapshot IDs, not
Lightning optimizer steps, because a framework-independent observer cannot infer a universal
optimizer-step counter.

`MetricLoggerSink` never finalizes an externally supplied logger. The Lightning `Trainer` owns its
logger lifecycle; custom callers retain the same responsibility. JSON remains the recommended
canonical output because scalar loggers cannot preserve shapes, dtypes, errors, unavailable-value
reasons, or module metadata.

A custom logger needs only the same small method:

```python
from collections.abc import Mapping


class ConsoleMetricLogger:
    """Print scalar telemetry for a minimal local diagnostic run."""

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        """Print one explicitly identified telemetry snapshot."""
        print({"snapshot_id": step, "metrics": dict(metrics)})


inject_observer(model, sink=MetricLoggerSink(ConsoleMetricLogger()))
```

## Output layout

```text
stats/
    run.json
    modules.json
    snapshots/
        000000.json
        000001.json
```

- `run.json` records the schema, package and PyTorch versions, creation time, and sampling policy.
- `modules.json` records selected module types, aliases, and parameter counts once per run.
- Each numbered snapshot records one sampled forward and, when observed, its backward gradients.

A forward-only run remains useful: its snapshot is written as `forward_complete`. If backward later
uses its graph, the same file is atomically replaced with `backward_observed`. Multiple outstanding
forwards remain separate, and reused modules retain a distinct `call_index` for every invocation.

## Safety and performance boundaries

- Injection adds no parameters, buffers, or modules, so `model.state_dict()` remains unchanged.
- Hooks observe tensors without replacing model outputs or gradients.
- Unsampled module hooks take a cheap inactive path and do not call reducers.
- Sampled reductions run on the tensor's device; only compact scalar results move to CPU.
- Raw activations and gradients are never copied to disk.
- Snapshot files are strict JSON and use atomic replacement, so readers do not see partial writes.
- Duplicate injection raises `ObserverAlreadyAttachedError` instead of silently adding hooks.
- `remove_observer(model)` removes module and pending graph hooks and closes the sink.

Collection does add reduction and device-synchronization cost on sampled passes. Every snapshot
records `collection_duration_ms` so telemetry cost remains visible rather than hidden.

## Compatibility and current scope

TorchInstruments requires Python 3.11 or newer and PyTorch 2.0 or newer. The core runtime depends
only on PyTorch and the Python standard library.

The current alpha supports CPU tensors and floating-point FP32, FP16, BF16, and FP64 diagnostics.
The test suite covers nested outputs, shared modules, multiple forwards combined into one
backward, inference-only execution, isolated reducer errors, composite output, and a real Lightning
`TensorBoardLogger`. Lightning, TensorBoard, and torchvision remain optional development/example
dependencies; they are not core wheel dependencies. CUDA, Accelerate, and `torch.compile` behavior
remain roadmap items and are not claimed as supported until they receive dedicated compatibility
tests.

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
  version = {0.2.0},
  url     = {https://github.com/Red-Eyed/torchinstruments}
}
```
