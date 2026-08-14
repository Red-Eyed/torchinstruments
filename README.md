# TorchInstruments

**See what is happening inside a PyTorch model before a bad loss curve becomes a failed run.**

TorchInstruments adds passive, trainer-agnostic telemetry to existing PyTorch models. Attach it
once, keep the training loop unchanged, and receive compact JSON snapshots of activation and
output-gradient behavior from the modules that matter.

It is useful when a scalar loss says that training failed but cannot tell you where: an activation
scale may be drifting with depth, gradients may disappear at one layer, or a single outlier may be
making a model difficult to quantize. TorchInstruments preserves the evidence needed to investigate
those questions without storing raw tensors or adopting a particular trainer or dashboard.

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
iterations. See the [example source](examples/basic_training.py) and its
[walkthrough](examples/README.md).

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
Lightning, Accelerate, or other trainers do not become core package dependencies. Dedicated
compatibility tests for those trainers remain roadmap work.

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
    DirectorySink,
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
| `sink` | Persist normalized run and snapshot records | `DirectorySink` or any compatible sink |
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
backward, inference-only execution, and isolated reducer errors. CUDA and
`torch.compile` behavior remain roadmap items and are not claimed as supported until they receive
dedicated compatibility tests.

Next development phases add input and parameter probes, richer reducer policies, aggregate
summaries, quantization-oriented metrics, distributed output policies, and domain-specific recipes
through the existing extension boundaries. See the [design document](docs/design.md) for lifecycle
semantics and the complete roadmap.

## License

TorchInstruments is released under the [MIT License](LICENSE).

## Citation

If TorchInstruments supports your research or engineering work, cite it as:

```bibtex
@software{stupakov_2026_torchinstruments,
  author  = {Vadym Stupakov},
  title   = {TorchInstruments: Passive PyTorch Model Telemetry},
  year    = {2026},
  version = {0.1.1},
  url     = {https://github.com/Red-Eyed/torchinstruments}
}
```
