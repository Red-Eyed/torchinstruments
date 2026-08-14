# TorchInstruments

TorchInstruments adds passive, trainer-agnostic telemetry to PyTorch models. It samples root
model forwards, collects compact activation and output-gradient statistics from selected
modules, and writes strict JSON records without requiring changes to the training loop.

```python
from datetime import timedelta

from torchinstruments import inject_observer

inject_observer(
    model,
    interval=timedelta(minutes=1),
    output_dir="stats",
)

train(model)
```

The observer is attached as ordinary PyTorch hooks. It does not add parameters, buffers, or
modules, and therefore does not change `state_dict()`.

## Lifecycle

Injection modifies the model in place and deliberately returns `None`. Duplicate injection raises
`ObserverAlreadyAttachedError` so configuration is never replaced silently.

```python
from torchinstruments import has_observer, remove_observer

assert has_observer(model)
remove_observer(model)
```

Removal detaches module and pending graph hooks, closes the sink, and deletes the observer's
private Python state.

## Configuration

The convenience API constructs the default sampler, leaf-module selector, reducers, and directory
sink. Each component can instead be supplied explicitly:

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

Supported error policies are `raise`, `warn`, and `ignore`. Both non-raising policies preserve
collection failures inside snapshot telemetry; `warn` additionally emits a Python warning.

## Output

```text
stats/
    run.json
    modules.json
    snapshots/
        000000.json
        000001.json
```

A sampled forward is written immediately as `forward_complete`. If its graph later participates
in backward, the same snapshot is atomically enriched to `backward_observed`. This preserves useful
telemetry for inference-only runs while correctly separating multiple outstanding forwards.

Built-in reducers report mean, population standard deviation, RMS, maximum absolute value, and
finite fraction. Statistics operate on finite values, and unavailable results carry explicit
reasons instead of non-standard JSON NaN or infinity values. Raw tensors are never written.

## Compatibility

TorchInstruments requires Python 3.11 or newer and PyTorch 2.0 or newer. The core runtime depends
only on PyTorch and the Python standard library. Trainer-specific integrations are not required.

## Phase 1 scope

The initial implementation provides:

- time-based and always-on root-forward sampling;
- leaf-module selection;
- nested tensor-output traversal with stable paths;
- mean, population standard deviation, RMS, maximum absolute value, and finite fraction;
- correlated output-gradient statistics;
- `run.json`, `modules.json`, and one atomically updated JSON file per snapshot;
- explicit observer removal and duplicate-injection detection.

See [the design document](docs/design.md) for lifecycle semantics, deliberate limitations, and
the project roadmap.

## License

TorchInstruments is released under the [MIT License](LICENSE).

## Citation

If TorchInstruments supports your research or engineering work, cite it as:

```bibtex
@software{stupakov_2026_torchinstruments,
  author  = {Vadym Stupakov},
  title   = {TorchInstruments: Passive PyTorch Model Telemetry},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/Red-Eyed/torchinstruments}
}
```
