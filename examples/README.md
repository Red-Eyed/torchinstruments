# Examples

## Basic training

[`basic_training.py`](basic_training.py) attaches TorchInstruments to an ordinary PyTorch model,
runs three optimizer iterations, and removes the observer afterward. The training loop contains no
telemetry calls or trainer-specific integration.

Run it from the repository root:

```bash
uv run examples/basic_training.py
```

The script selects every leaf module and samples every forward so a short run always demonstrates
both activation and output-gradient telemetry. It creates a collision-free directory like:

```text
stats/basic-training-a1b2c3d4/
    run.json
    modules.json
    snapshots/
        000000.json
        000001.json
        000002.json
```

Each snapshot starts with measurements from one forward and is atomically updated after backward.
For production training, omit `sampler=AlwaysSampler()` to restore the default one-minute interval
and reduce collection overhead.
