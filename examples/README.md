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
    index.md
    run.json
    modules.json
    snapshots/
        000000.json
        000001.json
        000002.json
```

Open `index.md` first for the run summary, file guide, and LLM prompt. Each snapshot starts with
measurements from one forward and is atomically updated after backward.
For production training, omit `sampler=AlwaysSampler()` to restore the default one-minute interval
and reduce collection overhead.

## Lightning, MNIST, and TensorBoard

[`lightning_mnist.py`](lightning_mnist.py) downloads MNIST, trains a small convolutional classifier,
and passes one real Lightning `TensorBoardLogger` to both the Lightning `Trainer` and
`TensorBoardSink`. Task loss and validation accuracy appear beside internal activation and
gradient scalars and histograms, while `CompositeSink` preserves complete JSON snapshots:

```bash
uv run examples/lightning_mnist.py
```

The default configuration automatically downloads MNIST, runs 100 training batches and 25
validation batches, samples every 25th computational forward, and logs histograms every fourth
telemetry snapshot. Histogram values use 64 fixed bins from `-8` to `8`, with explicit underflow
and overflow counts in JSON. These settings are visible in `MnistRunConfig`; adjust them for the
model scales and duration of a longer research run.

The example instruments `model.network`, the computational module invoked by `training_step()`.
Lightning does not guarantee that a trainer calls the outer `LightningModule.forward()`, so the
actual computational root—not necessarily the trainer-owned wrapper—should receive the observer.

Lightning, TensorBoard, and torchvision are development/example dependencies. They are not
installed with the TorchInstruments core wheel.
