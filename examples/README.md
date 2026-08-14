# Examples

## Basic training

[`basic_training.py`](basic_training.py) instruments an ordinary PyTorch model, runs three
optimizer iterations, and removes the observer. The training loop contains no telemetry calls.

```bash
uv run examples/basic_training.py
```

The example samples every forward so the short run demonstrates live activation and gradient
indicators:

```text
stats/basic-training-a1b2c3d4/
    index.md
    stats.json
```

Open `index.md` first. `stats.json` is updated after each sampled forward and correlated backward;
no per-sample files are created. Production training can omit `AlwaysSampler()` to restore the
one-minute default interval.

## Lightning, MNIST, and TensorBoard

[`lightning_mnist.py`](lightning_mnist.py) downloads MNIST, trains a convolutional classifier, and
shares one real Lightning `TensorBoardLogger` with `TensorBoardSink`.

```bash
uv run examples/lightning_mnist.py
```

Task loss and validation accuracy appear beside live activation and gradient events in
TensorBoard. `DirectorySink` independently maintains the canonical bounded `stats.json` with
distribution and temporal indicators. The default run samples every 25th computational forward
and collects fixed-range histograms every fourth telemetry sample.

The observer is attached to `model.network`, the module actually invoked by `training_step()`.
Lightning does not guarantee that the outer `LightningModule.forward()` is called.

Lightning, TensorBoard, and torchvision are development/example dependencies, not core wheel
dependencies.
