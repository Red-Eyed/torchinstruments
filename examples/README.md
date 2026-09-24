# Examples

## Reproduce and investigate model problems

`uv run examples/find_problems.py` creates five controlled experiments automatically, with a
progress bar. Use `--output PATH`, `--steps N` (at least eight), or `--quiet` to configure it.
No dataset download is needed.

Each problem has `baseline/`, `broken/`, and `fixed/` runs. Every run contains `history.parquet`,
`result.json`, `index.md`, and `tensorboard/`. The parent directory also contains a readable
explanation, `comparison.csv`, and a high-resolution `comparison.png`.

| Problem introduced at sample 4 | Query in the sampled history | What the fix changes |
| --- | --- | --- |
| Inactive ReLU | `activation`, output, zero_fraction reaches one | Restores the deliberately negative pre-activation bias |
| Saturated sigmoid | `pre`, output_gradient, max collapses towards zero | Restores pre-activation scale and bias |
| Growing gain | `gain`, output, std grows after the intervention | Holds gain at one |
| Invalid logarithm | `probe`, output, nonfinite_fraction becomes positive | Restores log1p(abs(x)) instead of log(-abs(x)) |
| Detached branch | `pre`, output_gradient disappears despite later forwards | Removes the bridge's detach operation |

The gain experiment holds weights fixed to isolate gain growth from optimizer feedback.
Baseline and fixed runs share initialization, data order, and optimizer settings. The broken
run differs only in the explicit intervention. The examples skip optimizer updates on nonfinite
loss to avoid making the next step's invalid parameters a second independent problem.

A missing gradient appears as a gap, never a measured zero. For sigmoid saturation, inspect the
layer *before* the sigmoid: the gradient with respect to the sigmoid's output alone does not
measure its derivative. All-zero activation across a few batches is evidence about those batches,
not proof every channel will remain dead forever. These examples verify measurement signatures;
they do not promise that changing one statistic improves a real task metric.

## Ordinary training

`uv run examples/basic_training.py` runs three small regression updates with the default
one-minute interval; the first forward is sampled immediately. It automatically finalizes the Parquet history and per-layer JSON on observer removal.
The output path is printed. No custom sampler, reducer, or history configuration is needed.

## Lightning and MNIST

`uv run examples/lightning_mnist.py` downloads MNIST when needed and trains a compact classifier.
Task metrics and focused histograms share Lightning's TensorBoard logger. The telemetry directory
also owns the required history, JSON summary, guide, and TensorBoard events.
`MnistClassifier.on_fit_start()` takes the logger from `self.trainer.logger` and attaches the
observer to `self.network`, the computational root called by `training_step()`.
`on_fit_end()` finalizes telemetry; the outer `finally` also cleans up if fitting fails.
The model constructor does not need a logger.
Lightning's logger remains caller-owned and is not closed by observer removal.

The controlled fault experiments observe every synthetic step. The Lightning example shares an
existing TensorBoard logger. Ordinary integration needs only the standard attach/remove calls.
