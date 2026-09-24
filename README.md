# TorchInstruments

Record per-layer activation and output-gradient statistics without changing your training loop.

```python
from datetime import timedelta
from torchinstruments import inject_observer, remove_observer

inject_observer(model, interval=timedelta(minutes=1), output_dir="stats")
try:
    train(model)
finally:
    remove_observer(model)
```

The first forward is sampled immediately, then once per interval. The default interval is one
minute. Evaluation works too, including `torch.no_grad()` and `torch.inference_mode()`.

## Output

Everything is created automatically:

| Artifact | Purpose |
|---|---|
| `history.parquet` | Full sampled scalar history |
| `result.json` | Indented history summary for each layer |
| `index.md` | Reading guide for an LLM, including schema and history queries |
| `tensorboard/` | Histogram history for selected layers |

`history.parquet`, `result.json`, `index.md`, and TensorBoard update after each sampled forward
or backward. They are readable without calling `remove_observer()`. Removal detaches hooks,
closes resources, and cleans up intermediate `history.parts/` chunks. Interrupted runs retain
the last published files and completed chunks. Each refresh rewrites the Parquet snapshot and
aggregates collected history, so its cost grows with run length.

## Statistics

Each sampled tensor reports mean, population standard deviation, min, max, p25, p50, p75,
zero fraction, and nonfinite fraction. Gradients are recorded separately. Train/eval context
is captured automatically, and missing measurements carry reasons.

JSON aggregates each metric's history, including previous/recent window means. These summarize
sampled statistics, not pooled tensor entries. Exact observations stay in Parquet. There are
no scores or diagnoses. Give `index.md` to an LLM as the starting point.

## Layer selection

All leaf modules contribute scalar history. Histograms cover the first eight selected modules.
To focus the dashboard, use `histogram_selector`:

```python
inject_observer(
    model,
    histogram_selector=lambda name, module: name.startswith("encoder.blocks.7."),
)
```

Use `selector` similarly to restrict all collection. Histogram focus must match an observed module.
Other existing extension arguments remain available; ordinary usage needs none of them.

## Examples and limits

- [Examples](examples/README.md): ordinary training, Lightning, and controlled model problems.
- [LLM guide](docs/llm-analysis.md): how to interpret measurements and query history.
- [Benchmarks](benchmarks/README.md): interval-based usage and separate stress tests.
- [Changelog](CHANGELOG.md): version changes and schema migration.

CPU, native MPS, in-place activations, and non-reentrant checkpointing are tested. Reentrant
checkpoint internals lack gradient coverage. Only the first backward is recorded, with the
caller's loss scaling. CUDA performance and torch.compile compatibility are unverified.
The interval amortizes collection cost; sampled forwards still scan their tensors.

MIT licensed. Author: Vadym Stupakov <vadim.stupakov@gmail.com>.
