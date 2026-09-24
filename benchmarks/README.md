# Telemetry cost measurements

`telemetry.py` creates its output directory, warms up the model, synchronizes accelerator timings,
and writes configuration, per-step timings in Parquet, and a resource summary. Run each
configuration in a fresh process so process peak RSS is not contaminated by another workload.

```bash
uv run benchmarks/telemetry.py
```

The default measures a small sequence model with the ordinary one-minute sampling interval.
Each run uses three warm-up iterations and twenty measured iterations, float32, and one CPU
thread. The CLI has five options: `--workload`, `--sampling`, `--device`, `--output`, and `--quiet`.

- Workloads: `sequence` (four 128-wide Linear layers), `conv` (four 64-channel Conv2d layers),
  or `many-layers` (1,000 Identity modules). Shapes are saved in configuration.json.
- Sampling: `interval` for normal usage, `off` for a reference, or `every-forward` for a stress test.
- Devices: `cpu`, `mps`, or `cuda`; unavailable devices do not silently fall back.

The historical measurements below used the earlier configurable harness. Their exact sizes and
iteration counts remain stated; they are not all reproduced by these three simplified workloads.

## Local observations, 2026-09-24

Environment: macOS 26.6.2, arm64, PyTorch 2.13.0, Polars 1.44.2, one CPU intra-op thread.
The baseline is the released 0.8.0 wheel loaded in a separate process. The candidate is the
uncommitted reliability implementation, still carrying 0.8.0 metadata when measured. No CUDA
device was available. These short exploratory runs do not establish statistical regressions,
improvements, or hardware-independent performance guarantees.

For the intended 60-second interval, a separate 10,000-iteration CPU run averaged **0.199 ms**
per forward/backward versus **0.189 ms** without observation. It collected exactly one forward
(the immediate first sample); the run finished before the next deadline. This distinguishes
interval usage from the every-forward stress results below. It is a short single-run comparison,
not a measurement of repeated 60-second steady-state sampling. Interval boundary behavior is
covered separately with injected-clock tests. Per-sample peak memory is unchanged by cadence.

Common workload: four Linear layers, batch 2, sequence length 128, width 128, float32, three
uninstrumented warm-up iterations, then ten measured forward/backward iterations. Sampling
every iteration includes synchronous artifact writes.

| Configuration | Mean forward + backward, ms |
|---|---:|
| Released 0.8.0, CPU, observer absent | 0.253 |
| Released 0.8.0, CPU, attached but unsampled | 0.305 |
| Released 0.8.0, CPU, every iteration sampled | 16.232 |
| Candidate, CPU, observer absent | 0.246 |
| Candidate, CPU, attached but unsampled | 0.271 |
| Candidate, CPU, every iteration sampled | 17.504 |
| Candidate, MPS, observer absent | 1.317 |
| Candidate, MPS, attached but unsampled | 1.405 |
| Candidate, MPS, every iteration sampled | 110.425 |
| Candidate, MPS, sampled with float16 | 90.858 |

Do not claim low overhead when sampling every forward. The default timed cadence avoids full
collection cost on most forwards. These measurements do not isolate kernels, synchronization,
and disk writes, so they do not justify background queues or changed numerical semantics yet.

## Scale checks

- 1,000 Identity modules, batch 2, width 8, ten sampled iterations: 205.821 ms per iteration,
  180,000 history rows, 41 chunks before consolidation, and 0.226 s removal.
- One Identity output with 16,777,217 float32 elements, batch 1, two sampled iterations after
  one warm-up: 1.187 s per iteration, 36 history rows, and 1.165 GB process peak RSS. Exact
  quartiles work, but compact output does not imply cheap collection.
- Four CPU Conv2d layers, batch 2, 64 channels, 64x64 spatial size, five sampled iterations:
  249.709 ms per iteration.
- Four MPS Linear layers, batch 2, sequence length 512, width 1024, five sampled iterations:
  2.264 s per iteration.

Holding ten layers and nine series per layer fixed, increasing synthetic history from 90,000
to 900,000 rows raised whole-process peak RSS from 375.8 MB to 735.2 MB. Consolidation plus
summary generation took 0.019 s and 0.067 s. RSS includes writing, imports, and aggregation;
it does not isolate the finalizer. Final files retain every generated row. These results do
not establish bounded native memory for arbitrary histories.

Full scalar results are in [local-results.csv](local-results.csv). Local per-run configurations,
raw timings, and telemetry live under `stats/reliability-benchmarks/`. Candidate measurements
preceded final documentation and additional validation changes; they are baseline evidence for
future optimization, not a final release benchmark. MPS reports current device allocation
because the harness has no supported peak counter for that backend; CUDA peak allocation is
measured when CUDA is available.

Further performance work needs a profile of the target model and an agreed sampling latency
and memory budget. Exact quartiles remain the default. Approximation requires an explicit
accuracy/performance decision and independently validated error bounds.
