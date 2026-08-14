# Changelog

All notable changes to TorchInstruments are documented here.

## [0.3.0] - 2026-08-14

### Highlights

- Add opt-in activation and output-gradient histograms that remain fully reproducible from the
  canonical JSON telemetry.
- Generate a bounded `index.md` in every run directory so humans and LLMs can understand and
  analyze a run without prior knowledge of the schema.

### New Features

- Add configurable fixed-range or dynamic histograms with independent every-N-snapshots sampling.
- Record bin edges and counts, underflow and overflow, finite prevalence, extrema, and compact
  moments without persisting raw tensors.
- Add `TensorBoardSink` for scalar and pre-aggregated histogram projection through externally owned
  Lightning-compatible loggers.
- Record monitored signals and built-in reducer configuration in `run.json`, while identifying
  opaque custom reducers by type.

### Documentation

- Rewrite the README around diagnostic outcomes, concrete next experiments, and the simple
  attach-train-analyze workflow.
- Extend the real MNIST example with JSON-backed TensorBoard histograms.

### Developers

- Add unit and real TensorBoard event coverage proving histogram events are derived from the same
  normalized records written to JSON.

## [0.2.0] - 2026-08-14

### Highlights

- Add optional scalar-logger output alongside canonical JSON for live research dashboards.

### New Features

- Add `MetricLoggerSink` for Lightning-compatible and custom `log_metrics()` implementations.
- Add `CompositeSink` for lossless JSON and scalar logger fan-out from the same snapshots.

### Documentation

- Add a tested Lightning MNIST example with task accuracy, `TensorBoardLogger`, and structured
  telemetry from a real dataset workflow.
- Add research-diagnosis and LLM-analysis workflows that separate observations, hypotheses,
  missing evidence, and next experiments.

### Developers

- Add Lightning, TensorBoard, and torchvision as development-only dependencies for real
  integration coverage.

## [0.1.1] - 2026-08-14

### Documentation

- Add a runnable training example that produces forward and output-gradient snapshots.
- Add a quick start, concrete use cases, monitoring boundaries, configuration recipes, and
  telemetry interpretation guidance to the project README.

### Developers

- Include runnable examples in Ruff and Pyrefly validation.

## [0.1.0] - 2026-08-14

### Highlights

- Add passive, trainer-agnostic telemetry that requires no training-loop callbacks.

### New Features

- Add time-based, periodic-forward, and always-on sampling policies.
- Add leaf-module output and correlated output-gradient statistics.
- Add strict, atomically updated JSON run metadata and snapshots.
- Add explicit observer detection, duplicate-injection errors, and cleanup.

### Documentation

- Document lifecycle semantics, extension boundaries, limitations, and the project roadmap.
- Publish citation guidance and the MIT License.

### Developers

- Support and test Python 3.11 through 3.14.
- Add Ruff, Pyrefly, pytest, and declarative `dirty-equals` assertions.
