# Changelog

All notable changes to TorchInstruments are documented here.

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
