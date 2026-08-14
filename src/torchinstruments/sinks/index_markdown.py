"""Bounded run-index rendering for human and LLM telemetry discovery."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC

from torchinstruments.records import (
    ModuleRecord,
    ReducerRecord,
    RunRecord,
    SnapshotRecord,
    TensorRecord,
)


@dataclass(frozen=True)
class ObservedTelemetry:
    """Summarize record fields observed so far without retaining snapshot values."""

    signals: frozenset[str] = frozenset()
    scalar_statistics: frozenset[str] = frozenset()
    histogram_names: frozenset[str] = frozenset()
    unavailable_statistics: frozenset[str] = frozenset()
    unavailable_histograms: frozenset[str] = frozenset()

    def merged(self, other: ObservedTelemetry) -> ObservedTelemetry:
        """Return the union of two immutable telemetry summaries."""
        return ObservedTelemetry(
            signals=self.signals | other.signals,
            scalar_statistics=self.scalar_statistics | other.scalar_statistics,
            histogram_names=self.histogram_names | other.histogram_names,
            unavailable_statistics=self.unavailable_statistics | other.unavailable_statistics,
            unavailable_histograms=self.unavailable_histograms | other.unavailable_histograms,
        )


def summarize_snapshot(snapshot: SnapshotRecord) -> ObservedTelemetry:
    """Extract bounded field-name evidence from one normalized snapshot."""
    signals: set[str] = set()
    scalar_statistics: set[str] = set()
    histogram_names: set[str] = set()
    unavailable_statistics: set[str] = set()
    unavailable_histograms: set[str] = set()

    for calls in snapshot.modules.values():
        for call in calls:
            _observe_tensors(
                call.outputs,
                signal="module outputs",
                signals=signals,
                scalar_statistics=scalar_statistics,
                histogram_names=histogram_names,
                unavailable_statistics=unavailable_statistics,
                unavailable_histograms=unavailable_histograms,
            )
            _observe_tensors(
                call.output_gradients,
                signal="module output gradients",
                signals=signals,
                scalar_statistics=scalar_statistics,
                histogram_names=histogram_names,
                unavailable_statistics=unavailable_statistics,
                unavailable_histograms=unavailable_histograms,
            )

    return ObservedTelemetry(
        signals=frozenset(signals),
        scalar_statistics=frozenset(scalar_statistics),
        histogram_names=frozenset(histogram_names),
        unavailable_statistics=frozenset(unavailable_statistics),
        unavailable_histograms=frozenset(unavailable_histograms),
    )


def render_run_index(
    run: RunRecord,
    modules: Mapping[str, ModuleRecord],
    *,
    snapshot_count: int,
    observed: ObservedTelemetry,
) -> str:
    """Render a deterministic, bounded guide to one canonical telemetry directory."""
    sampling_settings = json.dumps(dict(run.sampling.settings), sort_keys=True)
    created_at = run.created_at.astimezone(UTC).isoformat().replace("+00:00", "Z")
    latest_snapshot = str(snapshot_count - 1) if snapshot_count else "not available yet"
    lines = [
        "# TorchInstruments run index",
        "",
        "> Start here. This guide is derived from the run records; JSON remains the canonical",
        "> source of telemetry and contains everything projected to TensorBoard.",
        "",
        "## Run overview",
        "",
        f"- Schema version: `{run.schema_version}`",
        f"- TorchInstruments version: `{run.observer_version}`",
        f"- PyTorch version: `{run.torch_version}`",
        f"- Created at: `{created_at}`",
        f"- Sampling policy: `{run.sampling.type}` with settings `{sampling_settings}`",
        f"- Configured signals: {_render_inline(run.collection.signals)}",
        f"- Scalar reducers: {_render_reducers(run.collection.scalar_reducers)}",
        f"- Histogram reducers: {_render_reducers(run.collection.histogram_reducers)}",
        f"- Selected modules: `{len(modules)}`",
        f"- Snapshots written: `{snapshot_count}`",
        f"- Latest snapshot ID: `{latest_snapshot}`",
        "",
        "## Files",
        "",
        "- `index.md`: this bounded human and LLM entry point.",
        "- `run.json`: immutable schema, versions, timestamp, and sampling policy.",
        "- `modules.json`: selected module names, aliases, types, and parameter counts.",
        "- `snapshots/NNNNNN.json`: one sampled root forward, atomically enriched if its",
        "  correlated backward is observed.",
        "",
        "Snapshot IDs are telemetry IDs, not optimizer-step numbers. Module names and tensor",
        "paths in JSON preserve the model structure seen by the observer.",
        "",
        "## Telemetry observed so far",
        "",
        _render_names("Signals", observed.signals),
        _render_names("Scalar statistics", observed.scalar_statistics),
        _render_names("Histogram records", observed.histogram_names),
        _render_names("Unavailable scalar statistics", observed.unavailable_statistics),
        _render_names("Unavailable histograms", observed.unavailable_histograms),
        "",
        "Each tensor record also includes `shape`, `dtype`, `device`, and `numel`. Histogram",
        "records include bin edges and counts, underflow and overflow counts, finite and",
        "non-finite counts, minimum, maximum, sum, and sum of squares. These fields are enough",
        "to reproduce the corresponding TensorBoard histogram without the raw tensor.",
        "",
        "## Use this run with an LLM",
        "",
        "For an LLM with filesystem access, point it at this directory and use:",
        "",
        "```text",
        "Read index.md, run.json, modules.json, and representative files under snapshots/.",
        "Analyze the stated research question using only evidence present in those files.",
        "For every finding, report: (1) observed fact with module, snapshot, field, and value;",
        "(2) plausible hypothesis; (3) missing evidence; and (4) the smallest next experiment.",
        "Distinguish measured evidence from inference and state when telemetry does not explain",
        "the task-level result.",
        "```",
        "",
        "For an upload-based LLM, attach this file, `run.json`, `modules.json`, the task-level",
        "result, and a bounded set of snapshots. Start with early, middle, and late snapshots",
        "plus snapshots near a known loss or accuracy change; do not upload thousands blindly.",
        "",
        "## Evidence limits",
        "",
        "TorchInstruments currently observes selected-module outputs and their correlated output",
        "gradients. It does not observe loss values, labels, optimizer updates, module inputs,",
        "parameters, or parameter gradients unless those are supplied separately. Scale or",
        "distribution anomalies support hypotheses; they do not prove why accuracy changed.",
        "",
    ]
    return "\n".join(lines)


def _observe_tensors(
    tensors: Mapping[str, TensorRecord],
    *,
    signal: str,
    signals: set[str],
    scalar_statistics: set[str],
    histogram_names: set[str],
    unavailable_statistics: set[str],
    unavailable_histograms: set[str],
) -> None:
    """Accumulate names from one signal mapping while keeping rendering state bounded."""
    if not tensors:
        return

    signals.add(signal)
    for value in tensors.values():
        scalar_statistics.update(value.stats)
        histogram_names.update(value.histograms)
        unavailable_statistics.update(value.unavailable_stats)
        unavailable_histograms.update(value.unavailable_histograms)


def _render_names(label: str, values: frozenset[str]) -> str:
    """Render one stable inline set without allowing the index to grow per snapshot."""
    rendered = ", ".join(f"`{value}`" for value in sorted(values))
    return f"- {label}: {rendered or 'none observed yet'}"


def _render_reducers(reducers: tuple[ReducerRecord, ...]) -> str:
    """Render complete configured reducer metadata without relying on defaults."""
    if not reducers:
        return "none configured"
    rendered: list[str] = []
    for value in reducers:
        settings = json.dumps(dict(value.settings), sort_keys=True)
        rendered.append(f"`{value.type}` with settings `{settings}`")
    return "; ".join(rendered)


def _render_inline(values: tuple[str, ...]) -> str:
    """Render one configured tuple as stable inline code values."""
    return ", ".join(f"`{value}`" for value in values) or "none configured"
