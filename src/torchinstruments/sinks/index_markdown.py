"""Bounded human and LLM guidance for one live telemetry record."""

from __future__ import annotations

import json
from datetime import UTC

from torchinstruments.records import LiveStatsRecord


def render_run_index(stats: LiveStatsRecord) -> str:
    """Render a deterministic guide to the canonical live statistics file."""
    run = stats.run
    sampling_settings = json.dumps(dict(run.sampling.settings), sort_keys=True)
    created_at = run.created_at.astimezone(UTC).isoformat().replace("+00:00", "Z")
    statistic_names = _observed_statistic_names(stats)
    indicator_names = _observed_indicator_names(stats)
    lines = [
        "# TorchInstruments live run index",
        "",
        "> Start here. `stats.json` is the single canonical telemetry record and is updated",
        "> atomically after sampled forward and backward observations.",
        "",
        "## Run overview",
        "",
        f"- Schema version: `{stats.schema_version}`",
        f"- TorchInstruments version: `{run.observer_version}`",
        f"- PyTorch version: `{run.torch_version}`",
        f"- Created at: `{created_at}`",
        f"- Sampling policy: `{run.sampling.type}` with settings `{sampling_settings}`",
        f"- Invocation capture: `{run.collection.invocation_capture}`",
        f"- Selected modules: `{len(stats.module_catalog)}`",
        f"- Momentum horizons: `{list(stats.indicator_configuration.momentum_horizons)}`",
        f"- Recent indicator window: `{stats.indicator_configuration.recent_window}` observations",
        f"- Full warm-up floor: `{stats.indicator_configuration.warmup_observations}` observations",
        f"- Sampled forwards observed: `{stats.samples_observed}`",
        f"- Correlated backwards observed: `{stats.backward_samples_observed}`",
        f"- Dropped metric series: `{stats.dropped_series}`",
        f"- Dropped tensor-path observations: `{stats.dropped_tensor_path_observations}`",
        f"- Dropped module-call observations: `{stats.dropped_module_call_observations}`",
        f"- Dropped histogram observations: `{stats.dropped_histogram_observations}`",
        f"- Dropped error summaries: `{stats.dropped_error_summaries}`",
        "",
        "## Files",
        "",
        "- `stats.json`: run metadata, module catalog, live per-layer indicators, histograms,",
        "  observer overhead, and bounded error summaries.",
        "- `index.md`: this derived guide; it contains no additional telemetry.",
        "",
        "No per-sample snapshot files are written. Extrema in `stats.json` retain the sample ID",
        "and timestamp where they occurred.",
        "",
        "## Indicators observed so far",
        "",
        _render_names("Point-in-time tensor statistics", statistic_names),
        _render_names("Temporal indicators", indicator_names),
        "",
        "Each tensor path separates forward `outputs` from backward `output_gradients`. Shared",
        "modules retain distinct `call_index` entries. Every scalar series contains first, latest,",
        "minimum, maximum, count, warm-up state, and bounded trend, momentum, volatility,",
        "oscillation, autocorrelation, drawdown, runup, and CUSUM indicators.",
        "",
        "## Use this run with an LLM",
        "",
        "```text",
        "Read index.md and stats.json. Find layers whose forward distributions or backward",
        "gradients show scale drift, heavy-tail growth, skew, non-finite values, high volatility,",
        "persistent momentum, regime change, or collapse from a previous extreme.",
        "For every finding report: (1) exact module, call index, tensor path, statistic, and",
        "indicator values; (2) measured interpretation; (3) plausible mechanisms; (4) missing",
        "evidence; and (5) the smallest controlled experiment. Treat indicators without completed",
        "warm-up as weak evidence and distinguish measurements from hypotheses.",
        "```",
        "",
        "## Evidence limits",
        "",
        "TorchInstruments observes selected-module outputs and their correlated output gradients.",
        "It does not observe loss, labels, optimizer updates, module inputs, parameters, or",
        "parameter gradients unless those signals are added separately. Online indicators retain",
        "bounded distribution and temporal evidence, not the complete historical time series.",
        "",
    ]
    return "\n".join(lines)


def _observed_statistic_names(stats: LiveStatsRecord) -> frozenset[str]:
    """Collect scalar statistic names without retaining any measurement values."""
    names: set[str] = set()
    for calls in stats.layers.values():
        for call in calls:
            for tensors in (call.outputs, call.output_gradients):
                for tensor in tensors.values():
                    names.update(tensor.statistics)
    return frozenset(names)


def _observed_indicator_names(stats: LiveStatsRecord) -> frozenset[str]:
    """Collect temporal indicator names from the first available scalar series."""
    for calls in stats.layers.values():
        for call in calls:
            for tensors in (call.outputs, call.output_gradients):
                for tensor in tensors.values():
                    for series in tensor.statistics.values():
                        return frozenset(series.indicators)
    return frozenset()


def _render_names(label: str, names: frozenset[str]) -> str:
    """Render a sorted bounded name list or an explicit empty state."""
    if not names:
        return f"- {label}: none observed yet"
    return f"- {label}: " + ", ".join(f"`{name}`" for name in sorted(names))
