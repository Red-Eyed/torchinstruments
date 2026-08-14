"""Typed boundary parsers for JSON artifacts exercised by tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict, cast


class JsonHistogramRecord(TypedDict):
    """Describe a lossless serialized histogram consumed by tests."""

    bin_edges: list[float]
    bin_counts: list[int]
    finite_count: int
    nonfinite_count: int
    underflow_count: int
    overflow_count: int
    minimum: float
    maximum: float
    sum: float
    sum_squares: float


class JsonErrorRecord(TypedDict):
    """Describe serialized collection-error fields consumed by tests."""

    count: int
    first_timestamp: str
    latest_timestamp: str
    module: str | dict[str, str]
    probe: str
    exception_type: str
    message: str


class JsonModuleRecord(TypedDict):
    """Describe module-catalog fields consumed by alias tests."""

    aliases: list[str]


class JsonSamplingRecord(TypedDict):
    """Describe serialized sampling-policy metadata."""

    type: str
    settings: dict[str, bool | float | int | str]


class JsonReducerRecord(TypedDict):
    """Describe serialized reducer configuration consumed by tests."""

    type: str
    settings: dict[str, bool | float | int | str | list[bool | float | int | str]]


class JsonCollectionRecord(TypedDict):
    """Describe serialized collection boundaries consumed by tests."""

    invocation_capture: str
    signals: list[str]
    scalar_reducers: list[JsonReducerRecord]
    histogram_reducers: list[JsonReducerRecord]


class JsonRunRecord(TypedDict):
    """Describe immutable run metadata consumed by schema tests."""

    schema_version: int
    created_at: str
    observer_version: str
    torch_version: str
    sampling: JsonSamplingRecord
    collection: JsonCollectionRecord


class JsonMetricPoint(TypedDict):
    """Describe one scalar value located at a sampled forward."""

    value: float
    sample_id: int
    timestamp: str


class JsonSeriesSummary(TypedDict):
    """Describe one serialized live temporal-indicator series."""

    count: int
    warmup_complete: bool
    first: JsonMetricPoint
    latest: JsonMetricPoint
    minimum: JsonMetricPoint
    maximum: JsonMetricPoint
    indicators: dict[str, float | int]


class JsonHistogramSummary(TypedDict):
    """Describe latest and aggregate live histogram fields consumed by tests."""

    samples: int
    latest: JsonHistogramRecord
    aggregate: JsonHistogramRecord | dict[str, str]


class JsonLiveTensorRecord(TypedDict):
    """Describe one live tensor-path summary consumed by tests."""

    observations: int
    shape: list[int]
    shape_changes: int
    dtype: str
    device: str
    numel: int
    latest_statistics: dict[str, float]
    statistics: dict[str, JsonSeriesSummary]
    histograms: dict[str, JsonHistogramSummary]
    latest_unavailable_statistics: dict[str, str]
    latest_unavailable_histograms: dict[str, str]


class JsonLiveModuleCall(TypedDict):
    """Describe one live module-call position consumed by tests."""

    call_index: int
    outputs: dict[str, JsonLiveTensorRecord]
    output_gradients: dict[str, JsonLiveTensorRecord]


class JsonIndicatorConfiguration(TypedDict):
    """Describe serialized temporal-analysis settings consumed by tests."""

    fast_ema_alpha: float
    slow_ema_alpha: float
    change_volatility_alpha: float
    momentum_horizons: list[int]
    recent_window: int
    cusum_allowance: float
    warmup_observations: int
    max_series: int
    max_tensor_paths: int
    max_module_calls: int
    max_histograms: int
    max_error_summaries: int
    temporal_metrics: list[str]


class JsonLiveStatsRecord(TypedDict):
    """Describe the canonical single-file telemetry shape consumed by tests."""

    schema_version: int
    updated_at: str
    run: JsonRunRecord
    module_catalog: dict[str, JsonModuleRecord]
    indicator_configuration: JsonIndicatorConfiguration
    samples_observed: int
    backward_samples_observed: int
    observer_statistics: dict[str, JsonSeriesSummary]
    layers: dict[str, list[JsonLiveModuleCall]]
    errors: list[JsonErrorRecord]
    dropped_series: int
    dropped_tensor_path_observations: int
    dropped_module_call_observations: int
    dropped_histogram_observations: int
    dropped_error_summaries: int


def read_stats(path: Path) -> JsonLiveStatsRecord:
    """Parse the canonical live file after validating its required boundaries."""
    value = _read_object(path)
    required = {
        "schema_version",
        "updated_at",
        "run",
        "module_catalog",
        "indicator_configuration",
        "samples_observed",
        "backward_samples_observed",
        "observer_statistics",
        "layers",
        "errors",
        "dropped_series",
        "dropped_tensor_path_observations",
        "dropped_module_call_observations",
        "dropped_histogram_observations",
        "dropped_error_summaries",
    }
    if not required.issubset(value):
        raise ValueError(f"live stats are missing fields: {sorted(required - value.keys())}")
    return cast(JsonLiveStatsRecord, value)


def _read_object(path: Path) -> dict[str, object]:
    """Read one JSON object and reject non-object roots at the test boundary."""
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value
