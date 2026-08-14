"""Typed boundary parsers for JSON artifacts exercised by tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypeAlias, TypedDict, cast


class JsonAbsent(TypedDict):
    """Describe one serialized reason-carrying unavailable value."""

    status: str
    reason: str


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
    module: str | JsonAbsent
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
    aggregate: JsonHistogramRecord | JsonAbsent


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


class JsonEvidenceValue(TypedDict):
    """Describe one explicitly named scalar supporting a ranked finding."""

    name: str
    value: float | int


class JsonFinding(TypedDict):
    """Describe one bounded diagnostic finding consumed by behavior tests."""

    category: str
    rank: int
    ranking_score: float
    ranking_basis: str
    source_rank: int
    module: str
    module_type: str
    call_index: int
    signal: str
    tensor_path: str
    metric: str
    observations: int
    warmup_complete: bool
    first: JsonMetricPoint
    latest: JsonMetricPoint
    minimum: JsonMetricPoint
    maximum: JsonMetricPoint
    evidence: list[JsonEvidenceValue]
    interpretation: str


class JsonCategoryFindings(TypedDict):
    """Describe one independently ranked diagnostic category."""

    category: str
    findings: list[JsonFinding]


class JsonCategoryCount(TypedDict):
    """Describe the complete number of candidates in one category."""

    category: str
    count: int


class JsonRankRecord(TypedDict):
    """Describe report ownership for one distributed process."""

    rank: int
    world_size: int


class JsonReportConfiguration(TypedDict):
    """Describe exact report size and selection limits."""

    max_bytes: int
    top_k_per_category: int
    markdown_findings_per_category: int
    max_errors: int
    max_error_message_chars: int


class JsonReportCoverage(TypedDict):
    """Describe measured and omitted evidence in one bounded report."""

    selected_modules: int
    tensor_paths: int
    temporal_series: int
    histograms: int
    samples_observed: int
    backward_samples_observed: int
    candidates_by_category: list[JsonCategoryCount]
    findings_returned: int
    findings_omitted: int
    errors_returned: int
    errors_omitted: int
    report_truncated_by_byte_budget: bool
    dropped_series: int
    dropped_tensor_path_observations: int
    dropped_module_call_observations: int
    dropped_histogram_observations: int
    dropped_error_summaries: int


class JsonReportRecord(TypedDict):
    """Describe the complete bounded report schema consumed by tests."""

    report_schema_version: int
    telemetry_schema_version: int
    updated_at: str
    run: JsonRunRecord
    rank: JsonRankRecord
    report_configuration: JsonReportConfiguration
    indicator_configuration: JsonIndicatorConfiguration
    coverage: JsonReportCoverage
    findings: list[JsonCategoryFindings]
    errors: list[JsonErrorRecord]


class JsonMergedCoverage(TypedDict):
    """Describe completeness and selection for a merged DDP report."""

    expected_ranks: int
    ranks_present: list[int]
    rank_coverage_complete: bool
    source_reports_truncated: int
    findings_returned: int
    findings_omitted_after_global_ranking: int
    report_truncated_by_byte_budget: bool


class JsonMergedReportRecord(TypedDict):
    """Describe the bounded report produced from independent rank reports."""

    report_schema_version: int
    updated_at: str
    report_configuration: JsonReportConfiguration
    coverage: JsonMergedCoverage
    findings: list[JsonCategoryFindings]


JsonDocument: TypeAlias = JsonLiveStatsRecord | JsonReportRecord | JsonMergedReportRecord


def require_histogram(value: JsonHistogramRecord | JsonAbsent) -> JsonHistogramRecord:
    """Narrow an available aggregate histogram or fail with its absence reason."""
    if "status" in value:
        absent = cast(JsonAbsent, value)
        raise AssertionError(f"expected histogram data, got {absent['reason']}")
    return cast(JsonHistogramRecord, value)


def read_stats(path: Path) -> JsonLiveStatsRecord:
    """Parse explicitly enabled live details after validating required boundaries."""
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


def read_report(path: Path) -> JsonReportRecord:
    """Parse the bounded LLM report after validating its typed top-level boundary."""
    value = _read_object(path)
    required = {
        "report_schema_version",
        "telemetry_schema_version",
        "updated_at",
        "run",
        "rank",
        "report_configuration",
        "indicator_configuration",
        "coverage",
        "findings",
        "errors",
    }
    if not required.issubset(value):
        raise ValueError(f"report is missing fields: {sorted(required - value.keys())}")
    return cast(JsonReportRecord, value)


def read_merged_report(path: Path) -> JsonMergedReportRecord:
    """Parse a merged rank report after validating its typed top-level boundary."""
    value = _read_object(path)
    required = {
        "report_schema_version",
        "updated_at",
        "report_configuration",
        "coverage",
        "findings",
    }
    if not required.issubset(value):
        raise ValueError(f"merged report is missing fields: {sorted(required - value.keys())}")
    return cast(JsonMergedReportRecord, value)


def _read_object(path: Path) -> JsonDocument:
    """Read one JSON object and reject non-object roots at the test boundary."""
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return cast(JsonDocument, value)
