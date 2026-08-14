"""Typed, human-named records for bounded research-diagnostic reports."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import NamedTuple

from torchinstruments.records import (
    ErrorSummaryRecord,
    IndicatorConfigurationRecord,
    MetricPointRecord,
    RunRecord,
)

REPORT_SCHEMA_VERSION = 1


class FindingCategory(StrEnum):
    """Name independent diagnostic questions without an opaque health score."""

    ACTIVATION_SCALE_DRIFT = "activation_scale_drift"
    GRADIENT_SCALE_CHANGE = "gradient_scale_change"
    HEAVY_TAIL_GROWTH = "heavy_tail_growth"
    NONFINITE_VALUES = "nonfinite_values"
    ZERO_FRACTION_GROWTH = "zero_fraction_growth"
    VOLATILITY = "volatility"
    OSCILLATION = "oscillation"
    REGIME_CHANGE = "regime_change"


@dataclass(frozen=True)
class ReportConfig:
    """Bound report cost and the number of findings retained per category."""

    max_bytes: int = 256_000
    top_k_per_category: int = 20
    markdown_findings_per_category: int = 5
    max_errors: int = 20
    max_error_message_chars: int = 500

    def __post_init__(self) -> None:
        """Reject limits that cannot produce a useful self-describing report."""
        for name, value in (
            ("max_bytes", self.max_bytes),
            ("top_k_per_category", self.top_k_per_category),
            ("markdown_findings_per_category", self.markdown_findings_per_category),
            ("max_errors", self.max_errors),
            ("max_error_message_chars", self.max_error_message_chars),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_bytes < 8_192:
            raise ValueError("max_bytes must be at least 8192")


@dataclass(frozen=True)
class RankRecord:
    """Describe report ownership without implying cross-rank aggregation."""

    rank: int
    world_size: int


@dataclass(frozen=True)
class ReportConfigurationRecord:
    """Persist the exact limits that control report selection and truncation."""

    max_bytes: int
    top_k_per_category: int
    markdown_findings_per_category: int
    max_errors: int
    max_error_message_chars: int


class EvidenceValueRecord(NamedTuple):
    """Bind one descriptive evidence field name to its finite scalar value."""

    name: str
    value: float | int


class CategoryCountRecord(NamedTuple):
    """Store one finding category and its complete candidate count."""

    category: FindingCategory
    count: int


@dataclass(frozen=True)
class FindingRecord:
    """Preserve one locally ranked observation and its directly supporting evidence."""

    category: FindingCategory
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
    first: MetricPointRecord
    latest: MetricPointRecord
    minimum: MetricPointRecord
    maximum: MetricPointRecord
    evidence: tuple[EvidenceValueRecord, ...]
    interpretation: str


class CategoryFindingsRecord(NamedTuple):
    """Group ranked findings under one explicit diagnostic category."""

    category: FindingCategory
    findings: tuple[FindingRecord, ...]


@dataclass(frozen=True)
class ReportCoverageRecord:
    """Expose measured scope and every reason findings may be absent."""

    selected_modules: int
    tensor_paths: int
    temporal_series: int
    histograms: int
    samples_observed: int
    backward_samples_observed: int
    candidates_by_category: tuple[CategoryCountRecord, ...]
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


@dataclass(frozen=True)
class ReportRecord:
    """Represent the complete bounded JSON artifact intended for LLM analysis."""

    report_schema_version: int
    telemetry_schema_version: int
    updated_at: datetime
    run: RunRecord
    rank: RankRecord
    report_configuration: ReportConfigurationRecord
    indicator_configuration: IndicatorConfigurationRecord
    coverage: ReportCoverageRecord
    findings: tuple[CategoryFindingsRecord, ...]
    errors: tuple[ErrorSummaryRecord, ...]


@dataclass(frozen=True)
class MergedReportCoverageRecord:
    """Describe rank completeness and bounded selection in one merged report."""

    expected_ranks: int
    ranks_present: tuple[int, ...]
    rank_coverage_complete: bool
    source_reports_truncated: int
    findings_returned: int
    findings_omitted_after_global_ranking: int
    report_truncated_by_byte_budget: bool


@dataclass(frozen=True)
class MergedReportRecord:
    """Represent one bounded global ranking streamed from independent rank reports."""

    report_schema_version: int
    updated_at: datetime
    report_configuration: ReportConfigurationRecord
    coverage: MergedReportCoverageRecord
    findings: tuple[CategoryFindingsRecord, ...]
