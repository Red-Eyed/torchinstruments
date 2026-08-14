"""Typed, JSON-compatible domain records for telemetry artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import TypeAlias

SCHEMA_VERSION = 4


@dataclass(frozen=True)
class Absent:
    """Carry the reason a canonical telemetry value is unavailable."""

    reason: str


JsonScalar: TypeAlias = bool | float | int | str
JsonSetting: TypeAlias = JsonScalar | tuple[JsonScalar, ...]


@dataclass(frozen=True)
class SamplingRecord:
    """Describe a sampling policy without coupling records to its implementation."""

    type: str
    settings: Mapping[str, JsonScalar]


@dataclass(frozen=True)
class ReducerRecord:
    """Describe one configured reducer using stable JSON-compatible settings."""

    type: str
    settings: Mapping[str, JsonSetting]


@dataclass(frozen=True)
class CollectionRecord:
    """Describe the signal boundaries and reducers configured for one run."""

    invocation_capture: str
    signals: tuple[str, ...]
    scalar_reducers: tuple[ReducerRecord, ...]
    histogram_reducers: tuple[ReducerRecord, ...]


@dataclass(frozen=True)
class RunRecord:
    """Store immutable metadata shared by every sample in one observer run."""

    schema_version: int
    created_at: datetime
    torch_version: str
    observer_version: str
    sampling: SamplingRecord
    collection: CollectionRecord


@dataclass(frozen=True)
class ModuleRecord:
    """Describe one uniquely hooked module and all names that alias it."""

    type: str
    aliases: tuple[str, ...]
    parameter_count: int
    trainable_parameter_count: int


@dataclass(frozen=True)
class HistogramRecord:
    """Store a finite-value histogram without retaining its source tensor.

    Regular bins cover consecutive intervals defined by ``bin_edges``. Values below and above
    that range remain visible through explicit underflow and overflow counts. The compact moments
    are sufficient to reproduce TensorBoard's pre-aggregated histogram representation.
    """

    bin_edges: tuple[float, ...]
    bin_counts: tuple[int, ...]
    finite_count: int
    nonfinite_count: int
    underflow_count: int
    overflow_count: int
    minimum: float
    maximum: float
    sum: float
    sum_squares: float


@dataclass(frozen=True)
class TensorRecord:
    """Represent tensor metadata and compact scalar diagnostics without raw data."""

    shape: tuple[int, ...]
    dtype: str
    device: str
    numel: int
    stats: Mapping[str, float]
    unavailable_stats: Mapping[str, str]
    histograms: Mapping[str, HistogramRecord]
    unavailable_histograms: Mapping[str, str]


@dataclass(frozen=True)
class ModuleCallRecord:
    """Keep one module invocation distinct from other calls to a shared module."""

    call_index: int
    outputs: Mapping[str, TensorRecord]
    output_gradients: Mapping[str, TensorRecord]


@dataclass(frozen=True)
class ErrorRecord:
    """Persist an isolated instrumentation failure alongside model telemetry."""

    timestamp: datetime
    module: str | Absent
    probe: str
    exception_type: str
    message: str


class SampleState(StrEnum):
    """Identify whether a transient sample contains forward data or new gradients."""

    FORWARD_COMPLETE = "forward_complete"
    BACKWARD_OBSERVED = "backward_observed"


@dataclass(frozen=True)
class SampleRecord:
    """Carry one transient sampled-forward lifecycle event to configured sinks."""

    schema_version: int
    sample_id: int
    forward_index: int
    timestamp: datetime
    state: SampleState
    collection_duration_ms: float
    modules: Mapping[str, tuple[ModuleCallRecord, ...]]
    errors: tuple[ErrorRecord, ...]


@dataclass(frozen=True)
class MetricPointRecord:
    """Locate one scalar observation in sampled-forward time."""

    value: float
    sample_id: int
    timestamp: datetime


IndicatorValue: TypeAlias = float | int


@dataclass(frozen=True)
class SeriesSummaryRecord:
    """Describe one bounded live scalar series and its derived indicators."""

    count: int
    warmup_complete: bool
    first: MetricPointRecord
    latest: MetricPointRecord
    minimum: MetricPointRecord
    maximum: MetricPointRecord
    indicators: Mapping[str, IndicatorValue]


@dataclass(frozen=True)
class HistogramSummaryRecord:
    """Retain the latest histogram and an exact merge when bin edges remain stable."""

    samples: int
    latest: HistogramRecord
    aggregate: HistogramRecord | Absent


@dataclass(frozen=True)
class LiveTensorRecord:
    """Describe current tensor metadata and bounded indicators for one tensor path."""

    observations: int
    shape: tuple[int, ...]
    shape_changes: int
    dtype: str
    device: str
    numel: int
    latest_statistics: Mapping[str, float]
    statistics: Mapping[str, SeriesSummaryRecord]
    histograms: Mapping[str, HistogramSummaryRecord]
    latest_unavailable_statistics: Mapping[str, str]
    latest_unavailable_histograms: Mapping[str, str]


@dataclass(frozen=True)
class LiveModuleCallRecord:
    """Keep live forward and backward summaries separate for one module call position."""

    call_index: int
    outputs: Mapping[str, LiveTensorRecord]
    output_gradients: Mapping[str, LiveTensorRecord]


@dataclass(frozen=True)
class ErrorSummaryRecord:
    """Aggregate repeated instrumentation failures without an unbounded event log."""

    count: int
    first_timestamp: datetime
    latest_timestamp: datetime
    module: str | Absent
    probe: str
    exception_type: str
    message: str


@dataclass(frozen=True)
class IndicatorConfigurationRecord:
    """Describe the exact bounded temporal-analysis configuration for one run."""

    fast_ema_alpha: float
    slow_ema_alpha: float
    change_volatility_alpha: float
    momentum_horizons: tuple[int, ...]
    recent_window: int
    cusum_allowance: float
    warmup_observations: int
    max_series: int
    max_tensor_paths: int
    max_module_calls: int
    max_histograms: int
    max_error_summaries: int
    temporal_metrics: tuple[str, ...]


@dataclass(frozen=True)
class LiveStatsRecord:
    """Represent the complete canonical live telemetry file for one observer run."""

    schema_version: int
    updated_at: datetime
    run: RunRecord
    module_catalog: Mapping[str, ModuleRecord]
    indicator_configuration: IndicatorConfigurationRecord
    samples_observed: int
    backward_samples_observed: int
    observer_statistics: Mapping[str, SeriesSummaryRecord]
    layers: Mapping[str, tuple[LiveModuleCallRecord, ...]]
    errors: tuple[ErrorSummaryRecord, ...]
    dropped_series: int
    dropped_tensor_path_observations: int
    dropped_module_call_observations: int
    dropped_histogram_observations: int
    dropped_error_summaries: int
