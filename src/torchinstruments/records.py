"""Typed transient records for PyTorch telemetry collection."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import TypeAlias

SCHEMA_VERSION = 7


class ModuleMode(StrEnum):
    """Identify the selected module's mode at invocation time."""

    TRAIN = "train"
    EVAL = "eval"


@dataclass(frozen=True)
class ExecutionContext:
    """Keep module mode independent of autograd recording and later mode changes."""

    mode: ModuleMode
    grad_enabled: bool


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
    settings: dict[str, JsonScalar]


@dataclass(frozen=True)
class ReducerRecord:
    """Describe one configured reducer using stable JSON-compatible settings."""

    type: str
    settings: dict[str, JsonSetting]


@dataclass(frozen=True)
class CollectionRecord:
    """Describe the signal boundaries and reducers configured for one run."""

    invocation_capture: str
    signals: tuple[str, ...]
    scalar_reducers: tuple[ReducerRecord, ...]
    histogram_reducers: tuple[ReducerRecord, ...]
    histogram_modules: tuple[str, ...] = ()


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
    stats: dict[str, float]
    unavailable_stats: dict[str, str]
    histograms: dict[str, HistogramRecord]
    unavailable_histograms: dict[str, str]


@dataclass(frozen=True)
class ModuleCallRecord:
    """Keep one module invocation distinct from other calls to a shared module."""

    call_index: int
    context: ExecutionContext
    outputs: dict[str, TensorRecord]
    output_gradients: dict[str, TensorRecord]


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
    modules: dict[str, tuple[ModuleCallRecord, ...]]
    errors: tuple[ErrorRecord, ...]
