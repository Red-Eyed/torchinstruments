"""Bounded online indicators for live per-layer telemetry."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from itertools import pairwise

from torchinstruments.records import (
    SCHEMA_VERSION,
    Absent,
    ErrorRecord,
    ErrorSummaryRecord,
    HistogramRecord,
    HistogramSummaryRecord,
    IndicatorConfigurationRecord,
    IndicatorValue,
    LiveModuleCallRecord,
    LiveStatsRecord,
    LiveTensorRecord,
    MetricPointRecord,
    ModuleRecord,
    RunRecord,
    SampleRecord,
    SampleState,
    SeriesSummaryRecord,
    TensorRecord,
)

_EPSILON = 1e-12


@dataclass(frozen=True)
class IndicatorConfig:
    """Configure bounded temporal indicators calculated for every scalar series.

    EMA coefficients are fractions in ``(0, 1]``. Momentum horizons and the recent window are
    measured in observations of an individual metric, not optimizer steps. The ``max_*`` fields
    bound temporal series, dynamic output structure, histogram identities, and errors.
    """

    fast_ema_alpha: float = 0.25
    slow_ema_alpha: float = 0.05
    change_volatility_alpha: float = 0.1
    momentum_horizons: tuple[int, ...] = (1, 5, 20)
    recent_window: int = 20
    cusum_allowance: float = 0.5
    warmup_observations: int = 20
    max_series: int = 100_000
    max_tensor_paths: int = 20_000
    max_module_calls: int = 10_000
    max_histograms: int = 20_000
    max_error_summaries: int = 100
    temporal_metrics: tuple[str, ...] = (
        "mean",
        "std",
        "rms",
        "finite_fraction",
        "zero_fraction",
        "skewness",
        "excess_kurtosis",
        "p99_abs",
        "p999_abs",
        "max_to_rms",
        "p99_abs_to_rms",
        "p999_abs_to_rms",
        "normalized_magnitude_entropy",
    )

    def __post_init__(self) -> None:
        """Reject configuration that would make indicator semantics ambiguous."""
        for name, value in (
            ("fast_ema_alpha", self.fast_ema_alpha),
            ("slow_ema_alpha", self.slow_ema_alpha),
            ("change_volatility_alpha", self.change_volatility_alpha),
        ):
            if isinstance(value, bool) or not 0 < value <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        invalid_horizon = any(
            isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0
            for horizon in self.momentum_horizons
        )
        if not self.momentum_horizons or invalid_horizon:
            raise ValueError("momentum_horizons must contain positive integers")
        if len(set(self.momentum_horizons)) != len(self.momentum_horizons):
            raise ValueError("momentum_horizons must not contain duplicates")
        for name, value in (
            ("recent_window", self.recent_window),
            ("warmup_observations", self.warmup_observations),
            ("max_series", self.max_series),
            ("max_tensor_paths", self.max_tensor_paths),
            ("max_module_calls", self.max_module_calls),
            ("max_histograms", self.max_histograms),
            ("max_error_summaries", self.max_error_summaries),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be greater than zero")
        if isinstance(self.cusum_allowance, bool) or self.cusum_allowance < 0:
            raise ValueError("cusum_allowance must not be negative")
        if len(set(self.temporal_metrics)) != len(self.temporal_metrics):
            raise ValueError("temporal_metrics must not contain duplicates")


_DEFAULT_INDICATOR_CONFIG = IndicatorConfig()


class _OnlineSeries:
    """Maintain bounded technical indicators for one scalar metric."""

    def __init__(self, config: IndicatorConfig) -> None:
        """Initialize empty constant-size state and one bounded recent window."""
        self._config = config
        window_size = max(config.recent_window, *config.momentum_horizons) + 1
        self._recent_values: deque[float] = deque(maxlen=window_size)
        self._recent_deltas: deque[float] = deque(maxlen=config.recent_window)
        self._count = 0
        self._first: MetricPointRecord | None = None
        self._latest: MetricPointRecord | None = None
        self._minimum: MetricPointRecord | None = None
        self._maximum: MetricPointRecord | None = None
        self._mean = 0.0
        self._m2 = 0.0
        self._mean_x = 0.0
        self._sxx = 0.0
        self._sxy = 0.0
        self._fast_ema = 0.0
        self._slow_ema = 0.0
        self._ew_delta_mean = 0.0
        self._ew_delta_variance = 0.0
        self._ew_gain = 0.0
        self._ew_loss = 0.0
        self._cusum_positive = 0.0
        self._cusum_negative = 0.0
        self._latest_z_score: float | Absent = Absent("at least two varying values are required")
        self._highest = 0.0
        self._lowest = 0.0
        self._maximum_drawdown = 0.0
        self._maximum_runup = 0.0
        self._consecutive_increases = 0
        self._consecutive_decreases = 0

    def update(self, value: float, *, sample_id: int, timestamp: datetime) -> None:
        """Incorporate one finite scalar observation into every streaming indicator."""
        point = MetricPointRecord(value=value, sample_id=sample_id, timestamp=timestamp)
        if self._count == 0:
            self._initialize_first(point)
            return

        previous = self._require_point(self._latest)
        previous_mean = self._mean
        previous_std = math.sqrt(self._m2 / self._count)
        self._latest_z_score = _standardized_difference(value, previous_mean, previous_std)
        delta = value - previous.value

        self._count += 1
        self._latest = point
        self._minimum = min(self._require_point(self._minimum), point, key=lambda item: item.value)
        self._maximum = max(self._require_point(self._maximum), point, key=lambda item: item.value)
        self._update_moments(value, float(sample_id))
        self._update_ema(value)
        self._update_change_indicators(delta)
        self._update_extremes(value)
        self._recent_values.append(value)
        self._recent_deltas.append(delta)

    def to_record(self) -> SeriesSummaryRecord:
        """Freeze the current bounded state into an LLM-readable record."""
        if self._count == 0:
            raise RuntimeError("cannot serialize an empty online series")
        return SeriesSummaryRecord(
            count=self._count,
            warmup_complete=self._count >= self._full_warmup_observations(),
            first=self._require_point(self._first),
            latest=self._require_point(self._latest),
            minimum=self._require_point(self._minimum),
            maximum=self._require_point(self._maximum),
            indicators=self._indicators(),
        )

    def _initialize_first(self, point: MetricPointRecord) -> None:
        """Establish every state variable whose meaning requires an observation."""
        self._count = 1
        self._first = point
        self._latest = point
        self._minimum = point
        self._maximum = point
        self._mean = point.value
        self._mean_x = float(point.sample_id)
        self._fast_ema = point.value
        self._slow_ema = point.value
        self._highest = point.value
        self._lowest = point.value
        self._recent_values.append(point.value)

    def _full_warmup_observations(self) -> int:
        """Return the count required for every configured bounded indicator."""
        return max(
            self._config.warmup_observations,
            max(self._config.momentum_horizons) + 1,
            3,
        )

    def _update_moments(self, value: float, sample_id: float) -> None:
        """Update population moments and stable online linear-regression sums."""
        delta_y = value - self._mean
        delta_x = sample_id - self._mean_x
        self._mean += delta_y / self._count
        self._mean_x += delta_x / self._count
        self._m2 += delta_y * (value - self._mean)
        self._sxx += delta_x * (sample_id - self._mean_x)
        self._sxy += delta_x * (value - self._mean)

    def _update_ema(self, value: float) -> None:
        """Update fast and slow exponentially weighted levels."""
        fast_alpha = self._config.fast_ema_alpha
        slow_alpha = self._config.slow_ema_alpha
        self._fast_ema += fast_alpha * (value - self._fast_ema)
        self._slow_ema += slow_alpha * (value - self._slow_ema)

    def _update_change_indicators(self, delta: float) -> None:
        """Update volatility, directional balance, runs, and change accumulation."""
        alpha = self._config.change_volatility_alpha
        previous_delta_mean = self._ew_delta_mean
        self._ew_delta_mean += alpha * (delta - self._ew_delta_mean)
        self._ew_delta_variance = (1 - alpha) * (
            self._ew_delta_variance
            + alpha * (delta - previous_delta_mean) * (delta - previous_delta_mean)
        )
        self._ew_gain = (1 - alpha) * self._ew_gain + alpha * max(delta, 0.0)
        self._ew_loss = (1 - alpha) * self._ew_loss + alpha * max(-delta, 0.0)

        if isinstance(self._latest_z_score, float):
            allowance = self._config.cusum_allowance
            self._cusum_positive = max(0.0, self._cusum_positive + self._latest_z_score - allowance)
            self._cusum_negative = max(0.0, self._cusum_negative - self._latest_z_score - allowance)

        self._consecutive_increases = self._consecutive_increases + 1 if delta > 0 else 0
        self._consecutive_decreases = self._consecutive_decreases + 1 if delta < 0 else 0

    def _update_extremes(self, value: float) -> None:
        """Track absolute drawdown from a peak and runup from a trough."""
        self._highest = max(self._highest, value)
        self._lowest = min(self._lowest, value)
        self._maximum_drawdown = max(self._maximum_drawdown, self._highest - value)
        self._maximum_runup = max(self._maximum_runup, value - self._lowest)

    def _indicators(self) -> Mapping[str, IndicatorValue]:
        """Calculate descriptive indicator values from bounded running state."""
        variance = max(self._m2 / self._count, 0.0)
        standard_deviation = math.sqrt(variance)
        candidates: dict[str, IndicatorValue | Absent] = {
            "mean": self._mean,
            "standard_deviation": standard_deviation,
            "root_mean_square": math.sqrt(variance + self._mean * self._mean),
            "coefficient_of_variation": _safe_ratio(standard_deviation, abs(self._mean)),
            "linear_slope_per_sample": self._linear_slope(),
            "slope_r_squared": self._slope_r_squared(),
        }
        candidates.update(
            {
                "fast_ema": self._fast_ema,
                "slow_ema": self._slow_ema,
                "fast_slow_gap": self._fast_ema - self._slow_ema,
                "fast_slow_relative_gap": _safe_ratio(
                    self._fast_ema - self._slow_ema, abs(self._slow_ema)
                ),
                "exponentially_weighted_change_mean": self._ew_delta_mean,
                "exponentially_weighted_change_volatility": math.sqrt(
                    max(self._ew_delta_variance, 0.0)
                ),
                "latest_z_score": self._latest_z_score,
                "historical_range_position": self._historical_range_position(),
                "maximum_drawdown": self._maximum_drawdown,
                "maximum_runup": self._maximum_runup,
                "up_down_balance": self._up_down_balance(),
                "cusum_positive": self._cusum_positive,
                "cusum_negative": self._cusum_negative,
                "cusum_change_score": max(self._cusum_positive, self._cusum_negative),
                "lag1_autocorrelation": _lag_one_autocorrelation(tuple(self._recent_values)),
                "oscillation_fraction": _oscillation_fraction(tuple(self._recent_deltas)),
                "consecutive_increases": self._consecutive_increases,
                "consecutive_decreases": self._consecutive_decreases,
            }
        )
        candidates.update(self._momentum_indicators())
        return _available_indicators(candidates)

    def _linear_slope(self) -> IndicatorValue | Absent:
        """Return least-squares slope over all observations or an absence reason."""
        if self._sxx <= _EPSILON:
            return Absent("at least two distinct sample identifiers are required")
        return self._sxy / self._sxx

    def _slope_r_squared(self) -> IndicatorValue | Absent:
        """Return linear-trend fit strength without claiming causal significance."""
        denominator = self._sxx * self._m2
        if denominator <= _EPSILON:
            return Absent("both sample position and metric value must vary")
        return min(max((self._sxy * self._sxy) / denominator, 0.0), 1.0)

    def _historical_range_position(self) -> IndicatorValue | Absent:
        """Locate the latest value between the historical minimum and maximum."""
        minimum = self._require_point(self._minimum).value
        maximum = self._require_point(self._maximum).value
        latest = self._require_point(self._latest).value
        return _safe_ratio(latest - minimum, maximum - minimum)

    def _up_down_balance(self) -> IndicatorValue | Absent:
        """Return an RSI-like signed balance of exponentially weighted changes."""
        return _safe_ratio(self._ew_gain - self._ew_loss, self._ew_gain + self._ew_loss)

    def _momentum_indicators(self) -> Mapping[str, IndicatorValue | Absent]:
        """Calculate absolute and relative change over each configured horizon."""
        values = tuple(self._recent_values)
        latest = values[-1]
        indicators: dict[str, IndicatorValue | Absent] = {}
        for horizon in self._config.momentum_horizons:
            name = f"momentum_{horizon}_samples"
            relative_name = f"relative_momentum_{horizon}_samples"
            if len(values) <= horizon:
                reason = Absent(f"at least {horizon + 1} observations are required")
                indicators[name] = reason
                indicators[relative_name] = reason
                continue
            previous = values[-horizon - 1]
            change = latest - previous
            indicators[name] = change
            indicators[relative_name] = _safe_ratio(change, abs(previous))
        return indicators

    def _require_point(self, point: MetricPointRecord | None) -> MetricPointRecord:
        """Return initialized point state or expose an internal lifecycle violation."""
        if point is None:
            raise RuntimeError("online series point was not initialized")
        return point


class _OnlineHistogram:
    """Merge fixed-bin histograms while retaining the latest dynamic histogram."""

    def __init__(self, record: HistogramRecord) -> None:
        """Initialize latest and aggregate state from the first histogram."""
        self._samples = 1
        self._latest = record
        self._aggregate = record
        self._merge_failure: Absent | None = None

    def update(self, record: HistogramRecord) -> None:
        """Merge identical edges or mark changing bins as non-mergeable."""
        self._samples += 1
        self._latest = record
        if self._merge_failure is not None:
            return
        if record.bin_edges != self._aggregate.bin_edges:
            self._merge_failure = Absent(
                "histogram bin edges changed; configure a fixed value_range for live aggregation"
            )
            return
        self._aggregate = _merge_histograms(self._aggregate, record)

    def to_record(self) -> HistogramSummaryRecord:
        """Return latest distribution and exact aggregate or its absence reason."""
        aggregate: HistogramRecord | Absent = self._aggregate
        if self._merge_failure is not None:
            aggregate = self._merge_failure
        return HistogramSummaryRecord(
            samples=self._samples,
            latest=self._latest,
            aggregate=aggregate,
        )


class _OnlineTensor:
    """Aggregate metadata, scalar indicators, and histograms for one tensor path."""

    def __init__(
        self,
        tensor: TensorRecord,
        *,
        sample_id: int,
        timestamp: datetime,
        create_series: Callable[[str], _OnlineSeries | None],
        create_histogram: Callable[[HistogramRecord], _OnlineHistogram | None],
        temporal_metrics: frozenset[str],
    ) -> None:
        """Initialize one path and immediately observe its first tensor record."""
        self._observations = 0
        self._shape = tensor.shape
        self._shape_changes = 0
        self._dtype = tensor.dtype
        self._device = tensor.device
        self._numel = tensor.numel
        self._latest_statistics: Mapping[str, float] = {}
        self._statistics: dict[str, _OnlineSeries] = {}
        self._blocked_statistics: set[str] = set()
        self._histograms: dict[str, _OnlineHistogram] = {}
        self._temporal_metrics = temporal_metrics
        self._latest_unavailable_statistics: Mapping[str, str] = {}
        self._latest_unavailable_histograms: Mapping[str, str] = {}
        self.update(
            tensor,
            sample_id=sample_id,
            timestamp=timestamp,
            create_series=create_series,
            create_histogram=create_histogram,
        )

    def update(
        self,
        tensor: TensorRecord,
        *,
        sample_id: int,
        timestamp: datetime,
        create_series: Callable[[str], _OnlineSeries | None],
        create_histogram: Callable[[HistogramRecord], _OnlineHistogram | None],
    ) -> None:
        """Update current metadata and every bounded statistic series."""
        if self._observations and tensor.shape != self._shape:
            self._shape_changes += 1
        self._observations += 1
        self._shape = tensor.shape
        self._dtype = tensor.dtype
        self._device = tensor.device
        self._numel = tensor.numel
        self._latest_statistics = dict(tensor.stats)
        self._latest_unavailable_statistics = dict(tensor.unavailable_stats)
        if tensor.unavailable_histograms:
            self._latest_unavailable_histograms = dict(tensor.unavailable_histograms)

        for name, value in tensor.stats.items():
            if name not in self._temporal_metrics or name in self._blocked_statistics:
                continue
            series = self._statistics.get(name)
            if series is None:
                series = create_series(name)
                if series is None:
                    self._blocked_statistics.add(name)
                    continue
                self._statistics[name] = series
            series.update(value, sample_id=sample_id, timestamp=timestamp)

        for name, histogram in tensor.histograms.items():
            if name in self._latest_unavailable_histograms:
                mutable_unavailable = dict(self._latest_unavailable_histograms)
                mutable_unavailable.pop(name)
                self._latest_unavailable_histograms = mutable_unavailable
            existing = self._histograms.get(name)
            if existing is None:
                created = create_histogram(histogram)
                if created is not None:
                    self._histograms[name] = created
            else:
                existing.update(histogram)

    def to_record(self) -> LiveTensorRecord:
        """Freeze this tensor path into current metadata and bounded indicators."""
        return LiveTensorRecord(
            observations=self._observations,
            shape=self._shape,
            shape_changes=self._shape_changes,
            dtype=self._dtype,
            device=self._device,
            numel=self._numel,
            latest_statistics=dict(self._latest_statistics),
            statistics={name: series.to_record() for name, series in self._statistics.items()},
            histograms={name: value.to_record() for name, value in self._histograms.items()},
            latest_unavailable_statistics=dict(self._latest_unavailable_statistics),
            latest_unavailable_histograms=dict(self._latest_unavailable_histograms),
        )


@dataclass
class _OnlineCall:
    """Own mutable forward and backward tensor paths for one call position."""

    call_index: int
    outputs: dict[str, _OnlineTensor] = field(default_factory=dict)
    output_gradients: dict[str, _OnlineTensor] = field(default_factory=dict)

    def to_record(self) -> LiveModuleCallRecord:
        """Freeze both signal directions without merging their meanings."""
        return LiveModuleCallRecord(
            call_index=self.call_index,
            outputs={name: value.to_record() for name, value in self.outputs.items()},
            output_gradients={
                name: value.to_record() for name, value in self.output_gradients.items()
            },
        )


@dataclass
class _OnlineError:
    """Count repeated errors that share stable diagnostic identity."""

    count: int
    first_timestamp: datetime
    latest_timestamp: datetime
    record: ErrorRecord

    def observe(self, timestamp: datetime) -> None:
        """Advance count and latest occurrence time."""
        self.count += 1
        self.latest_timestamp = timestamp

    def to_record(self) -> ErrorSummaryRecord:
        """Freeze bounded error recurrence information."""
        return ErrorSummaryRecord(
            count=self.count,
            first_timestamp=self.first_timestamp,
            latest_timestamp=self.latest_timestamp,
            module=self.record.module,
            probe=self.record.probe,
            exception_type=self.record.exception_type,
            message=self.record.message,
        )


class LiveAggregator:
    """Maintain one bounded live summary from transient forward/backward events."""

    def __init__(self, config: IndicatorConfig = _DEFAULT_INDICATOR_CONFIG) -> None:
        """Bind explicit indicator and memory limits without retaining model tensors."""
        self._config = config
        self._run: RunRecord | None = None
        self._modules: Mapping[str, ModuleRecord] = {}
        self._updated_at: datetime | None = None
        self._samples_observed = 0
        self._backward_samples_observed = 0
        self._layers: dict[str, dict[int, _OnlineCall]] = {}
        self._observer_statistics: dict[str, _OnlineSeries] = {}
        self._errors: dict[tuple[str, str, str, str], _OnlineError] = {}
        self._temporal_metrics = frozenset(config.temporal_metrics)
        self._series_count = 0
        self._tensor_path_count = 0
        self._module_call_count = 0
        self._histogram_count = 0
        self._dropped_series = 0
        self._dropped_tensor_path_observations = 0
        self._dropped_module_call_observations = 0
        self._dropped_histogram_observations = 0
        self._dropped_error_summaries = 0

    def initialize(self, run: RunRecord, modules: Mapping[str, ModuleRecord]) -> LiveStatsRecord:
        """Initialize immutable run context and return an empty live record."""
        if self._run is not None:
            raise RuntimeError("live aggregator is already initialized")
        self._run = run
        self._modules = dict(modules)
        self._updated_at = run.created_at
        return self.to_record()

    def observe(self, sample: SampleRecord) -> LiveStatsRecord:
        """Update only the lifecycle phase newly supplied by one transient sample."""
        self._require_run()
        self._updated_at = max(self._require_updated_at(), sample.timestamp)
        if sample.state is SampleState.FORWARD_COMPLETE:
            self._samples_observed = max(self._samples_observed, sample.sample_id + 1)
            self._observe_modules(sample, forward=True)
            duration_name = "forward_collection_duration_ms"
        else:
            self._backward_samples_observed += 1
            self._observe_modules(sample, forward=False)
            duration_name = "total_collection_duration_ms"
        self._observer_series(duration_name).update(
            sample.collection_duration_ms,
            sample_id=sample.sample_id,
            timestamp=sample.timestamp,
        )
        for error in sample.errors:
            self._observe_error(error)
        return self.to_record()

    def to_record(self) -> LiveStatsRecord:
        """Freeze current bounded state into the canonical JSON record."""
        run = self._require_run()
        return LiveStatsRecord(
            schema_version=SCHEMA_VERSION,
            updated_at=self._require_updated_at(),
            run=run,
            module_catalog=dict(self._modules),
            indicator_configuration=self._indicator_configuration(),
            samples_observed=self._samples_observed,
            backward_samples_observed=self._backward_samples_observed,
            observer_statistics={
                name: series.to_record() for name, series in self._observer_statistics.items()
            },
            layers={
                module_name: tuple(call.to_record() for _index, call in sorted(calls.items()))
                for module_name, calls in sorted(self._layers.items())
            },
            errors=tuple(error.to_record() for _key, error in sorted(self._errors.items())),
            dropped_series=self._dropped_series,
            dropped_tensor_path_observations=self._dropped_tensor_path_observations,
            dropped_module_call_observations=self._dropped_module_call_observations,
            dropped_histogram_observations=self._dropped_histogram_observations,
            dropped_error_summaries=self._dropped_error_summaries,
        )

    def _indicator_configuration(self) -> IndicatorConfigurationRecord:
        """Serialize every setting that determines temporal indicator semantics."""
        return IndicatorConfigurationRecord(
            fast_ema_alpha=self._config.fast_ema_alpha,
            slow_ema_alpha=self._config.slow_ema_alpha,
            change_volatility_alpha=self._config.change_volatility_alpha,
            momentum_horizons=self._config.momentum_horizons,
            recent_window=self._config.recent_window,
            cusum_allowance=self._config.cusum_allowance,
            warmup_observations=self._config.warmup_observations,
            max_series=self._config.max_series,
            max_tensor_paths=self._config.max_tensor_paths,
            max_module_calls=self._config.max_module_calls,
            max_histograms=self._config.max_histograms,
            max_error_summaries=self._config.max_error_summaries,
            temporal_metrics=self._config.temporal_metrics,
        )

    def _observe_modules(self, sample: SampleRecord, *, forward: bool) -> None:
        """Route one lifecycle phase into stable module, call, and tensor paths."""
        for module_name, calls in sample.modules.items():
            module_calls = self._layers.setdefault(module_name, {})
            for call_record in calls:
                call = module_calls.get(call_record.call_index)
                if call is None:
                    call = self._create_module_call(call_record.call_index)
                    if call is None:
                        continue
                    module_calls[call_record.call_index] = call
                source = call_record.outputs if forward else call_record.output_gradients
                target = call.outputs if forward else call.output_gradients
                self._observe_tensors(
                    target,
                    source,
                    sample_id=sample.sample_id,
                    timestamp=sample.timestamp,
                )

    def _observe_tensors(
        self,
        target: dict[str, _OnlineTensor],
        source: Mapping[str, TensorRecord],
        *,
        sample_id: int,
        timestamp: datetime,
    ) -> None:
        """Update tensor paths while enforcing the configured global series bound."""
        for path, tensor in source.items():
            current = target.get(path)
            if current is None:
                if self._tensor_path_count >= self._config.max_tensor_paths:
                    self._dropped_tensor_path_observations += 1
                    continue
                self._tensor_path_count += 1
                target[path] = _OnlineTensor(
                    tensor,
                    sample_id=sample_id,
                    timestamp=timestamp,
                    create_series=self._create_series,
                    create_histogram=self._create_histogram,
                    temporal_metrics=self._temporal_metrics,
                )
                continue
            current.update(
                tensor,
                sample_id=sample_id,
                timestamp=timestamp,
                create_series=self._create_series,
                create_histogram=self._create_histogram,
            )

    def _create_module_call(self, call_index: int) -> _OnlineCall | None:
        """Allocate one call-position bucket within the global structural limit."""
        if self._module_call_count >= self._config.max_module_calls:
            self._dropped_module_call_observations += 1
            return None
        self._module_call_count += 1
        return _OnlineCall(call_index=call_index)

    def _create_series(self, metric_name: str) -> _OnlineSeries | None:
        """Allocate one bounded series or count a path rejected by the memory cap."""
        if metric_name not in self._config.temporal_metrics:
            return None
        if self._series_count >= self._config.max_series:
            self._dropped_series += 1
            return None
        self._series_count += 1
        return _OnlineSeries(self._config)

    def _create_histogram(self, record: HistogramRecord) -> _OnlineHistogram | None:
        """Allocate one histogram identity within the global structural limit."""
        if self._histogram_count >= self._config.max_histograms:
            self._dropped_histogram_observations += 1
            return None
        self._histogram_count += 1
        return _OnlineHistogram(record)

    def _observer_series(self, name: str) -> _OnlineSeries:
        """Return one unconditionally retained observer-overhead series."""
        series = self._observer_statistics.get(name)
        if series is None:
            series = _OnlineSeries(self._config)
            self._observer_statistics[name] = series
        return series

    def _observe_error(self, error: ErrorRecord) -> None:
        """Aggregate repeated failures and bound the number of distinct error identities."""
        module_key = error.module if isinstance(error.module, str) else error.module.reason
        key = (module_key, error.probe, error.exception_type, error.message)
        current = self._errors.get(key)
        if current is not None:
            current.observe(error.timestamp)
            return
        if len(self._errors) >= self._config.max_error_summaries:
            self._dropped_error_summaries += 1
            return
        self._errors[key] = _OnlineError(
            count=1,
            first_timestamp=error.timestamp,
            latest_timestamp=error.timestamp,
            record=error,
        )

    def _require_run(self) -> RunRecord:
        """Return initialized run context or expose lifecycle misuse."""
        if self._run is None:
            raise RuntimeError("live aggregator must be initialized before use")
        return self._run

    def _require_updated_at(self) -> datetime:
        """Return the initialized last-update time."""
        if self._updated_at is None:
            raise RuntimeError("live aggregator update time is not initialized")
        return self._updated_at


def _standardized_difference(
    value: float,
    mean: float,
    standard_deviation: float,
) -> float | Absent:
    """Normalize a new value against prior history when that history has scale."""
    if standard_deviation <= _EPSILON:
        return Absent("prior observations have zero standard deviation")
    return (value - mean) / standard_deviation


def _safe_ratio(numerator: float, denominator: float) -> float | Absent:
    """Return a finite ratio or a reason the denominator has no useful scale."""
    if abs(denominator) <= _EPSILON:
        return Absent("denominator is zero")
    value = numerator / denominator
    if not math.isfinite(value):
        return Absent("ratio is not finite")
    return value


def _available_indicators(
    candidates: Mapping[str, IndicatorValue | Absent],
) -> Mapping[str, IndicatorValue]:
    """Omit unavailable derived values while warm-up state remains explicit."""
    return {name: value for name, value in candidates.items() if not isinstance(value, Absent)}


def _lag_one_autocorrelation(values: tuple[float, ...]) -> float | Absent:
    """Measure persistence between adjacent values in the bounded recent window."""
    if len(values) < 3:
        return Absent("at least three observations are required")
    left = values[:-1]
    right = values[1:]
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    covariance = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right, strict=True)
    )
    left_variance = sum((value - left_mean) ** 2 for value in left)
    right_variance = sum((value - right_mean) ** 2 for value in right)
    denominator = math.sqrt(left_variance * right_variance)
    return _safe_ratio(covariance, denominator)


def _oscillation_fraction(deltas: tuple[float, ...]) -> float | Absent:
    """Return the fraction of non-zero adjacent changes that reverse direction."""
    signs = tuple(1 if delta > 0 else -1 for delta in deltas if delta != 0)
    if len(signs) < 2:
        return Absent("at least two non-zero changes are required")
    reversals = sum(left != right for left, right in pairwise(signs))
    return reversals / (len(signs) - 1)


def _merge_histograms(left: HistogramRecord, right: HistogramRecord) -> HistogramRecord:
    """Merge two histograms whose fixed bin edges are exactly identical."""
    return HistogramRecord(
        bin_edges=left.bin_edges,
        bin_counts=tuple(
            left_count + right_count
            for left_count, right_count in zip(left.bin_counts, right.bin_counts, strict=True)
        ),
        finite_count=left.finite_count + right.finite_count,
        nonfinite_count=left.nonfinite_count + right.nonfinite_count,
        underflow_count=left.underflow_count + right.underflow_count,
        overflow_count=left.overflow_count + right.overflow_count,
        minimum=min(left.minimum, right.minimum),
        maximum=max(left.maximum, right.maximum),
        sum=left.sum + right.sum,
        sum_squares=left.sum_squares + right.sum_squares,
    )
