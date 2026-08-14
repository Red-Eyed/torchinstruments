"""Independent diagnostic rules that rank measured telemetry signatures."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol

from torchinstruments.records import SeriesSummaryRecord
from torchinstruments.reporting.records import EvidenceValueRecord, FindingCategory


@dataclass(frozen=True)
class SeriesContext:
    """Locate one temporal series and the tensor distribution that produced it."""

    module: str
    module_type: str
    call_index: int
    signal: str
    tensor_path: str
    metric: str
    tensor_observations: int
    series: SeriesSummaryRecord


@dataclass(frozen=True)
class FindingCandidate:
    """Carry one rule result before category ranking and byte-budget selection."""

    category: FindingCategory
    ranking_score: float
    ranking_basis: str
    context: SeriesContext
    evidence: tuple[EvidenceValueRecord, ...]
    interpretation: str


class FindingRule(Protocol):
    """Evaluate one temporal series for one independently ranked diagnostic category."""

    @property
    def category(self) -> FindingCategory:
        """Return the independently ranked category produced by this rule."""
        ...

    def __call__(self, context: SeriesContext) -> FindingCandidate | None:
        """Return measured evidence when the series is relevant to this category."""
        ...


@dataclass(frozen=True)
class _ConfiguredRule:
    """Bind one category to a pure series-scoring function."""

    category: FindingCategory
    evaluate: Callable[
        [SeriesContext],
        tuple[float, str, tuple[EvidenceValueRecord, ...], str] | None,
    ]

    def __call__(self, context: SeriesContext) -> FindingCandidate | None:
        """Convert a positive finite rule score into a normalized candidate."""
        result = self.evaluate(context)
        if result is None:
            return None
        score, basis, evidence, interpretation = result
        if not math.isfinite(score) or score <= 0:
            return None
        return FindingCandidate(
            category=self.category,
            ranking_score=score,
            ranking_basis=basis,
            context=context,
            evidence=evidence,
            interpretation=interpretation,
        )


def default_finding_rules() -> tuple[FindingRule, ...]:
    """Return descriptive category rules without combining them into one health score."""
    return (
        _ConfiguredRule(FindingCategory.ACTIVATION_SCALE_DRIFT, _activation_scale_drift),
        _ConfiguredRule(FindingCategory.GRADIENT_SCALE_CHANGE, _gradient_scale_change),
        _ConfiguredRule(FindingCategory.HEAVY_TAIL_GROWTH, _heavy_tail_growth),
        _ConfiguredRule(FindingCategory.NONFINITE_VALUES, _nonfinite_values),
        _ConfiguredRule(FindingCategory.ZERO_FRACTION_GROWTH, _zero_fraction_growth),
        _ConfiguredRule(FindingCategory.VOLATILITY, _volatility),
        _ConfiguredRule(FindingCategory.OSCILLATION, _oscillation),
        _ConfiguredRule(FindingCategory.REGIME_CHANGE, _regime_change),
    )


def _activation_scale_drift(
    context: SeriesContext,
) -> tuple[float, str, tuple[EvidenceValueRecord, ...], str] | None:
    """Rank relative output-RMS movement across sampled forwards."""
    if context.signal != "outputs" or context.metric != "rms":
        return None
    score, evidence = _scale_change_evidence(context.series)
    return (
        score,
        "largest absolute relative output-RMS change",
        evidence,
        "Activation scale changed relative to its earlier sampled behavior.",
    )


def _gradient_scale_change(
    context: SeriesContext,
) -> tuple[float, str, tuple[EvidenceValueRecord, ...], str] | None:
    """Rank relative output-gradient RMS movement without calling it an optimizer update."""
    if context.signal != "output_gradients" or context.metric != "rms":
        return None
    score, evidence = _scale_change_evidence(context.series)
    return (
        score,
        "largest absolute relative output-gradient RMS change",
        evidence,
        "Output-gradient scale changed relative to its earlier sampled behavior.",
    )


def _heavy_tail_growth(
    context: SeriesContext,
) -> tuple[float, str, tuple[EvidenceValueRecord, ...], str] | None:
    """Rank growth in dimensionless tail and outlier metrics."""
    if context.signal != "outputs" or context.metric not in {
        "max_to_rms",
        "p99_abs_to_rms",
        "p999_abs_to_rms",
        "excess_kurtosis",
    }:
        return None
    relative_change = _relative_change(context.series.first.value, context.series.latest.value)
    if relative_change is None:
        return None
    score = max(relative_change, 0.0)
    evidence = _selected_indicators(
        context.series,
        "fast_slow_relative_gap",
        "relative_momentum_5_samples",
        "relative_momentum_20_samples",
        "latest_z_score",
    )
    evidence = (*evidence, EvidenceValueRecord("relative_change_from_first", relative_change))
    return (
        score,
        "largest positive relative change in a tail-to-scale metric",
        evidence,
        "The sampled activation distribution became more tail-dominated or asymmetric.",
    )


def _nonfinite_values(
    context: SeriesContext,
) -> tuple[float, str, tuple[EvidenceValueRecord, ...], str] | None:
    """Rank the current fraction of non-finite tensor values."""
    if context.metric != "finite_fraction":
        return None
    nonfinite_fraction = max(0.0, 1.0 - context.series.latest.value)
    return (
        nonfinite_fraction,
        "largest latest non-finite fraction",
        (EvidenceValueRecord("latest_nonfinite_fraction", nonfinite_fraction),),
        "The sampled tensor contained NaN or infinite values.",
    )


def _zero_fraction_growth(
    context: SeriesContext,
) -> tuple[float, str, tuple[EvidenceValueRecord, ...], str] | None:
    """Rank positive growth in exact-zero prevalence."""
    if context.metric != "zero_fraction":
        return None
    growth = context.series.latest.value - context.series.first.value
    return (
        max(growth, 0.0),
        "largest increase in zero fraction from the first observation",
        (EvidenceValueRecord("absolute_growth_from_first", growth),),
        "Exact zeros became more prevalent in this tensor path.",
    )


def _volatility(
    context: SeriesContext,
) -> tuple[float, str, tuple[EvidenceValueRecord, ...], str] | None:
    """Rank exponentially weighted RMS-change volatility relative to current scale."""
    if context.metric != "rms":
        return None
    indicators = context.series.indicators
    volatility = _float_indicator(indicators, "exponentially_weighted_change_volatility")
    slow_ema = _float_indicator(indicators, "slow_ema")
    if volatility is None or slow_ema is None or abs(slow_ema) <= 1e-12:
        return None
    relative_volatility = volatility / abs(slow_ema)
    return (
        relative_volatility,
        "largest exponentially weighted RMS-change volatility divided by slow EMA",
        (
            EvidenceValueRecord("relative_change_volatility", relative_volatility),
            EvidenceValueRecord("exponentially_weighted_change_volatility", volatility),
            EvidenceValueRecord("slow_ema", slow_ema),
        ),
        "RMS changed rapidly relative to its recent level.",
    )


def _oscillation(
    context: SeriesContext,
) -> tuple[float, str, tuple[EvidenceValueRecord, ...], str] | None:
    """Rank alternating RMS behavior using reversals and negative autocorrelation."""
    if context.metric != "rms":
        return None
    indicators = context.series.indicators
    fraction = _float_indicator(indicators, "oscillation_fraction")
    autocorrelation = _float_indicator(indicators, "lag1_autocorrelation")
    if fraction is None or autocorrelation is None:
        return None
    score = fraction * max(-autocorrelation, 0.0)
    return (
        score,
        "oscillation fraction multiplied by negative lag-one autocorrelation",
        (
            EvidenceValueRecord("oscillation_fraction", fraction),
            EvidenceValueRecord("lag1_autocorrelation", autocorrelation),
        ),
        "Successive sampled RMS changes repeatedly reversed direction.",
    )


def _regime_change(
    context: SeriesContext,
) -> tuple[float, str, tuple[EvidenceValueRecord, ...], str] | None:
    """Rank normalized CUSUM evidence for a persistent level change."""
    if context.metric != "rms":
        return None
    score = _float_indicator(context.series.indicators, "cusum_change_score")
    if score is None:
        return None
    normalized = score / math.sqrt(max(context.series.count, 1))
    return (
        normalized,
        "CUSUM change score divided by the square root of observations",
        (
            EvidenceValueRecord("cusum_change_score", score),
            EvidenceValueRecord("normalized_cusum_score", normalized),
        ),
        "Recent RMS behavior accumulated evidence of a sustained level change.",
    )


def _scale_change_evidence(
    series: SeriesSummaryRecord,
) -> tuple[float, tuple[EvidenceValueRecord, ...]]:
    """Return the strongest dimensionless scale movement and its component evidence."""
    evidence = _selected_indicators(
        series,
        "fast_slow_relative_gap",
        "relative_momentum_5_samples",
        "relative_momentum_20_samples",
        "latest_z_score",
        "linear_slope_per_sample",
        "slope_r_squared",
        "maximum_drawdown",
        "maximum_runup",
    )
    relative_change = _relative_change(series.first.value, series.latest.value)
    if relative_change is not None:
        evidence = (*evidence, EvidenceValueRecord("relative_change_from_first", relative_change))
    dimensionless = [
        abs(item.value)
        for item in evidence
        if item.name
        in {
            "fast_slow_relative_gap",
            "relative_momentum_5_samples",
            "relative_momentum_20_samples",
            "relative_change_from_first",
        }
    ]
    return (max(dimensionless, default=0.0), evidence)


def _selected_indicators(
    series: SeriesSummaryRecord,
    *names: str,
) -> tuple[EvidenceValueRecord, ...]:
    """Copy only evidence fields relevant to one finding category."""
    return tuple(
        EvidenceValueRecord(name, series.indicators[name])
        for name in names
        if name in series.indicators
    )


def _relative_change(first: float, latest: float) -> float | None:
    """Return signed relative movement when the first observation supplies scale."""
    if abs(first) <= 1e-12:
        return None
    value = (latest - first) / abs(first)
    return value if math.isfinite(value) else None


def _float_indicator(indicators: Mapping[str, float | int], name: str) -> float | None:
    """Read one finite indicator as a float without accepting booleans."""
    value = indicators.get(name)
    if value is None or isinstance(value, bool):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None
