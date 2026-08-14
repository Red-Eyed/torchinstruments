"""Build deterministic byte-bounded reports from complete in-memory live state."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

from torchinstruments.distributed import RankInfo
from torchinstruments.records import ErrorSummaryRecord, LiveStatsRecord
from torchinstruments.reporting.records import (
    REPORT_SCHEMA_VERSION,
    CategoryCountRecord,
    CategoryFindingsRecord,
    FindingCategory,
    FindingRecord,
    RankRecord,
    ReportConfig,
    ReportConfigurationRecord,
    ReportCoverageRecord,
    ReportRecord,
)
from torchinstruments.reporting.rules import (
    FindingCandidate,
    FindingRule,
    SeriesContext,
    default_finding_rules,
)
from torchinstruments.serialization import json_size_bytes


def build_report(
    stats: LiveStatsRecord,
    *,
    rank: RankInfo,
    config: ReportConfig,
    rules: Sequence[FindingRule] = (),
) -> ReportRecord:
    """Rank live state locally and retain findings within the exact UTF-8 byte budget."""
    active_rules = tuple(rules) or default_finding_rules()
    candidates, tensor_paths, temporal_series, histograms = _collect_candidates(
        stats,
        active_rules,
    )
    ranked = _rank_candidates(candidates, config.top_k_per_category)
    errors = _bounded_errors(stats.errors, config)
    report = _empty_report(
        stats,
        rank=rank,
        config=config,
        candidate_counts={category: len(values) for category, values in candidates.items()},
        tensor_paths=tensor_paths,
        temporal_series=temporal_series,
        histograms=histograms,
        errors=errors,
    )
    if json_size_bytes(report) > config.max_bytes:
        report = replace(
            report,
            errors=(),
            coverage=replace(
                report.coverage,
                errors_returned=0,
                errors_omitted=len(stats.errors),
                report_truncated_by_byte_budget=True,
            ),
        )
    report = _add_findings_within_budget(report, ranked, config.max_bytes)
    if json_size_bytes(report) > config.max_bytes:
        raise ValueError("report metadata exceeds max_bytes; increase ReportConfig.max_bytes")
    return report


def _collect_candidates(
    stats: LiveStatsRecord,
    rules: Sequence[FindingRule],
) -> tuple[dict[FindingCategory, list[FindingCandidate]], int, int, int]:
    """Traverse live state once and evaluate every independent diagnostic rule."""
    candidates = {category: [] for category in FindingCategory}
    tensor_paths = 0
    temporal_series = 0
    histograms = 0
    for module_name, calls in stats.layers.items():
        module_type = stats.module_catalog[module_name].type
        for call in calls:
            for signal, tensors in (
                ("outputs", call.outputs),
                ("output_gradients", call.output_gradients),
            ):
                for tensor_path, tensor in tensors.items():
                    tensor_paths += 1
                    histograms += len(tensor.histograms)
                    for metric, series in tensor.statistics.items():
                        temporal_series += 1
                        context = SeriesContext(
                            module=module_name,
                            module_type=module_type,
                            call_index=call.call_index,
                            signal=signal,
                            tensor_path=tensor_path,
                            metric=metric,
                            tensor_observations=tensor.observations,
                            series=series,
                        )
                        for rule in rules:
                            candidate = rule(context)
                            if candidate is not None:
                                candidates[candidate.category].append(candidate)
    return candidates, tensor_paths, temporal_series, histograms


def _rank_candidates(
    candidates: Mapping[FindingCategory, list[FindingCandidate]],
    top_k: int,
) -> dict[FindingCategory, tuple[FindingCandidate, ...]]:
    """Select deterministic top candidates independently within each category."""
    return {
        category: tuple(sorted(values, key=_candidate_sort_key)[:top_k])
        for category, values in candidates.items()
    }


def _candidate_sort_key(candidate: FindingCandidate) -> tuple[object, ...]:
    """Sort by descending score and stable telemetry identity for reproducible reports."""
    context = candidate.context
    return (
        -candidate.ranking_score,
        context.module,
        context.call_index,
        context.signal,
        context.tensor_path,
        context.metric,
    )


def _empty_report(
    stats: LiveStatsRecord,
    *,
    rank: RankInfo,
    config: ReportConfig,
    candidate_counts: Mapping[FindingCategory, int],
    tensor_paths: int,
    temporal_series: int,
    histograms: int,
    errors: tuple[ErrorSummaryRecord, ...],
) -> ReportRecord:
    """Create report metadata before byte-budgeted findings are added."""
    total_candidates = sum(candidate_counts.values())
    return ReportRecord(
        report_schema_version=REPORT_SCHEMA_VERSION,
        telemetry_schema_version=stats.schema_version,
        updated_at=stats.updated_at,
        run=stats.run,
        rank=RankRecord(rank=rank.rank, world_size=rank.world_size),
        report_configuration=ReportConfigurationRecord(
            max_bytes=config.max_bytes,
            top_k_per_category=config.top_k_per_category,
            markdown_findings_per_category=config.markdown_findings_per_category,
            max_errors=config.max_errors,
            max_error_message_chars=config.max_error_message_chars,
        ),
        indicator_configuration=stats.indicator_configuration,
        coverage=ReportCoverageRecord(
            selected_modules=len(stats.module_catalog),
            tensor_paths=tensor_paths,
            temporal_series=temporal_series,
            histograms=histograms,
            samples_observed=stats.samples_observed,
            backward_samples_observed=stats.backward_samples_observed,
            candidates_by_category=tuple(
                CategoryCountRecord(category, candidate_counts[category])
                for category in FindingCategory
            ),
            findings_returned=0,
            findings_omitted=total_candidates,
            errors_returned=len(errors),
            errors_omitted=len(stats.errors) - len(errors),
            report_truncated_by_byte_budget=False,
            dropped_series=stats.dropped_series,
            dropped_tensor_path_observations=stats.dropped_tensor_path_observations,
            dropped_module_call_observations=stats.dropped_module_call_observations,
            dropped_histogram_observations=stats.dropped_histogram_observations,
            dropped_error_summaries=stats.dropped_error_summaries,
        ),
        findings=tuple(CategoryFindingsRecord(category, ()) for category in FindingCategory),
        errors=errors,
    )


def _add_findings_within_budget(
    report: ReportRecord,
    ranked: Mapping[FindingCategory, tuple[FindingCandidate, ...]],
    max_bytes: int,
) -> ReportRecord:
    """Add candidates round-robin so one category cannot consume the report budget."""
    selected: dict[str, list[FindingRecord]] = {category.value: [] for category in FindingCategory}
    candidates = _round_robin_candidates(ranked)
    truncated = False
    for candidate in candidates:
        category_name = candidate.category.value
        rank = len(selected[category_name]) + 1
        finding = _finding_record(candidate, rank=rank, source_rank=report.rank.rank)
        trial_selected = tuple(
            CategoryFindingsRecord(
                category,
                (
                    *selected[category.value],
                    *((finding,) if category.value == category_name else ()),
                ),
            )
            for category in FindingCategory
        )
        returned = sum(len(group.findings) for group in trial_selected)
        trial = replace(
            report,
            findings=trial_selected,
            coverage=replace(
                report.coverage,
                findings_returned=returned,
                findings_omitted=_total_candidates(report) - returned,
            ),
        )
        if json_size_bytes(trial) <= max_bytes:
            selected[category_name].append(finding)
            report = trial
            continue
        truncated = True
    if truncated:
        report = replace(
            report,
            coverage=replace(report.coverage, report_truncated_by_byte_budget=True),
        )
    return report


def _round_robin_candidates(
    ranked: Mapping[FindingCategory, tuple[FindingCandidate, ...]],
) -> tuple[FindingCandidate, ...]:
    """Interleave category ranks while preserving deterministic category order."""
    maximum = max((len(values) for values in ranked.values()), default=0)
    return tuple(
        ranked[category][index]
        for index in range(maximum)
        for category in FindingCategory
        if index < len(ranked[category])
    )


def _finding_record(
    candidate: FindingCandidate,
    *,
    rank: int,
    source_rank: int,
) -> FindingRecord:
    """Convert an internal rule result into its public evidence record."""
    context = candidate.context
    series = context.series
    return FindingRecord(
        category=candidate.category,
        rank=rank,
        ranking_score=candidate.ranking_score,
        ranking_basis=candidate.ranking_basis,
        source_rank=source_rank,
        module=context.module,
        module_type=context.module_type,
        call_index=context.call_index,
        signal=context.signal,
        tensor_path=context.tensor_path,
        metric=context.metric,
        observations=context.tensor_observations,
        warmup_complete=series.warmup_complete,
        first=series.first,
        latest=series.latest,
        minimum=series.minimum,
        maximum=series.maximum,
        evidence=candidate.evidence,
        interpretation=candidate.interpretation,
    )


def _bounded_errors(
    errors: tuple[ErrorSummaryRecord, ...],
    config: ReportConfig,
) -> tuple[ErrorSummaryRecord, ...]:
    """Limit error count and individual message size before report budgeting."""
    return tuple(
        replace(error, message=_truncate(error.message, config.max_error_message_chars))
        for error in errors[: config.max_errors]
    )


def _truncate(value: str, limit: int) -> str:
    """Truncate one message visibly without exceeding its configured character limit."""
    if len(value) <= limit:
        return value
    suffix = "…"
    return f"{value[: limit - len(suffix)]}{suffix}"


def _total_candidates(report: ReportRecord) -> int:
    """Return the complete pre-ranking candidate count recorded in coverage."""
    return sum(item.count for item in report.coverage.candidates_by_category)
