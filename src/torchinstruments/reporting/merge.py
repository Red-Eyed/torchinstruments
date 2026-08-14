"""Streaming merge of independent rank reports into one bounded global ranking."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import TypedDict, cast

from torchinstruments.distributed import RankInfo
from torchinstruments.records import MetricPointRecord
from torchinstruments.reporting.records import (
    REPORT_SCHEMA_VERSION,
    CategoryFindingsRecord,
    EvidenceValueRecord,
    FindingCategory,
    FindingRecord,
    MergedReportCoverageRecord,
    MergedReportRecord,
    ReportConfig,
    ReportConfigurationRecord,
)
from torchinstruments.serialization import json_size_bytes
from torchinstruments.sinks.index_markdown import render_merged_index
from torchinstruments.sinks.json import write_json_atomic, write_text_atomic

_DEFAULT_REPORT_CONFIG = ReportConfig()


class _JsonMetricPoint(TypedDict):
    """Describe one serialized metric point at the JSON parsing boundary."""

    value: float
    sample_id: int
    timestamp: str


class _JsonEvidenceValue(TypedDict):
    """Describe one serialized named evidence scalar."""

    name: str
    value: float | int


class _JsonFinding(TypedDict):
    """Describe fields required to reconstruct a typed finding."""

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
    first: _JsonMetricPoint
    latest: _JsonMetricPoint
    minimum: _JsonMetricPoint
    maximum: _JsonMetricPoint
    evidence: list[_JsonEvidenceValue]
    interpretation: str


class _JsonCategoryFindings(TypedDict):
    """Describe one serialized category group."""

    category: str
    findings: list[_JsonFinding]


class _JsonRank(TypedDict):
    """Describe one serialized rank identity."""

    rank: int
    world_size: int


class _JsonCoverage(TypedDict):
    """Describe source-report truncation needed for merge confidence."""

    report_truncated_by_byte_budget: bool


class _JsonRankReport(TypedDict):
    """Describe the source-report fields consumed by the streaming merger."""

    updated_at: str
    rank: _JsonRank
    coverage: _JsonCoverage
    findings: list[_JsonCategoryFindings]


def merge_rank_reports(
    input_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    config: ReportConfig = _DEFAULT_REPORT_CONFIG,
) -> tuple[Path, Path]:
    """Stream rank-local JSON reports into bounded global JSON and Markdown artifacts.

    The function reads one already-bounded rank report at a time. It never synchronizes workers
    or assumes every expected rank has finished; completeness is recorded in the merged output.
    Existing merged artifacts are atomically replaced so analysis can be refreshed safely.
    """
    source_root = Path(input_dir)
    destination = Path(output_dir) if output_dir is not None else source_root
    reports = _rank_report_paths(source_root)
    if not reports:
        raise FileNotFoundError(f"no rank reports found under {source_root}")

    buckets = {category: [] for category in FindingCategory}
    ranks: set[int] = set()
    expected_ranks: int | None = None
    latest_update: datetime | None = None
    source_reports_truncated = 0
    source_findings = 0
    for path in reports:
        source = _read_rank_report(path)
        rank = source["rank"]
        rank_info = RankInfo(rank=rank["rank"], world_size=rank["world_size"])
        expected_ranks = _consistent_world_size(expected_ranks, rank["world_size"], path)
        if rank_info.rank in ranks:
            raise ValueError(f"duplicate rank report for rank {rank_info.rank}: {path}")
        ranks.add(rank_info.rank)
        latest_update = _latest_timestamp(latest_update, _parse_timestamp(source["updated_at"]))
        source_reports_truncated += int(source["coverage"]["report_truncated_by_byte_budget"])
        for group in source["findings"]:
            category = FindingCategory(group["category"])
            parsed = tuple(_parse_finding(value) for value in group["findings"])
            source_findings += len(parsed)
            buckets[category].extend(parsed)
            buckets[category] = sorted(buckets[category], key=_finding_sort_key)[
                : config.top_k_per_category
            ]

    if expected_ranks is None or latest_update is None:
        raise RuntimeError("rank report merge did not observe metadata")
    merged = _build_merged_report(
        buckets,
        updated_at=latest_update,
        expected_ranks=expected_ranks,
        ranks_present=tuple(sorted(ranks)),
        source_reports_truncated=source_reports_truncated,
        source_findings=source_findings,
        config=config,
    )
    json_path = destination / "global-report.json"
    markdown_path = destination / "global-index.md"
    write_json_atomic(json_path, merged)
    write_text_atomic(markdown_path, render_merged_index(merged))
    return json_path, markdown_path


def _rank_report_paths(root: Path) -> tuple[Path, ...]:
    """Return stable rank-local report paths while excluding merged output."""
    return tuple(sorted(root.glob("rank-*/report.json")))


def _read_rank_report(path: Path) -> _JsonRankReport:
    """Parse one untrusted JSON object into the precise merger boundary shape."""
    with path.open(encoding="utf-8") as file:
        value: object = json.load(file)
    if not isinstance(value, dict):
        raise TypeError(f"rank report must be a JSON object: {path}")
    required = {"updated_at", "rank", "coverage", "findings"}
    if not required.issubset(value):
        raise ValueError(f"rank report is missing required fields: {path}")
    return cast(_JsonRankReport, value)


def _parse_finding(value: _JsonFinding) -> FindingRecord:
    """Validate enum and timestamp fields while constructing one typed finding."""
    return FindingRecord(
        category=FindingCategory(value["category"]),
        rank=value["rank"],
        ranking_score=value["ranking_score"],
        ranking_basis=value["ranking_basis"],
        source_rank=value["source_rank"],
        module=value["module"],
        module_type=value["module_type"],
        call_index=value["call_index"],
        signal=value["signal"],
        tensor_path=value["tensor_path"],
        metric=value["metric"],
        observations=value["observations"],
        warmup_complete=value["warmup_complete"],
        first=_parse_point(value["first"]),
        latest=_parse_point(value["latest"]),
        minimum=_parse_point(value["minimum"]),
        maximum=_parse_point(value["maximum"]),
        evidence=tuple(
            EvidenceValueRecord(item["name"], item["value"]) for item in value["evidence"]
        ),
        interpretation=value["interpretation"],
    )


def _parse_point(value: _JsonMetricPoint) -> MetricPointRecord:
    """Parse one ISO timestamp exactly once at the JSON boundary."""
    return MetricPointRecord(
        value=value["value"],
        sample_id=value["sample_id"],
        timestamp=_parse_timestamp(value["timestamp"]),
    )


def _parse_timestamp(value: str) -> datetime:
    """Convert one serialized UTC timestamp into a strict datetime."""
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _consistent_world_size(current: int | None, observed: int, path: Path) -> int:
    """Require every rank report to describe the same process group size."""
    if observed <= 0:
        raise ValueError(f"rank report has invalid world_size: {path}")
    if current is not None and current != observed:
        raise ValueError(f"rank reports disagree about world_size: {path}")
    return observed


def _latest_timestamp(current: datetime | None, observed: datetime) -> datetime:
    """Return the later timezone-aware report update timestamp."""
    return observed if current is None else max(current, observed)


def _finding_sort_key(finding: FindingRecord) -> tuple[object, ...]:
    """Rank descending score with deterministic rank and tensor identity ties."""
    return (
        -finding.ranking_score,
        finding.source_rank,
        finding.module,
        finding.call_index,
        finding.signal,
        finding.tensor_path,
        finding.metric,
    )


def _build_merged_report(
    buckets: dict[FindingCategory, list[FindingRecord]],
    *,
    updated_at: datetime,
    expected_ranks: int,
    ranks_present: tuple[int, ...],
    source_reports_truncated: int,
    source_findings: int,
    config: ReportConfig,
) -> MergedReportRecord:
    """Apply global ranking and the exact JSON byte budget to streamed candidates."""
    configuration = ReportConfigurationRecord(
        max_bytes=config.max_bytes,
        top_k_per_category=config.top_k_per_category,
        markdown_findings_per_category=config.markdown_findings_per_category,
        max_errors=config.max_errors,
        max_error_message_chars=config.max_error_message_chars,
    )
    selected = {category: [] for category in FindingCategory}
    base = MergedReportRecord(
        report_schema_version=REPORT_SCHEMA_VERSION,
        updated_at=updated_at,
        report_configuration=configuration,
        coverage=MergedReportCoverageRecord(
            expected_ranks=expected_ranks,
            ranks_present=ranks_present,
            rank_coverage_complete=ranks_present == tuple(range(expected_ranks)),
            source_reports_truncated=source_reports_truncated,
            findings_returned=0,
            findings_omitted_after_global_ranking=source_findings,
            report_truncated_by_byte_budget=False,
        ),
        findings=tuple(CategoryFindingsRecord(category, ()) for category in FindingCategory),
    )
    truncated = False
    maximum = max((len(values) for values in buckets.values()), default=0)
    for index in range(maximum):
        for category in FindingCategory:
            if index >= len(buckets[category]):
                continue
            finding = replace(buckets[category][index], rank=len(selected[category]) + 1)
            groups = tuple(
                CategoryFindingsRecord(
                    current,
                    (
                        *selected[current],
                        *((finding,) if current is category else ()),
                    ),
                )
                for current in FindingCategory
            )
            returned = sum(len(group.findings) for group in groups)
            trial = replace(
                base,
                findings=groups,
                coverage=replace(
                    base.coverage,
                    findings_returned=returned,
                    findings_omitted_after_global_ranking=source_findings - returned,
                ),
            )
            if json_size_bytes(trial) <= config.max_bytes:
                selected[category].append(finding)
                base = trial
            else:
                truncated = True
    if truncated:
        base = replace(
            base,
            coverage=replace(base.coverage, report_truncated_by_byte_budget=True),
        )
    if json_size_bytes(base) > config.max_bytes:
        raise ValueError("merged report metadata exceeds max_bytes")
    return base
