"""Tests for bounded human and LLM report generation."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from tests.json_records import JsonCategoryFindings, JsonFinding, read_report
from torchinstruments import AlwaysSampler, ReportConfig, inject_observer, remove_observer


def test_report_config_rejects_budget_too_small_for_metadata() -> None:
    """Reject a byte limit that cannot hold the self-describing report envelope."""
    with pytest.raises(ValueError, match="at least 8192"):
        ReportConfig(max_bytes=8_191)


def test_default_output_contains_only_bounded_human_and_llm_reports(
    telemetry_dir: Path,
) -> None:
    """Avoid persisting exhaustive per-layer telemetry in the default workflow."""
    model = nn.Identity()
    inject_observer(model, sampler=AlwaysSampler(), output_dir=telemetry_dir)

    model(torch.tensor(1.0))

    assert sorted(path.name for path in telemetry_dir.iterdir()) == ["index.md", "report.json"]
    report = read_report(telemetry_dir / "report.json")
    assert report["coverage"]["samples_observed"] == 1
    remove_observer(model)


def test_report_ranks_activation_drift_with_exact_evidence(telemetry_dir: Path) -> None:
    """Convert a persistent internal change into a compact auditable finding."""
    model = nn.Identity()
    inject_observer(model, sampler=AlwaysSampler(), output_dir=telemetry_dir)

    for value in range(1, 26):
        model(torch.tensor(float(value)))

    report = read_report(telemetry_dir / "report.json")
    finding = _first_finding(report["findings"], "activation_scale_drift")
    assert finding["module"] == ""
    assert finding["signal"] == "outputs"
    assert finding["metric"] == "rms"
    assert finding["first"]["value"] == 1.0
    assert finding["latest"]["value"] == 25.0
    assert finding["warmup_complete"] is True
    remove_observer(model)


def test_report_enforces_exact_utf8_byte_budget(telemetry_dir: Path) -> None:
    """Bound LLM input even when many modules produce positive-scoring candidates."""
    model = nn.Sequential(*(nn.Identity() for _index in range(80)))
    config = ReportConfig(max_bytes=16_000, top_k_per_category=80)
    inject_observer(
        model,
        sampler=AlwaysSampler(),
        report_config=config,
        output_dir=telemetry_dir,
    )

    for value in range(1, 26):
        model(torch.tensor(float(value)))

    report_path = telemetry_dir / "report.json"
    report = read_report(report_path)
    assert report_path.stat().st_size <= config.max_bytes
    assert report["coverage"]["report_truncated_by_byte_budget"] is True
    assert report["coverage"]["findings_omitted"] > 0
    remove_observer(model)


def _first_finding(
    categories: list[JsonCategoryFindings],
    category_name: str,
) -> JsonFinding:
    """Return the first typed finding from one serialized category."""
    for category in categories:
        if category["category"] != category_name:
            continue
        findings = category["findings"]
        if not findings:
            raise AssertionError(f"category {category_name!r} has no findings")
        return findings[0]
    raise AssertionError(f"missing category {category_name!r}")
