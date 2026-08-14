"""Tests for rank ownership and collision-free distributed report paths."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import torch
from torch import nn

from tests.json_records import read_merged_report, read_report
from torchinstruments import (
    AlwaysSampler,
    ReportConfig,
    has_observer,
    inject_observer,
    merge_rank_reports,
    remove_observer,
)
from torchinstruments.distributed import detect_rank


def test_rank0_policy_skips_hooks_and_files_on_nonzero_rank(
    telemetry_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Make the default DDP policy free of nonzero-rank collection and write overhead."""
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")
    model = nn.Identity()

    inject_observer(model, sampler=AlwaysSampler(), output_dir=telemetry_dir)
    model(torch.tensor(1.0))

    assert not has_observer(model)
    assert not telemetry_dir.exists()


def test_all_rank_policy_uses_private_human_readable_directories(
    telemetry_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prevent concurrent ranks from writing the same JSON or Markdown path."""
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")
    model = nn.Identity()

    inject_observer(
        model,
        sampler=AlwaysSampler(),
        output_dir=telemetry_dir,
        rank_policy="all",
    )
    model(torch.tensor(1.0))

    rank_dir = telemetry_dir / "rank-001"
    assert sorted(path.name for path in rank_dir.iterdir()) == ["index.md", "report.json"]
    report = read_report(rank_dir / "report.json")
    assert report["rank"] == {"rank": 1, "world_size": 2}
    remove_observer(model)


@pytest.mark.parametrize(
    ("rank", "world_size"),
    [("not-an-integer", "2"), ("2", "2"), ("0", "0")],
)
def test_invalid_distributed_identity_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    rank: str,
    world_size: str,
) -> None:
    """Reject malformed environment rank state instead of risking path collisions."""
    monkeypatch.setenv("RANK", rank)
    monkeypatch.setenv("WORLD_SIZE", world_size)

    with pytest.raises(ValueError):
        detect_rank()


def test_rank_reports_merge_streamingly_into_one_bounded_global_report(
    telemetry_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rank findings globally without shared training-time writers or unbounded JSON input."""
    monkeypatch.setenv("WORLD_SIZE", "2")
    for rank in range(2):
        monkeypatch.setenv("RANK", str(rank))
        model = nn.Identity()
        inject_observer(
            model,
            sampler=AlwaysSampler(),
            output_dir=telemetry_dir,
            rank_policy="all",
        )
        for value in range(1, 26):
            observed = float(value if rank == 0 else value * value)
            model(torch.tensor(observed))
        remove_observer(model)

    config = ReportConfig(max_bytes=32_000)
    json_path, markdown_path = merge_rank_reports(telemetry_dir, config=config)

    merged = read_merged_report(json_path)
    assert json_path.stat().st_size <= config.max_bytes
    assert markdown_path.name == "global-index.md"
    assert merged["coverage"]["ranks_present"] == [0, 1]
    assert merged["coverage"]["rank_coverage_complete"] is True
    activation = next(
        group for group in merged["findings"] if group["category"] == "activation_scale_drift"
    )
    assert activation["findings"][0]["source_rank"] == 1


def test_merge_reports_incomplete_high_rank_coverage(
    telemetry_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Discover ranks beyond three digits and state that absent workers remain unknown."""
    monkeypatch.setenv("RANK", "1000")
    monkeypatch.setenv("WORLD_SIZE", "1001")
    model = nn.Identity()
    inject_observer(model, output_dir=telemetry_dir, rank_policy="all")
    remove_observer(model)

    json_path, _markdown_path = merge_rank_reports(telemetry_dir)

    merged = read_merged_report(json_path)
    assert merged["coverage"]["ranks_present"] == [1000]
    assert merged["coverage"]["rank_coverage_complete"] is False


def test_merge_rejects_duplicate_rank_identity(
    telemetry_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject copied reports that would otherwise make rank coverage ambiguous."""
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")
    model = nn.Identity()
    inject_observer(model, output_dir=telemetry_dir, rank_policy="all")
    remove_observer(model)
    shutil.copytree(telemetry_dir / "rank-000", telemetry_dir / "rank-copy")

    with pytest.raises(ValueError, match="duplicate rank report"):
        merge_rank_reports(telemetry_dir)
