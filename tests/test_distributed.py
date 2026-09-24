"""Regression coverage for independent distributed artifact ownership."""

from pathlib import Path

import polars as pl
import pytest
import torch
from torch import nn

from torchinstruments import AlwaysSampler, has_observer, inject_observer, remove_observer
from torchinstruments.distributed import detect_rank


def test_rank0_skips_nonzero_rank(telemetry_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid hooks, histogram work, and files on excluded ranks."""
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")
    model = nn.Identity()
    inject_observer(model, sampler=AlwaysSampler(), output_dir=telemetry_dir)
    model(torch.ones(1))
    assert not has_observer(model)
    assert not telemetry_dir.exists()


def test_all_ranks_keep_private_artifacts(
    telemetry_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure two ranks never share writers or output filenames."""
    monkeypatch.setenv("WORLD_SIZE", "2")
    for rank in range(2):
        monkeypatch.setenv("RANK", str(rank))
        model = nn.Identity()
        inject_observer(model, sampler=AlwaysSampler(), output_dir=telemetry_dir, rank_policy="all")
        model(torch.tensor(float(rank)))
        remove_observer(model)
        directory = telemetry_dir / f"rank-{rank:03d}"
        assert {p.name for p in directory.iterdir()} == {
            "history.parquet",
            "result.json",
            "index.md",
            "tensorboard",
        }
        frame = pl.read_parquet(directory / "history.parquet")
        assert frame.filter(pl.col("metric") == "mean")["value"].item() == rank


@pytest.mark.parametrize(("rank", "world_size"), [("x", "2"), ("2", "2"), ("0", "0")])
def test_invalid_rank_identity(monkeypatch: pytest.MonkeyPatch, rank: str, world_size: str) -> None:
    """Reject identities that could produce ambiguous rank ownership."""
    monkeypatch.setenv("RANK", rank)
    monkeypatch.setenv("WORLD_SIZE", world_size)
    with pytest.raises(ValueError):
        detect_rank()
