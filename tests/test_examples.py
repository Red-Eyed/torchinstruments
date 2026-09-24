"""Verify example outputs and controlled failure signatures."""

from pathlib import Path
from typing import assert_never

import polars as pl
import pytest

from examples.basic_training import run_demo
from examples.find_problems import EXPERIMENTS, Problem, evidence, run_case


def test_basic_example_writes_all_artifacts(tmp_path: Path) -> None:
    """Keep the ordinary training example representative of the required workflow."""
    output = tmp_path / "demo"
    run_demo(output)
    assert {p.name for p in output.iterdir()} == {
        "history.parquet",
        "result.json",
        "index.md",
        "tensorboard",
    }
    history = pl.read_parquet(output / "history.parquet")
    assert history["sample_id"].n_unique() == 1
    assert set(history["layer"]) == {"", "0", "1", "2"}
    assert set(history["signal"]) == {"output", "output_gradient"}


@pytest.mark.parametrize("problem", list(Problem))
def test_fault_and_fix_have_distinguishable_measurements(tmp_path: Path, problem: Problem) -> None:
    """Verify measured fault signatures, not just successful example execution."""
    experiment = EXPERIMENTS[problem]
    for variant in ("baseline", "broken", "fixed"):
        run_case(problem, variant, tmp_path / variant, steps=12)
    broken = evidence(tmp_path / "broken", experiment).filter(pl.col("sample_id") >= 4)["value"]
    fixed = evidence(tmp_path / "fixed", experiment).filter(pl.col("sample_id") >= 4)["value"]
    baseline = evidence(tmp_path / "baseline", experiment)
    assert baseline.equals(evidence(tmp_path / "fixed", experiment))
    match problem:
        case Problem.INACTIVE_RELU:
            assert broken.min() == 1
            assert fixed.max() == 0
        case Problem.SATURATION:
            assert broken.abs().max() == 0
            assert _peak(fixed) > 0.001
        case Problem.SCALE_GROWTH:
            assert _peak(broken) > _peak(fixed) * 3
            assert broken[-1] > broken[0]
        case Problem.NONFINITE:
            assert broken.min() == 1
            assert fixed.max() == 0
        case Problem.DETACHED:
            assert broken.null_count() == 8
            assert fixed.null_count() == 0
        case _:
            assert_never(problem)


def _peak(values: pl.Series) -> float:
    """Validate a numeric scalar at the Polars result boundary."""
    value = values.abs().max()
    assert isinstance(value, (float, int))
    return float(value)
