"""Tests for reducer composition, precision, and scalar contracts."""

from __future__ import annotations

import pytest
import torch

from torchinstruments import combine, finite_fraction, max_abs, mean, rms, std
from torchinstruments.reducers import reduce_tensor


def test_combined_statistics_are_materialized_together() -> None:
    """Compute the default diagnostic values through one composed reducer."""
    reducer = combine(mean(), std(), rms(), max_abs(), finite_fraction())

    result = reduce_tensor(torch.tensor([1.0, 3.0]), [reducer])

    assert result.stats == {
        "mean": pytest.approx(2.0),
        "std": pytest.approx(1.0),
        "rms": pytest.approx(5**0.5),
        "max_abs": pytest.approx(3.0),
        "finite_fraction": pytest.approx(1.0),
    }


def test_float64_reductions_are_not_downcast_during_transfer() -> None:
    """Preserve finite float64 magnitudes that exceed the float32 range."""
    result = reduce_tensor(torch.tensor([1e40], dtype=torch.float64), [mean()])

    assert result.stats["mean"] == pytest.approx(1e40)


def test_combining_duplicate_builtin_metrics_is_rejected() -> None:
    """Reject ambiguous mappings with duplicate built-in metric names."""
    with pytest.raises(ValueError, match="duplicate metrics"):
        combine(rms(), rms())


def test_reducer_output_must_be_scalar() -> None:
    """Reject custom reducers that return full tensors instead of compact scalars."""

    def invalid_reducer(tensor: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return a deliberately invalid non-scalar metric."""
        return {"invalid": tensor}

    with pytest.raises(ValueError, match="must be scalar"):
        reduce_tensor(torch.ones(2), [invalid_reducer])
