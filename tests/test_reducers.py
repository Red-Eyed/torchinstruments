"""Tests for reducer composition, precision, and scalar contracts."""

from __future__ import annotations

import pytest
import torch

from torchinstruments import combine, finite_fraction, histogram, max_abs, mean, rms, std
from torchinstruments.reducers import (
    HistogramReductionResult,
    default_reducers,
    reduce_histograms,
    reduce_tensor,
)


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


def test_default_profile_distinguishes_equal_scale_but_skewed_distributions() -> None:
    """Expose asymmetry and tails that identical mean and standard deviation hide."""
    symmetric = reduce_tensor(torch.tensor([-1.0, -1.0, 1.0, 1.0]), default_reducers())
    skewed = reduce_tensor(
        torch.tensor([-0.5773503, -0.5773503, -0.5773503, 1.7320508]),
        default_reducers(),
    )

    assert symmetric.stats["mean"] == pytest.approx(skewed.stats["mean"], abs=1e-6)
    assert symmetric.stats["std"] == pytest.approx(skewed.stats["std"], rel=1e-5)
    assert symmetric.stats["skewness"] == pytest.approx(0.0)
    assert skewed.stats["skewness"] > 1.0
    assert skewed.stats["p999_abs_to_rms"] > symmetric.stats["p999_abs_to_rms"]


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


def test_fixed_histogram_preserves_outliers_nonfinite_counts_and_moments() -> None:
    """Retain every value needed to reconstruct a fixed-range histogram."""
    values = torch.tensor([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, float("nan")])
    reducer = histogram(bins=2, value_range=(-1.0, 1.0), every_n_samples=1)

    result = reduce_histograms(values, [reducer], sample_id=0)

    record = result.histograms["distribution"]
    assert record.bin_edges == pytest.approx((-1.0, 0.0, 1.0))
    assert record.bin_counts == (2, 3)
    assert record.finite_count == 7
    assert record.nonfinite_count == 1
    assert record.underflow_count == 1
    assert record.overflow_count == 1
    assert record.minimum == pytest.approx(-2.0)
    assert record.maximum == pytest.approx(2.0)
    assert record.sum == pytest.approx(0.0)
    assert record.sum_squares == pytest.approx(10.5)


def test_histogram_cadence_is_independent_of_forward_sampling() -> None:
    """Collect the first histogram and then only each configured sample interval."""
    reducer = histogram(every_n_samples=2)

    skipped = reduce_histograms(torch.ones(4), [reducer], sample_id=1)
    collected = reduce_histograms(torch.ones(4), [reducer], sample_id=2)

    assert skipped.histograms == {}
    assert skipped.unavailable_histograms == {}
    assert set(collected.histograms) == {"distribution"}


def test_histogram_without_finite_values_carries_a_reason() -> None:
    """Represent an attempted but impossible histogram without JSON nulls or NaNs."""
    reducer = histogram(every_n_samples=1)

    result = reduce_histograms(torch.tensor([float("nan"), float("inf")]), [reducer], sample_id=0)

    assert result.histograms == {}
    assert result.unavailable_histograms == {"distribution": "tensor has no finite values"}


def test_custom_histogram_reducer_receives_a_detached_tensor() -> None:
    """Prevent custom histogram reducers from accidentally retaining an autograd graph."""
    observed_requires_grad: list[bool] = []

    def custom_histogram(
        tensor: torch.Tensor,
        *,
        sample_id: int,
    ) -> HistogramReductionResult:
        """Record graph attachment without producing a histogram."""
        del sample_id
        observed_requires_grad.append(tensor.requires_grad)
        return HistogramReductionResult(histograms={}, unavailable_histograms={})

    reduce_histograms(
        torch.ones(2, requires_grad=True),
        [custom_histogram],
        sample_id=0,
    )

    assert observed_requires_grad == [False]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"bins": 0}, "bins"),
        ({"every_n_samples": 0}, "every_n_samples"),
        ({"value_range": (1.0, 1.0)}, "lower bound"),
        ({"name": " "}, "name"),
    ],
)
def test_histogram_rejects_invalid_configuration(
    kwargs: dict[str, object],
    message: str,
) -> None:
    """Reject configurations whose serialized meaning would be ambiguous."""
    with pytest.raises(ValueError, match=message):
        histogram(**kwargs)  # type: ignore[arg-type]
