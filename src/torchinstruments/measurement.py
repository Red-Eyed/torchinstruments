"""Per-module measurement plans keep histogram selection outside capture lifecycle logic."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from torchinstruments.records import TensorRecord
from torchinstruments.reducers import HistogramReducer, Reducer, reduce_histograms, reduce_tensor


@dataclass(frozen=True)
class TensorMeasurement:
    """Reduce one tensor with a fixed set of statistics and histograms."""

    reducers: Sequence[Reducer]
    histograms: Sequence[HistogramReducer]

    def __call__(self, tensor: torch.Tensor, *, sample_id: int) -> TensorRecord:
        """Transfer only compact measurements and detach them from the model graph."""
        scalar = reduce_tensor(tensor, self.reducers)
        histogram = reduce_histograms(tensor, self.histograms, sample_id=sample_id)
        return TensorRecord(
            shape=tuple(tensor.shape),
            dtype=str(tensor.dtype).removeprefix("torch."),
            device=str(tensor.device),
            numel=tensor.numel(),
            stats=scalar.stats,
            unavailable_stats=scalar.unavailable_stats,
            histograms=histogram.histograms,
            unavailable_histograms=histogram.unavailable_histograms,
        )


@dataclass(frozen=True)
class Measurements:
    """Bind scalar collection everywhere and histogram collection only at selected modules."""

    reducers: Sequence[Reducer]
    histograms: Sequence[HistogramReducer]
    by_module: dict[str, TensorMeasurement]
