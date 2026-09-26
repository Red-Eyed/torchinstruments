"""Typed scalar observations shared by history storage and summaries."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING

from torchinstruments.records import Absent, ExecutionContext, SampleRecord, SampleState

if TYPE_CHECKING:
    from collections.abc import Iterator


class Signal(StrEnum):
    """Distinguish activations from gradients with respect to those activations."""

    OUTPUT = "output"
    GRADIENT = "output_gradient"


@dataclass(frozen=True)
class Observation:
    """Locate one measured scalar or an explicit unavailable measurement."""

    layer: str
    call_index: int
    signal: Signal
    tensor_path: str
    sample_id: int
    forward_index: int
    timestamp: datetime
    shape: tuple[int, ...]
    dtype: str
    metric: str
    value: float | Absent
    context: ExecutionContext


def observations(sample: SampleRecord) -> Iterator[Observation]:
    """Emit only the new lifecycle stage, never duplicating forward values on backward."""
    forward = sample.state is SampleState.FORWARD_COMPLETE
    signal = Signal.OUTPUT if forward else Signal.GRADIENT
    for name, calls in sample.modules.items():
        for call in calls:
            tensors = call.outputs if forward else call.output_gradients
            for path, tensor in tensors.items():
                values: dict[str, float | Absent] = dict(tensor.stats)
                values.update(
                    (metric, Absent(reason)) for metric, reason in tensor.unavailable_stats.items()
                )
                for metric, value in values.items():
                    yield Observation(
                        name,
                        call.call_index,
                        signal,
                        path,
                        sample.sample_id,
                        sample.forward_index,
                        sample.timestamp,
                        tensor.shape,
                        tensor.dtype,
                        metric,
                        value,
                        call.context,
                    )
