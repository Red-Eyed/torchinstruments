"""Bounded Polars writes and streaming finalization of scalar histories."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import TypedDict, assert_never

import polars as pl

from torchinstruments.history.records import Observation
from torchinstruments.records import Absent


class _Row(TypedDict):
    """Represent nullable Parquet cells with an accompanying absence reason."""

    layer: str
    call_index: int
    signal: str
    tensor_path: str
    sample_id: int
    forward_index: int
    timestamp: datetime
    shape: list[int]
    dtype: str
    metric: str
    value: float | None
    unavailable_reason: str
    mode: str
    grad_enabled: bool


SCHEMA = {
    "layer": pl.String,
    "call_index": pl.Int64,
    "signal": pl.String,
    "tensor_path": pl.String,
    "sample_id": pl.Int64,
    "forward_index": pl.Int64,
    "timestamp": pl.Datetime("us", "UTC"),
    "shape": pl.List(pl.Int64),
    "dtype": pl.String,
    "metric": pl.String,
    "value": pl.Float64,
    "unavailable_reason": pl.String,
    "mode": pl.String,
    "grad_enabled": pl.Boolean,
}


class ParquetHistory:
    """Write fixed-size buffers to immutable chunks, then stream them into one file."""

    def __init__(self, path: Path, *, buffer_rows: int) -> None:
        """Bind a destination without creating files before run initialization."""
        if buffer_rows < 1:
            raise ValueError("buffer_rows must be positive")
        self.path = path
        self.parts = path.with_suffix(".parts")
        self._buffer_rows = buffer_rows
        self._buffer: list[_Row] = []
        self._part = 0

    def initialize(self) -> None:
        """Create an empty queryable chunk with the complete typed schema."""
        self.parts.mkdir(parents=True, exist_ok=False)
        self._write(pl.DataFrame(schema=SCHEMA))
        self.publish()

    def append(self, observation: Observation) -> None:
        """Buffer one compact observation without retaining a tensor or full history."""
        value: float | None
        reason = ""
        match observation.value:
            case Absent(reason=missing):
                value, reason = None, missing
            case float() as measured:
                value = measured
            case _:
                assert_never(observation.value)
        self._buffer.append(
            _Row(
                layer=observation.layer,
                call_index=observation.call_index,
                signal=observation.signal.value,
                tensor_path=observation.tensor_path,
                sample_id=observation.sample_id,
                forward_index=observation.forward_index,
                timestamp=observation.timestamp,
                shape=list(observation.shape),
                dtype=observation.dtype,
                metric=observation.metric,
                value=value,
                unavailable_reason=reason,
                mode=observation.context.mode.value,
                grad_enabled=observation.context.grad_enabled,
            )
        )
        if len(self._buffer) >= self._buffer_rows:
            self.flush()

    def flush(self) -> None:
        """Publish buffered rows as one completed Parquet chunk."""
        if not self._buffer:
            return
        self._write(pl.DataFrame(self._buffer, schema=SCHEMA))
        self._buffer.clear()

    def publish(self) -> None:
        """Publish a readable history snapshot while retaining chunks for future samples."""
        self.flush()
        temporary = self.path.with_suffix(".parquet.tmp")
        try:
            pl.scan_parquet(self.parts / "*.parquet").sink_parquet(
                temporary,
                compression="zstd",
                engine="streaming",
            )
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)

    def close(self) -> None:
        """Publish remaining observations and remove the intermediate chunks."""
        self.publish()
        for part in self.parts.glob("*.parquet"):
            part.unlink()
        self.parts.rmdir()

    def _write(self, frame: pl.DataFrame) -> None:
        """Publish only complete chunks so live readers cannot see a partial footer."""
        destination = self.parts / f"{self._part:08d}.parquet"
        temporary = destination.with_suffix(".tmp")
        try:
            frame.write_parquet(temporary, compression="zstd")
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        self._part += 1
