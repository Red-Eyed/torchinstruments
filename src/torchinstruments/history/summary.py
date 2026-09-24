"""Polars aggregation of scalar histories into descriptive per-layer summaries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from torchinstruments.records import ModuleRecord


@dataclass(frozen=True)
class HistoryConfig:
    """Configure bounded write buffers and the two summary comparison windows."""

    buffer_rows: int = 8192
    window: int = 20

    def __post_init__(self) -> None:
        """Reject settings that cannot retain useful history or comparison windows."""
        for value in (self.buffer_rows, self.window):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("history limits must be positive integers")


_IDENTITY = ["layer", "call_index", "signal", "tensor_path", "metric"]


def aggregate_history(source: Path, window: int) -> pl.LazyFrame:
    """Describe each metric with Polars expressions, preserving missing measurements."""
    value = pl.col("value").sort_by("sample_id")
    recent = value.tail(window)
    previous = value.head((pl.len().cast(pl.Int64) - window).clip(0)).tail(window)
    point = pl.struct("sample_id", "timestamp", "value", "unavailable_reason")
    return (
        pl.scan_parquet(source)
        .group_by(_IDENTITY)
        .agg(
            pl.len().alias("observations"),
            pl.col("value").null_count().alias("unavailable_observations"),
            point.sort_by("sample_id").first().alias("first"),
            point.sort_by("sample_id").last().alias("latest"),
            point.sort_by("value", "sample_id", nulls_last=True).first().alias("minimum"),
            point.sort_by("value", "sample_id", descending=[True, False], nulls_last=True)
            .first()
            .alias("maximum"),
            pl.col("shape").sort_by("sample_id").last().alias("latest_shape"),
            pl.col("dtype").sort_by("sample_id").last(),
            _window(recent).alias("recent_window"),
            _window(previous).alias("previous_window"),
        )
    )


def _window(values: pl.Expr) -> pl.Expr:
    """Label means as means of sample statistics and report actual valid counts."""
    average = values.mean()
    return pl.struct(
        values.len().alias("observations"),
        values.count().alias("valid_observations"),
        pl.when(average.is_finite()).then(average).otherwise(None).alias("mean"),
        pl.when(average.is_finite())
        .then(pl.lit(""))
        .otherwise(pl.lit("no finite mean available"))
        .alias("unavailable_reason"),
    )


def layer_results(source: Path, modules: dict[str, ModuleRecord], window: int) -> pl.LazyFrame:
    """Build nested layer records with Polars, including unexecuted selected modules."""
    catalog = pl.DataFrame(
        {
            "layer": list(modules),
            "type": [module.type for module in modules.values()],
            "aliases": [list(module.aliases) for module in modules.values()],
        },
        schema={"layer": pl.String, "type": pl.String, "aliases": pl.List(pl.String)},
    ).lazy()
    metrics = aggregate_history(source, window)
    metric_fields = [
        "metric",
        "observations",
        "unavailable_observations",
        "first",
        "latest",
        "minimum",
        "maximum",
        "previous_window",
        "recent_window",
    ]
    tensors = metrics.group_by("layer", "call_index", "signal", "tensor_path").agg(
        pl.col("latest_shape").first(),
        pl.col("dtype").first(),
        pl.struct(metric_fields).sort_by("metric").alias("statistics"),
    )
    layers = tensors.group_by("layer").agg(
        pl.struct("call_index", "signal", "tensor_path", "latest_shape", "dtype", "statistics")
        .sort_by("call_index", "signal", "tensor_path")
        .alias("tensors"),
    )
    return (
        catalog.join(layers, on="layer", how="left")
        .with_columns(
            pl.col("tensors").fill_null([]),
        )
        .sort("layer")
    )


def write_result(
    source: Path, destination: Path, modules: dict[str, ModuleRecord], window: int
) -> None:
    """Write only per-layer information to JSON using the Polars JSON serializer."""
    temporary = destination.with_suffix(".json.tmp")
    try:
        layer_results(source, modules, window).collect(engine="streaming").write_json(temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
