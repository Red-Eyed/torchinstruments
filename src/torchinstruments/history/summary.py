"""Polars aggregation of scalar histories into descriptive per-layer summaries."""

from __future__ import annotations

import json
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


_IDENTITY = ["layer", "call_index", "signal", "tensor_path", "mode", "grad_enabled", "metric"]


def aggregate_history(source: Path, window: int) -> pl.LazyFrame:
    """Describe each metric with Polars expressions, preserving missing measurements."""
    history = pl.scan_parquet(source)
    if not {"mode", "grad_enabled"}.issubset(history.collect_schema().names()):
        raise ValueError(
            "history lacks execution context; use the matching older reader or explicitly "
            "migrate it with known modes, never infer train/eval from missing gradients"
        )
    value = pl.col("value").sort_by("sample_id")
    recent = value.tail(window)
    previous = value.head((pl.len().cast(pl.Int64) - window).clip(0)).tail(window)
    summaries = [
        value.mean().alias("mean"),
        value.std(ddof=0).alias("std"),
        value.min().alias("min"),
        value.quantile(0.25, interpolation="linear").alias("p25"),
        value.quantile(0.5, interpolation="linear").alias("p50"),
        value.quantile(0.75, interpolation="linear").alias("p75"),
        value.max().alias("max"),
        previous.mean().alias("previous_window_mean"),
        recent.mean().alias("recent_window_mean"),
    ]
    return (
        history.group_by(_IDENTITY)
        .agg(
            pl.len().alias("observations"),
            pl.col("value").null_count().alias("unavailable"),
            *summaries,
            pl.col("shape").sort_by("sample_id").last().alias("latest_shape"),
            pl.col("dtype").sort_by("sample_id").last(),
        )
        .with_columns(
            pl.when(pl.col(name).is_finite()).then(pl.col(name)).otherwise(None).alias(name)
            for name in _AGGREGATES
        )
        .with_columns(
            pl.when(pl.any_horizontal(pl.col(name).is_null() for name in _AGGREGATES))
            .then(pl.lit("null aggregates have no finite observations or exceed numeric range"))
            .otherwise(pl.lit(""))
            .alias("unavailable_reason")
        )
    )


_AGGREGATES = [
    "mean",
    "std",
    "min",
    "p25",
    "p50",
    "p75",
    "max",
    "previous_window_mean",
    "recent_window_mean",
]


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
    metrics = aggregate_history(source, window).collect(engine="streaming")
    if metrics.is_empty():
        return catalog.with_columns(pl.lit([]).alias("tensors")).sort("layer")
    tensors = _tensor_summaries(metrics)
    layers = tensors.group_by("layer").agg(
        pl.struct(
            "call_index",
            "signal",
            "tensor_path",
            "latest_shape",
            "dtype",
            "mode",
            "grad_enabled",
            "statistics",
        )
        .sort_by("call_index", "signal", "tensor_path", "mode", "grad_enabled")
        .alias("tensors"),
    )
    return (
        catalog.join(layers, on="layer", how="left")
        .with_columns(pl.col("tensors").fill_null([]))
        .sort("layer")
    )


def _tensor_summaries(metrics: pl.DataFrame) -> pl.LazyFrame:
    """Pivot aggregated metrics into named fields without materializing history rows."""
    metric_fields = ["observations", "unavailable", *_AGGREGATES, "unavailable_reason"]
    metric_names = metrics["metric"].unique().sort().to_list()
    identity = [
        "layer",
        "call_index",
        "signal",
        "tensor_path",
        "latest_shape",
        "dtype",
        "mode",
        "grad_enabled",
    ]
    return (
        metrics.with_columns(
            pl.struct(metric_fields).alias("summary"),
            (pl.lit("metric:") + pl.col("metric")).alias("metric"),
        )
        .pivot(
            on="metric",
            index=identity,
            values="summary",
            sort_columns=True,
        )
        .lazy()
        .select(
            *identity,
            pl.struct(
                pl.col(f"metric:{name}").fill_null(_unobserved_metric()).alias(name)
                for name in metric_names
            ).alias("statistics"),
        )
    )


def _unobserved_metric() -> pl.Expr:
    """Explain metric gaps when reducers emit different names for different tensors."""
    return pl.struct(
        pl.lit(0, dtype=pl.UInt32).alias("observations"),
        pl.lit(0, dtype=pl.UInt32).alias("unavailable"),
        *(pl.lit(None, dtype=pl.Float64).alias(field) for field in _AGGREGATES),
        pl.lit("metric was not observed for this tensor").alias("unavailable_reason"),
    )


def write_result(
    source: Path, destination: Path, modules: dict[str, ModuleRecord], window: int
) -> None:
    """Indent Polars' serialized layer records and atomically publish readable JSON."""
    temporary = destination.with_suffix(".json.tmp")
    try:
        payload = layer_results(source, modules, window).collect(engine="streaming").write_json()
        with temporary.open("w", encoding="utf-8") as output:
            json.dump(json.loads(payload), output, indent=2, ensure_ascii=False, allow_nan=False)
            output.write("\n")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
