"""Compare normal interval sampling and every-forward stress on three fixed workloads."""

from __future__ import annotations

import json
import platform
import resource
import time
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import NamedTuple, assert_never
from uuid import uuid4

import polars as pl
import torch
from pydantic import Field
from pydantic_settings import BaseSettings, CliApp, CliImplicitFlag
from torch import nn
from tqdm import tqdm

from torchinstruments import AlwaysSampler, __version__, inject_observer, remove_observer


class Workload(StrEnum):
    """Name fixed model shapes instead of exposing geometry knobs."""

    SEQUENCE = "sequence"
    CONV = "conv"
    MANY_LAYERS = "many-layers"


class Sampling(StrEnum):
    """Distinguish intended usage, a stress test, and an observer-free reference."""

    INTERVAL = "interval"
    EVERY_FORWARD = "every-forward"
    OFF = "off"


class Device(StrEnum):
    """Choose a backend explicitly; unavailable devices never silently fall back."""

    CPU = "cpu"
    MPS = "mps"
    CUDA = "cuda"


class Settings(BaseSettings):
    """Select a workload, device, and observation mode with fixed run sizes."""

    workload: Workload = Workload.SEQUENCE
    sampling: Sampling = Sampling.INTERVAL
    device: Device = Device.CPU
    output: Path = Field(default_factory=lambda: Path("stats") / f"benchmark-{uuid4().hex[:8]}")
    quiet: CliImplicitFlag[bool] = False

    def cli_cmd(self) -> None:
        """Run the comparison and report the automatically created output directory."""
        run(self)
        print(self.output)


class Timing(NamedTuple):
    """Keep each synchronized forward and backward duration in milliseconds."""

    forward_ms: float
    backward_ms: float


def workload(name: Workload) -> tuple[nn.Module, tuple[int, ...]]:
    """Build a sequence model, convolutional model, or high module-count stress case."""
    match name:
        case Workload.SEQUENCE:
            return nn.Sequential(*(nn.Linear(128, 128) for _ in range(4))), (2, 128, 128)
        case Workload.CONV:
            return nn.Sequential(*(nn.Conv2d(64, 64, 3, padding=1) for _ in range(4))), (
                2,
                64,
                64,
                64,
            )
        case Workload.MANY_LAYERS:
            return nn.Sequential(*(nn.Identity() for _ in range(1000))), (2, 8)
        case _:
            assert_never(name)


def synchronize(device: Device) -> None:
    """Include actual device completion in timing boundaries."""
    match device:
        case Device.CPU:
            return
        case Device.MPS:
            torch.mps.synchronize()
        case Device.CUDA:
            torch.cuda.synchronize()
        case _:
            assert_never(device)


def step(model: nn.Module, inputs: torch.Tensor, device: Device) -> Timing:
    """Measure one forward and backward without optimizer or data-loading costs."""
    model.zero_grad(set_to_none=True)
    inputs.grad = None
    synchronize(device)
    started = time.perf_counter()
    output = model(inputs)
    synchronize(device)
    forward_ms = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    output.mean().backward()
    synchronize(device)
    return Timing(forward_ms, (time.perf_counter() - started) * 1000)


def attach(model: nn.Module, config: Settings) -> None:
    """Keep interval sampling separate from the explicitly requested stress mode."""
    match config.sampling:
        case Sampling.INTERVAL:
            inject_observer(model, output_dir=config.output / "telemetry", error_policy="raise")
        case Sampling.EVERY_FORWARD:
            inject_observer(
                model,
                output_dir=config.output / "telemetry",
                sampler=AlwaysSampler(),
                error_policy="raise",
            )
        case Sampling.OFF:
            return
        case _:
            assert_never(config.sampling)


def run(config: Settings) -> None:
    """Warm up, measure twenty iterations, and preserve raw timings and configuration."""
    torch.set_num_threads(1)
    torch.manual_seed(7)
    model, shape = workload(config.workload)
    model.to(config.device.value)
    inputs = torch.randn(shape, device=config.device.value, requires_grad=True)
    config.output.mkdir(parents=True, exist_ok=False)
    for _ in range(3):
        step(model, inputs, config.device)
    attach(model, config)
    try:
        timings = [
            step(model, inputs, config.device)
            for _ in tqdm(range(20), desc="Measuring", disable=config.quiet)
        ]
    finally:
        started = time.perf_counter()
        remove_observer(model)
        removal_seconds = time.perf_counter() - started
    frame = pl.DataFrame(timings, schema=list(Timing._fields), orient="row")
    frame.write_parquet(config.output / "timings.parquet")
    summary = frame.select(pl.all().mean()).with_columns(
        pl.lit(removal_seconds).alias("removal_seconds"),
        pl.lit(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss).alias("peak_rss_native_units"),
    )
    metadata = config.model_dump(mode="json") | {
        "torch_version": str(torch.__version__),
        "observer_version": __version__,
        "polars_version": pl.__version__,
        "platform": platform.platform(),
        "recorded_at": datetime.now(UTC).isoformat(),
        "shape": shape,
        "iterations": 20,
        "warmup": 3,
        "threads": 1,
        "dtype": "float32",
        "rss_unit": "bytes" if platform.system() == "Darwin" else "KiB",
    }
    (config.output / "configuration.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (config.output / "result.json").write_text(
        json.dumps(json.loads(summary.write_json()), indent=2) + "\n"
    )


if __name__ == "__main__":
    CliApp.run(Settings)
