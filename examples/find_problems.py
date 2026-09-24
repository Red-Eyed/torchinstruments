"""Reproduce five measurement signatures and compare each fault with a controlled fix."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import assert_never
from uuid import uuid4

import matplotlib
import polars as pl
import torch
from pydantic import Field
from pydantic_settings import BaseSettings, CliApp, CliImplicitFlag
from torch import nn
from tqdm import tqdm

from torchinstruments import AlwaysSampler, HistoryConfig, inject_observer, remove_observer

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class Problem(StrEnum):
    """Name controlled experiments, never categories embedded in telemetry."""

    INACTIVE_RELU = "inactive_relu"
    SATURATION = "saturation"
    SCALE_GROWTH = "scale_growth"
    NONFINITE = "nonfinite"
    DETACHED = "detached"


@dataclass(frozen=True)
class Experiment:
    """Define the observation used to test a known intervention."""

    layer: str
    signal: str
    metric: str
    interpretation: str
    fix: str


EXPERIMENTS = {
    Problem.INACTIVE_RELU: Experiment(
        "activation",
        "output",
        "zero_fraction",
        "The ReLU becomes entirely zero on the observed positive input batches.",
        "Restore the pre-activation bias instead of forcing it strongly negative.",
    ),
    Problem.SATURATION: Experiment(
        "pre",
        "output_gradient",
        "max",
        "Gradients at the layer BEFORE the sigmoid collapse when its inputs saturate.",
        "Restore the pre-activation scale and bias.",
    ),
    Problem.SCALE_GROWTH: Experiment(
        "gain",
        "output",
        "std",
        "The output spread grows after an increasing multiplicative gain is introduced.",
        "Keep the gain at one.",
    ),
    Problem.NONFINITE: Experiment(
        "probe",
        "output",
        "nonfinite_fraction",
        "The probe produces invalid values after taking log of negative inputs.",
        "Use the intended log1p(abs(x)) operation with a valid domain.",
    ),
    Problem.DETACHED: Experiment(
        "pre",
        "output_gradient",
        "max",
        "Upstream backward measurements disappear while forward and head gradients continue.",
        "Keep the bridge connected instead of applying detach().",
    ),
}


class Gain(nn.Module):
    """Expose one explicitly controlled multiplicative gain."""

    def __init__(self) -> None:
        """Start at unit gain."""
        super().__init__()
        self.factor = 1.0

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the gain without hiding it inside another module."""
        return inputs * self.factor


class Bridge(nn.Module):
    """Reproduce accidental disconnection without stopping downstream training."""

    def __init__(self) -> None:
        """Start with the computation graph connected."""
        super().__init__()
        self.disconnected = False

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Detach only when the experiment explicitly introduces the fault."""
        return inputs.detach() if self.disconnected else inputs


class LogProbe(nn.Module):
    """Expose a controlled invalid-domain operation at an observable boundary."""

    def __init__(self) -> None:
        """Start with an operation defined on all real inputs."""
        super().__init__()
        self.invalid = False

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Switch between intended and deliberately invalid logarithms."""
        return torch.log(-inputs.abs()) if self.invalid else torch.log1p(inputs.abs())


class DiagnosticModel(nn.Module):
    """Expose concrete typed components for controlled interventions."""

    def __init__(self, problem: Problem) -> None:
        """Initialize matched models with named observable boundaries."""
        super().__init__()
        self.pre = nn.Linear(2, 2)
        self.activation = nn.Sigmoid() if problem is Problem.SATURATION else nn.ReLU()
        self.gain = Gain()
        self.bridge = Bridge()
        self.probe = LogProbe()
        self.use_log = problem is Problem.NONFINITE
        self.head = nn.Linear(2, 1)
        with torch.no_grad():
            self.pre.weight.copy_(torch.eye(2))
            assert self.pre.bias is not None and self.head.bias is not None
            self.pre.bias.fill_(0.2)
            self.head.weight.fill_(0.5)
            self.head.bias.zero_()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the named blocks without hiding the diagnostic boundaries."""
        hidden = self.activation(self.pre(inputs))
        hidden = self.bridge(self.gain(hidden))
        if self.use_log:
            hidden = self.probe(hidden)
        return self.head(hidden)


def introduce_fault(model: DiagnosticModel, problem: Problem, step: int) -> None:
    """Change one known mechanism after four healthy samples."""
    if step < 4:
        return
    with torch.no_grad():
        assert model.pre.bias is not None
        match problem:
            case Problem.INACTIVE_RELU:
                model.pre.bias.fill_(-10)
            case Problem.SATURATION:
                model.pre.weight.copy_(torch.eye(2) * 30)
                model.pre.bias.fill_(30)
            case Problem.SCALE_GROWTH:
                model.gain.factor = 10 ** ((step - 3) / 3)
            case Problem.NONFINITE:
                model.probe.invalid = True
            case Problem.DETACHED:
                model.bridge.disconnected = True
            case _:
                assert_never(problem)


def run_case(problem: Problem, variant: str, output: Path, steps: int) -> None:
    """Train matched runs and finalize all four telemetry artifacts automatically."""
    torch.manual_seed(17)
    generator = torch.Generator().manual_seed(41)
    model = DiagnosticModel(problem)
    # Fixed weights isolate gain growth from secondary optimizer-induced activation collapse.
    learning_rate = 0.0 if problem is Problem.SCALE_GROWTH else 0.01
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    experiment = EXPERIMENTS[problem]
    focus = {experiment.layer, "pre", "activation", "head"}
    inject_observer(
        model,
        output_dir=output,
        sampler=AlwaysSampler(),
        histogram_selector=lambda name, module: name in focus,
        history_config=HistoryConfig(window=4),
        error_policy="raise",
    )
    try:
        for step in range(steps):
            if variant == "broken":
                introduce_fault(model, problem, step)
            inputs = torch.rand(32, 2, generator=generator) + 0.1
            optimizer.zero_grad()
            prediction = model(inputs)
            loss = (prediction - inputs.sum(dim=1, keepdim=True)).square().mean()
            loss.backward()
            if torch.isfinite(loss):
                optimizer.step()
    finally:
        remove_observer(model)


def evidence(path: Path, experiment: Experiment) -> pl.DataFrame:
    """Query only the selected series and retain gaps as missing measurements."""
    history = pl.scan_parquet(path / "history.parquet")
    samples = history.select("sample_id").unique()
    values = history.filter(
        (pl.col("layer") == experiment.layer)
        & (pl.col("signal") == experiment.signal)
        & (pl.col("metric") == experiment.metric)
        & (pl.col("call_index") == 0)
    ).select("sample_id", "value")
    return samples.join(values, on="sample_id", how="left").sort("sample_id").collect()


def demonstrate(problem: Problem, root: Path, steps: int) -> None:
    """Run a controlled comparison, print evidence, and write a standalone plot."""
    experiment = EXPERIMENTS[problem]
    tables: list[pl.DataFrame] = []
    for variant in ("baseline", "broken", "fixed"):
        destination = root / problem.value / variant
        run_case(problem, variant, destination, steps)
        table = evidence(destination, experiment).with_columns(pl.lit(variant).alias("run"))
        tables.append(table)
    destination = root / problem.value
    plot_comparison(problem, tables, destination / "comparison.png")
    comparison = pl.concat(tables)
    comparison.write_csv(destination / "comparison.csv")
    print(f"\n{problem}: {experiment.interpretation}\nFix: {experiment.fix}")
    print(comparison.filter(pl.col("sample_id") >= steps - 3))
    (destination / "README.md").write_text(
        f"# {problem.value}\n\n{experiment.interpretation}\n\n"
        f"Query `{experiment.layer}`, `{experiment.signal}`, `{experiment.metric}` "
        "in history.parquet.\n"
        "The broken run introduces its fault at sample 4. "
        "Baseline and fixed runs share initialization, "
        "input batches, and optimizer settings; the fixed run removes that intervention.\n\n"
        f"Fix: {experiment.fix}\n\n"
        "See comparison.csv and comparison.png. "
        "A plot gap means no gradient measurement, not zero. "
        "These are controlled demonstrations, not universal thresholds or proof of model quality. "
        "Inspect finite-value coverage and compare the same boundaries across matched runs.\n",
        encoding="utf-8",
    )


def plot_comparison(problem: Problem, tables: list[pl.DataFrame], output: Path) -> None:
    """Draw distinct run styles and make missing backward observations visible."""
    experiment = EXPERIMENTS[problem]
    figure, axes = plt.subplots(figsize=(8, 4))
    styles = [(":", "o"), ("-", "x"), ("--", ".")]
    for table, (line, marker) in zip(tables, styles, strict=True):
        variant = table["run"][0]
        axes.plot(
            table["sample_id"],
            table["value"],
            linestyle=line,
            marker=marker,
            markersize=7,
            label=variant,
            fillstyle="none",
        )
    axes.axvline(4, color="gray", linestyle="--", label="fault introduced in broken run")
    if problem is Problem.DETACHED:
        axes.text(
            0.52,
            0.48,
            "Broken run: no upstream gradients\nafter sample 3",
            transform=axes.transAxes,
        )
    axes.set(xlabel="Sample ID", ylabel=experiment.metric, title=f"{problem}: {experiment.layer}")
    axes.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


class Settings(BaseSettings):
    """Configure reproducible demonstrations with progress enabled by default."""

    output: Path = Field(default_factory=lambda: Path("stats") / f"problems-{uuid4().hex[:8]}")
    steps: int = Field(default=12, ge=8)
    quiet: CliImplicitFlag[bool] = False

    def cli_cmd(self) -> None:
        """Create every experiment and announce where its evidence was written."""
        for problem in tqdm(list(Problem), desc="Debugging examples", disable=self.quiet):
            demonstrate(problem, self.output, self.steps)
        print(f"Examples written to {self.output}")


if __name__ == "__main__":
    CliApp.run(Settings)
