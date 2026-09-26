"""Compare Transformer telemetry against independent baseline tensor hooks."""

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch import nn
from torch.utils.hooks import RemovableHandle
from transformers import AutoModel, BertConfig, BertModel, GPT2Config, GPT2Model

from torchinstruments import AlwaysSampler, inject_observer, remove_observer
from torchinstruments.pytree import iter_tensor_leaves
from torchinstruments.sinks.paths import tensor_path_prefix


@dataclass(frozen=True)
class ExpectedTensor:
    """Retain only compact baseline measurements for one tensor signal."""

    layer: str
    path: str
    call_index: int
    signal: str
    statistics: dict[str, float]
    count: int


@pytest.fixture(params=["bert", "gpt2"])
def architecture(request: pytest.FixtureRequest) -> str:
    """Exercise both bidirectional and causal Transformer implementations."""
    value: object = request.param
    match value:
        case str():
            return value
        case _:
            pytest.fail("architecture must be a string")


@pytest.fixture(params=["config", "hub"])
def transformer(request: pytest.FixtureRequest, architecture: str) -> nn.Module:
    """Build offline models or fetch immutable tiny checkpoints from the Hub."""
    if request.param == "hub":
        if not request.config.getoption("--hf-hub"):
            pytest.skip("pass --hf-hub to download and test pinned checkpoints")
        revisions = {
            "bert": "f171d7baecaf37b5da5a3616d8833b9969753535",
            "gpt2": "71034c5d8bde858ff824298bdedc65515b97d2b9",
        }
        model: object = AutoModel.from_pretrained(
            f"hf-internal-testing/tiny-random-{architecture}",
            revision=revisions[architecture],
            use_safetensors=True,
            attn_implementation="eager",
        )
        match model:
            case nn.Module():
                return model
            case _:
                pytest.fail("checkpoint loader must return a module")
    if architecture == "bert":
        return BertModel(
            BertConfig(
                vocab_size=32,
                hidden_size=16,
                num_hidden_layers=2,
                num_attention_heads=2,
                intermediate_size=24,
            )
        )
    return GPT2Model(
        GPT2Config(
            vocab_size=32,
            n_embd=16,
            n_layer=2,
            n_head=2,
            n_positions=16,
            use_cache=False,
        )
    )


def baseline_hooks(model: nn.Module, expected: list[ExpectedTensor]) -> Iterator[RemovableHandle]:
    """Observe actual forward calls and gradients without using observer capture."""
    for name, module in model.named_modules():
        yield module.register_forward_hook(baseline_hook(name, expected))


def baseline_hook(
    name: str, expected: list[ExpectedTensor]
) -> Callable[[nn.Module, tuple[object, ...], object], None]:
    """Bind the canonical layer name and invocation counter to a baseline hook."""
    calls = 0

    def hook(module: nn.Module, inputs: tuple[object, ...], output: object) -> None:
        """Record each output and bind its gradient to this exact invocation."""
        nonlocal calls
        call_index = calls
        calls += 1
        for leaf in iter_tensor_leaves(output, "output"):
            expected.append(measure(name, leaf.path, call_index, "output", leaf.tensor))
            if leaf.tensor.requires_grad:
                leaf.tensor.register_hook(gradient_hook(name, leaf.path, call_index, expected))

    return hook


def gradient_hook(
    name: str, path: str, call_index: int, expected: list[ExpectedTensor]
) -> Callable[[torch.Tensor], None]:
    """Retain baseline output-gradient measurements without replacing gradients."""

    def hook(gradient: torch.Tensor) -> None:
        """Record the gradient delivered by autograd to this output."""
        expected.append(measure(name, path, call_index, "output_gradient", gradient))

    return hook


def measure(
    name: str, path: str, call_index: int, signal: str, tensor: torch.Tensor
) -> ExpectedTensor:
    """Compute an independent tensor-wide baseline before discarding the tensor."""
    values = tensor.detach().float().flatten()
    quartiles = values.quantile(torch.tensor([0.25, 0.5, 0.75])).tolist()
    statistics = {
        "mean": values.mean().item(),
        "std": values.std(correction=0).item(),
        "min": values.min().item(),
        "max": values.max().item(),
        "p25": quartiles[0],
        "p50": quartiles[1],
        "p75": quartiles[2],
        "zero_fraction": (values == 0).float().mean().item(),
        "nonfinite_fraction": (~values.isfinite()).float().mean().item(),
    }
    return ExpectedTensor(name, path, call_index, signal, statistics, tensor.numel())


def run_backward(model: nn.Module) -> torch.Tensor:
    """Use deterministic dropout and a loss connected to every returned tensor."""
    torch.manual_seed(123)
    output = model(input_ids=torch.tensor([[1, 2, 3, 4]]), use_cache=False)
    leaves = list(iter_tensor_leaves(output, "output"))
    loss = torch.stack([leaf.tensor.square().sum() for leaf in leaves]).sum()
    loss.backward()
    return leaves[0].tensor.detach().clone()


@pytest.mark.parametrize("training", [True, False], ids=["train", "eval"])
def test_transformer_layer_coverage(
    transformer: nn.Module,
    telemetry_dir: Path,
    training: bool,
) -> None:
    """Require every executed tensor in all artifacts and preserve model behavior."""
    transformer.train(training)
    expected: list[ExpectedTensor] = []
    handles = list(baseline_hooks(transformer, expected))
    try:
        baseline = run_backward(transformer)
    finally:
        for handle in handles:
            handle.remove()
    gradients = {
        name: p.grad.clone() for name, p in transformer.named_parameters() if p.grad is not None
    }
    transformer.zero_grad(set_to_none=True)
    inject_observer(
        transformer, sampler=AlwaysSampler(), output_dir=telemetry_dir, error_policy="raise"
    )
    try:
        actual = run_backward(transformer)
        assert torch.equal(actual, baseline)
        for name, parameter in transformer.named_parameters():
            if name in gradients:
                assert parameter.grad is not None
                assert torch.equal(parameter.grad, gradients[name]), name
            else:
                assert parameter.grad is None, name
        assert_artifacts(telemetry_dir, transformer, expected, training)
    finally:
        remove_observer(transformer)


def assert_artifacts(
    directory: Path, model: nn.Module, expected: list[ExpectedTensor], training: bool
) -> None:
    """Compare live scalar values and complete histogram tags to baseline hooks."""
    history = pl.read_parquet(directory / "history.parquet")
    summary = pl.read_json(directory / "result.json")
    assert set(summary["layer"]) == set(dict(model.named_modules()))
    measured = summary.filter(pl.col("tensors").list.len() > 0)
    assert set(measured["layer"]) == {item.layer for item in expected}
    assert len(set(measured["layer"])) > 8
    assert history.height == sum(len(item.statistics) for item in expected)
    tensors = measured.explode("tensors", empty_as_null=False).unnest("tensors")
    assert tensors.height == len(expected)
    events = EventAccumulator(
        str(directory / "tensorboard"), size_guidance={"histograms": 0}
    ).Reload()
    mode = "train" if training else "eval"
    histogram_tags = events.Tags()["histograms"]
    assert isinstance(histogram_tags, list)
    tags = set()
    for item in expected:
        path = (
            item.path if item.signal == "output" else item.path.replace("output", "grad_output", 1)
        )
        assert_statistics(history, tensors, item, path, mode)
        base = tensor_path_prefix(item.layer, item.call_index, path)
        tag = f"torchinstruments/{mode}/grad_enabled_true/{base}/histograms/distribution"
        tags.add(tag)
        assert tag in histogram_tags, item
        histogram = events.Histograms(tag)[0]
        assert histogram.step == 0
        assert histogram.histogram_value.num == item.count
        assert sum(histogram.histogram_value.bucket) == item.count
    assert set(histogram_tags) == tags


def assert_statistics(
    history: pl.DataFrame, tensors: pl.DataFrame, item: ExpectedTensor, path: str, mode: str
) -> None:
    """Require every scalar in Parquet and JSON to match the independent baseline."""
    identity = (
        (pl.col("layer") == item.layer)
        & (pl.col("tensor_path") == path)
        & (pl.col("signal") == item.signal)
        & (pl.col("call_index") == item.call_index)
        & (pl.col("mode") == mode)
        & pl.col("grad_enabled")
    )
    rows = history.filter(identity)
    summary = tensors.filter(identity)
    assert summary.height == 1, item
    assert set(rows["metric"]) == set(item.statistics), item
    for metric, value in item.statistics.items():
        row = rows.filter(pl.col("metric") == metric)
        assert row.height == 1, item
        assert row["value"].item() == pytest.approx(value, abs=1e-7), (item, metric)
        aggregate = summary.select(pl.col("statistics").struct.field(metric)).unnest(metric)
        assert aggregate["mean"].item() == pytest.approx(value, abs=1e-7), (item, metric)
        assert aggregate["observations"].item() == 1
        assert aggregate["unavailable"].item() == 0
