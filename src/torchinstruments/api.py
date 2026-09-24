"""Attach per-layer history, summaries, and focused TensorBoard histograms."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import timedelta
from enum import Enum
from pathlib import Path

from torch import nn

from torchinstruments.capture import ForwardCallCapture, HookCallCapture
from torchinstruments.distributed import RankPolicy, detect_rank, parse_rank_policy, rank_is_enabled
from torchinstruments.errors import ErrorPolicy, ObserverAlreadyAttachedError, parse_error_policy
from torchinstruments.history.summary import HistoryConfig
from torchinstruments.measurement import Measurements, TensorMeasurement
from torchinstruments.observer import Observer
from torchinstruments.reducers import HistogramReducer, Reducer, default_reducers, histogram
from torchinstruments.sampling import SamplingPolicy, TimedSampler
from torchinstruments.selectors import ModuleSelector, leaf_modules
from torchinstruments.sinks import CompositeSink, DirectorySink, Sink

_DEFAULT_HISTORY = HistoryConfig()

_OBSERVER_ATTRIBUTE = "__torchinstruments_observer__"
_DEFAULT_INTERVAL = timedelta(minutes=1)


class _UseDefault(Enum):
    """Distinguish an omitted injectable component from user configuration."""

    TOKEN = "use_default"


_USE_DEFAULT = _UseDefault.TOKEN


def inject_observer(
    model: nn.Module,
    *,
    interval: timedelta = _DEFAULT_INTERVAL,
    output_dir: str | Path = "stats",
    sampler: SamplingPolicy | _UseDefault = _USE_DEFAULT,
    selector: ModuleSelector | _UseDefault = _USE_DEFAULT,
    reducers: Sequence[Reducer] | _UseDefault = _USE_DEFAULT,
    histograms: Sequence[HistogramReducer] | _UseDefault = _USE_DEFAULT,
    histogram_selector: ModuleSelector | _UseDefault = _USE_DEFAULT,
    max_histogram_modules: int = 8,
    sink: Sink | _UseDefault = _USE_DEFAULT,
    history_config: HistoryConfig = _DEFAULT_HISTORY,
    rank_policy: RankPolicy | str = RankPolicy.RANK0,
    error_policy: ErrorPolicy | str = ErrorPolicy.WARN,
    capture_direct_forwards: bool = False,
) -> None:
    """Attach passive observation without changing the training loop.

    Every directory run writes history, a per-layer JSON summary, an index, and TensorBoard
    events. Histograms cover at most ``max_histogram_modules`` selected modules in traversal
    order; supply ``histogram_selector`` to focus them. Scalars cover all selected modules.
    Supplied sinks receive an additional copy of events; a supplied DirectorySink owns the
    run destination itself. Externally supplied sinks retain their own resource ownership.
    """
    if hasattr(model, _OBSERVER_ATTRIBUTE):
        raise ObserverAlreadyAttachedError("model already has a TorchInstruments observer")
    rank = detect_rank()
    policy = parse_rank_policy(rank_policy)
    if not rank_is_enabled(policy, rank):
        return
    if (
        isinstance(max_histogram_modules, bool)
        or not isinstance(max_histogram_modules, int)
        or max_histogram_modules < 1
    ):
        raise ValueError("max_histogram_modules must be positive")
    if sampler is not _USE_DEFAULT and interval != _DEFAULT_INTERVAL:
        raise ValueError("interval and sampler cannot be configured together")
    resolved_sampler = TimedSampler(interval) if sampler is _USE_DEFAULT else sampler
    select = leaf_modules() if selector is _USE_DEFAULT else selector
    selected = {name: module for name, module in model.named_modules() if select(name, module)}
    scalar = default_reducers() if reducers is _USE_DEFAULT else tuple(reducers)
    if not scalar:
        raise ValueError("at least one scalar reducer is required")
    distributions = (
        (histogram(every_n_samples=1),) if histograms is _USE_DEFAULT else tuple(histograms)
    )
    measurements = _measurements(
        selected, scalar, distributions, histogram_selector, max_histogram_modules
    )
    directory = (
        sink
        if isinstance(sink, DirectorySink)
        else DirectorySink(
            output_dir,
            history_config=history_config,
            rank=rank,
            isolate_rank=rank.is_distributed and policy is RankPolicy.ALL,
        )
    )
    destination = (
        directory if sink is _USE_DEFAULT or sink is directory else CompositeSink(directory, sink)
    )
    observer = Observer(
        model=model,
        sampler=resolved_sampler,
        selector=lambda name, module: name in selected,
        measurements=measurements,
        sink=destination,
        error_policy=parse_error_policy(error_policy),
        capture=ForwardCallCapture() if capture_direct_forwards else HookCallCapture(),
    )
    observer.attach()
    setattr(model, _OBSERVER_ATTRIBUTE, observer)


def remove_observer(model: nn.Module) -> None:
    """Detach capture and finalize owned output, including the consolidated Parquet history."""
    observer = getattr(model, _OBSERVER_ATTRIBUTE, _USE_DEFAULT)
    if not isinstance(observer, Observer):
        return
    observer.remove()
    delattr(model, _OBSERVER_ATTRIBUTE)


def has_observer(model: nn.Module) -> bool:
    """Report whether the model currently owns an attached observer."""
    return isinstance(getattr(model, _OBSERVER_ATTRIBUTE, _USE_DEFAULT), Observer)


def _measurements(
    selected: dict[str, nn.Module],
    scalar: Sequence[Reducer],
    distributions: Sequence[HistogramReducer],
    histogram_selector: ModuleSelector | _UseDefault,
    max_histogram_modules: int,
) -> Measurements:
    """Bind per-module plans so capture never decides histogram policy."""
    if not distributions:
        raise ValueError("at least one histogram reducer is required")
    focus = [
        name
        for name, module in selected.items()
        if histogram_selector is _USE_DEFAULT or histogram_selector(name, module)
    ][:max_histogram_modules]
    if selected and not focus:
        raise ValueError("histogram_selector must match at least one observed module")
    focused = set(focus)
    return Measurements(
        scalar,
        distributions,
        {
            name: TensorMeasurement(scalar, distributions if name in focused else ())
            for name in selected
        },
    )
