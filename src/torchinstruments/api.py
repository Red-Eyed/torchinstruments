"""Model-level functions for attaching, inspecting, and removing observers."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import timedelta
from enum import Enum
from pathlib import Path

from torch import nn

from torchinstruments.capture import ForwardCallCapture, HookCallCapture
from torchinstruments.errors import (
    ErrorPolicy,
    ObserverAlreadyAttachedError,
    parse_error_policy,
)
from torchinstruments.observer import Observer
from torchinstruments.reducers import HistogramReducer, Reducer, default_reducers
from torchinstruments.sampling import SamplingPolicy, TimedSampler
from torchinstruments.selectors import ModuleSelector, leaf_modules
from torchinstruments.sinks import DirectorySink, Sink

_OBSERVER_ATTRIBUTE = "__torchinstruments_observer__"
_DEFAULT_INTERVAL = timedelta(minutes=1)
_DEFAULT_OUTPUT_DIR = Path("stats")


class _UseDefault(Enum):
    """Distinguish an omitted component from any valid user-supplied value."""

    TOKEN = "use_default"


_USE_DEFAULT = _UseDefault.TOKEN


def inject_observer(
    model: nn.Module,
    *,
    interval: timedelta = _DEFAULT_INTERVAL,
    output_dir: str | Path = _DEFAULT_OUTPUT_DIR,
    sampler: SamplingPolicy | _UseDefault = _USE_DEFAULT,
    selector: ModuleSelector | _UseDefault = _USE_DEFAULT,
    reducers: Sequence[Reducer] | _UseDefault = _USE_DEFAULT,
    histograms: Sequence[HistogramReducer] = (),
    sink: Sink | _UseDefault = _USE_DEFAULT,
    error_policy: ErrorPolicy | str = ErrorPolicy.WARN,
    capture_direct_forwards: bool = False,
) -> None:
    """Attach passive telemetry capture to ``model`` in place.

    Convenience arguments are mutually exclusive with their corresponding injected component:
    ``interval`` with ``sampler``, and ``output_dir`` with ``sink``. ``histograms`` is empty by
    default because distribution reduction is more expensive than scalar diagnostics; each
    configured histogram owns an independent snapshot cadence. Collection errors follow
    ``error_policy``. By default, native PyTorch hooks observe normal ``module(...)`` dispatch.
    Set ``capture_direct_forwards=True`` when model code invokes literal ``module.forward(...)``;
    this replaces selected modules' instance-level ``forward`` attributes until
    :func:`remove_observer` restores them. Successful attachment returns ``None``.
    """
    if hasattr(model, _OBSERVER_ATTRIBUTE):
        raise ObserverAlreadyAttachedError("model already has a TorchInstruments observer")

    resolved_sampler = _resolve_sampler(interval, sampler)
    resolved_selector = leaf_modules() if selector is _USE_DEFAULT else selector
    resolved_reducers = default_reducers() if reducers is _USE_DEFAULT else tuple(reducers)
    resolved_sink = _resolve_sink(output_dir, sink)
    capture = ForwardCallCapture() if capture_direct_forwards else HookCallCapture()

    observer = Observer(
        model=model,
        sampler=resolved_sampler,
        selector=resolved_selector,
        reducers=resolved_reducers,
        histograms=tuple(histograms),
        sink=resolved_sink,
        error_policy=parse_error_policy(error_policy),
        capture=capture,
    )
    observer.attach()
    setattr(model, _OBSERVER_ATTRIBUTE, observer)


def remove_observer(model: nn.Module) -> None:
    """Remove all TorchInstruments capture behavior and private state from ``model``.

    Native hooks are detached and any observer-owned forward wrappers are restored. Calling this
    function for a model without an observer is a no-op.
    """
    observer = getattr(model, _OBSERVER_ATTRIBUTE, _USE_DEFAULT)
    if not isinstance(observer, Observer):
        return

    observer.remove()
    delattr(model, _OBSERVER_ATTRIBUTE)


def has_observer(model: nn.Module) -> bool:
    """Report whether ``model`` currently owns a TorchInstruments observer."""
    return isinstance(getattr(model, _OBSERVER_ATTRIBUTE, _USE_DEFAULT), Observer)


def _resolve_sampler(
    interval: timedelta,
    sampler: SamplingPolicy | _UseDefault,
) -> SamplingPolicy:
    """Resolve convenience sampling arguments without hiding conflicting configuration."""
    if sampler is _USE_DEFAULT:
        return TimedSampler(interval)
    if interval != _DEFAULT_INTERVAL:
        raise ValueError("interval and sampler cannot be configured together")
    return sampler


def _resolve_sink(output_dir: str | Path, sink: Sink | _UseDefault) -> Sink:
    """Resolve the persistence sink and reject ambiguous output configuration."""
    if sink is _USE_DEFAULT:
        return DirectorySink(output_dir)
    if Path(output_dir) != _DEFAULT_OUTPUT_DIR:
        raise ValueError("output_dir and sink cannot be configured together")
    return sink
