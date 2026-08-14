"""Observer orchestration and sampled forward/backward lifecycle management."""

from __future__ import annotations

import contextvars
import threading
import time
import warnings
import weakref
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib.metadata import version as package_version

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle

from torchinstruments.capture import CallCapture, CaptureCallbacks
from torchinstruments.errors import ErrorPolicy
from torchinstruments.pytree import iter_tensor_leaves
from torchinstruments.records import (
    SCHEMA_VERSION,
    Absent,
    CollectionRecord,
    ErrorRecord,
    ModuleCallRecord,
    ModuleRecord,
    ReducerRecord,
    RunRecord,
    SamplingRecord,
    SnapshotRecord,
    SnapshotState,
    TensorRecord,
)
from torchinstruments.reducers import HistogramReducer, Reducer, reduce_histograms, reduce_tensor
from torchinstruments.reducers.base import DescribedReducer
from torchinstruments.sampling import SamplingEvent, SamplingPolicy
from torchinstruments.sampling.base import DescribedSamplingPolicy
from torchinstruments.selectors import ModuleSelector
from torchinstruments.sinks import Sink

_NOT_COLLECTING = object()


@dataclass(frozen=True)
class _GradientBinding:
    """Locate one gradient record inside a snapshot's module-call structure."""

    module_name: str
    call_index: int
    path: str


@dataclass(frozen=True)
class _GradientTarget:
    """Group every telemetry path that aliases the same graph tensor."""

    tensor: torch.Tensor
    bindings: tuple[_GradientBinding, ...]


@dataclass
class _MutableModuleCall:
    """Accumulate compact records for one selected module invocation."""

    call_index: int
    outputs: dict[str, TensorRecord] = field(default_factory=dict)
    output_gradients: dict[str, TensorRecord] = field(default_factory=dict)


class _SnapshotBuilder:
    """Build one snapshot while its forward and optional backward are in flight."""

    def __init__(
        self,
        *,
        snapshot_id: int,
        forward_index: int,
        timestamp: datetime,
    ) -> None:
        """Initialize compact mutable state for one sampled root forward."""
        self.snapshot_id = snapshot_id
        self.forward_index = forward_index
        self.timestamp = timestamp
        self.state = SnapshotState.FORWARD_COMPLETE
        self.collection_duration_ns = 0
        self.module_calls: dict[str, list[_MutableModuleCall]] = {}
        self.errors: list[ErrorRecord] = []
        self._gradient_targets: list[tuple[torch.Tensor, _GradientBinding]] = []
        self._lock = threading.Lock()

    def add_module_call(self, module_name: str) -> _MutableModuleCall:
        """Append and return the next ordered invocation for ``module_name``."""
        with self._lock:
            calls = self.module_calls.setdefault(module_name, [])
            call = _MutableModuleCall(call_index=len(calls))
            calls.append(call)
            return call

    def add_output(
        self,
        module_name: str,
        call: _MutableModuleCall,
        path: str,
        tensor: torch.Tensor,
        record: TensorRecord,
    ) -> None:
        """Store an output record and retain its short-lived gradient binding."""
        with self._lock:
            call.outputs[path] = record
            if tensor.requires_grad:
                gradient_path = path.replace("output", "grad_output", 1)
                binding = _GradientBinding(module_name, call.call_index, gradient_path)
                self._gradient_targets.append((tensor, binding))

    def add_output_gradient(self, binding: _GradientBinding, record: TensorRecord) -> None:
        """Attach a compact gradient record to its exact module invocation."""
        with self._lock:
            call = self.module_calls[binding.module_name][binding.call_index]
            call.output_gradients[binding.path] = record

    def take_gradient_targets(self) -> tuple[_GradientTarget, ...]:
        """Deduplicate aliased tensors and release builder-owned raw references."""
        with self._lock:
            grouped: dict[int, tuple[torch.Tensor, list[_GradientBinding]]] = {}
            for tensor, binding in self._gradient_targets:
                identity = id(tensor)
                if identity not in grouped:
                    grouped[identity] = (tensor, [])
                grouped[identity][1].append(binding)
            self._gradient_targets.clear()

        return tuple(
            _GradientTarget(tensor=tensor, bindings=tuple(bindings))
            for tensor, bindings in grouped.values()
        )

    def add_collection_duration(self, duration_ns: int) -> None:
        """Accumulate observer reduction time in monotonic nanoseconds."""
        with self._lock:
            self.collection_duration_ns += duration_ns

    def add_error(self, error: ErrorRecord) -> None:
        """Append an isolated collection failure to the snapshot."""
        with self._lock:
            self.errors.append(error)

    def mark_backward_observed(self) -> None:
        """Transition the snapshot after its first correlated backward callback."""
        with self._lock:
            self.state = SnapshotState.BACKWARD_OBSERVED

    def to_record(self) -> SnapshotRecord:
        """Freeze current mutable state into a normalized persistence record."""
        with self._lock:
            modules = {
                module_name: tuple(
                    ModuleCallRecord(
                        call_index=call.call_index,
                        outputs=dict(call.outputs),
                        output_gradients=dict(call.output_gradients),
                    )
                    for call in calls
                )
                for module_name, calls in self.module_calls.items()
            }
            return SnapshotRecord(
                schema_version=SCHEMA_VERSION,
                snapshot_id=self.snapshot_id,
                forward_index=self.forward_index,
                timestamp=self.timestamp,
                state=self.state,
                collection_duration_ms=self.collection_duration_ns / 1_000_000,
                modules=modules,
                errors=tuple(self.errors),
            )


class Observer:
    """Orchestrate sampling, invocation capture, reduction, and persistence."""

    def __init__(
        self,
        *,
        model: nn.Module,
        sampler: SamplingPolicy,
        selector: ModuleSelector,
        reducers: Sequence[Reducer],
        histograms: Sequence[HistogramReducer],
        sink: Sink,
        error_policy: ErrorPolicy,
        capture: CallCapture,
        monotonic_clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        performance_clock: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        """Bind replaceable components and clocks without mutating the model."""
        self._model = model
        self._sampler = sampler
        self._selector = selector
        self._reducers = tuple(reducers)
        self._histogram_reducers = tuple(histograms)
        self._sink = sink
        self._error_policy = error_policy
        self._capture = capture
        self._monotonic_clock = monotonic_clock
        self._wall_clock = wall_clock
        self._performance_clock = performance_clock
        self._forward_index = 0
        self._snapshot_id = 0
        self._id_lock = threading.Lock()
        self._sink_lock = threading.Lock()
        # Weak ownership lets completed or abandoned graphs release callback handles naturally,
        # while still allowing explicit observer removal to detach live callbacks.
        self._gradient_hook_handles: weakref.WeakSet[RemovableHandle] = weakref.WeakSet()
        self._gradient_handles_lock = threading.Lock()
        self._contexts: contextvars.ContextVar[tuple[object, ...]] = contextvars.ContextVar(
            f"torchinstruments_context_{id(self)}", default=()
        )

    def attach(self) -> None:
        """Initialize persistence and attach the configured invocation capture strategy."""
        selected_modules, module_records = self._select_modules()
        run = RunRecord(
            schema_version=SCHEMA_VERSION,
            created_at=self._wall_clock(),
            torch_version=str(torch.__version__),
            observer_version=package_version("torchinstruments"),
            sampling=self._sampling_record(),
            collection=CollectionRecord(
                invocation_capture=self._capture.capture_type(),
                signals=("module_outputs", "module_output_gradients"),
                scalar_reducers=self._reducer_records(self._reducers),
                histogram_reducers=self._reducer_records(self._histogram_reducers),
            ),
        )
        self._sink.initialize(run, module_records)

        try:
            self._capture.attach(
                self._model,
                selected_modules,
                CaptureCallbacks(
                    start_root=self._start_root_forward,
                    finish_root=self._finish_root_forward,
                    observe_output=self._collect_module_output,
                ),
            )
        except BaseException:
            self.remove()
            raise

    def remove(self) -> None:
        """Remove invocation capture and graph hooks, then close the configured sink."""
        self._capture.remove()

        with self._gradient_handles_lock:
            gradient_handles = tuple(self._gradient_hook_handles)
            self._gradient_hook_handles.clear()
        for handle in gradient_handles:
            handle.remove()

        self._sink.close()

    def _select_modules(self) -> tuple[list[tuple[str, nn.Module]], Mapping[str, ModuleRecord]]:
        """Select unique module objects while preserving every discovered alias."""
        aliases_by_identity: dict[int, list[str]] = {}
        modules_by_identity: dict[int, nn.Module] = {}
        for name, module in self._model.named_modules(remove_duplicate=False):
            identity = id(module)
            aliases_by_identity.setdefault(identity, []).append(name)
            modules_by_identity[identity] = module

        selected: list[tuple[str, nn.Module]] = []
        records: dict[str, ModuleRecord] = {}
        for identity, aliases in aliases_by_identity.items():
            canonical_name = aliases[0]
            module = modules_by_identity[identity]
            if not self._selector(canonical_name, module):
                continue

            selected.append((canonical_name, module))
            parameters = tuple(module.parameters(recurse=False))
            records[canonical_name] = ModuleRecord(
                type=type(module).__qualname__,
                aliases=tuple(aliases),
                parameter_count=sum(parameter.numel() for parameter in parameters),
                trainable_parameter_count=sum(
                    parameter.numel() for parameter in parameters if parameter.requires_grad
                ),
            )
        return selected, records

    def _sampling_record(self) -> SamplingRecord:
        """Describe built-in samplers and gracefully identify custom policies."""
        if isinstance(self._sampler, DescribedSamplingPolicy):
            return SamplingRecord(
                type=self._sampler.sampling_type(),
                settings=dict(self._sampler.sampling_settings()),
            )
        return SamplingRecord(type=type(self._sampler).__qualname__, settings={})

    def _reducer_records(self, reducers: Sequence[object]) -> tuple[ReducerRecord, ...]:
        """Describe built-in reducers and identify opaque custom callables by type."""
        records: list[ReducerRecord] = []
        for reducer in reducers:
            if isinstance(reducer, DescribedReducer):
                records.append(
                    ReducerRecord(
                        type=reducer.reducer_type(),
                        settings=dict(reducer.reducer_settings()),
                    )
                )
                continue
            callable_name = getattr(reducer, "__qualname__", type(reducer).__qualname__)
            records.append(ReducerRecord(type=callable_name, settings={}))
        return tuple(records)

    def _start_root_forward(self) -> None:
        """Choose sampling once per root forward and push its context-local state."""
        forward_index = self._next_forward_index()
        event = SamplingEvent(
            forward_index=forward_index,
            monotonic_time=self._monotonic_clock(),
        )
        try:
            should_sample = self._sampler.should_sample(event)
        except Exception as error:
            self._handle_error(
                error, builder=None, module_name=Absent("root sampling failed"), probe="sampling"
            )
            should_sample = False

        context: object = _NOT_COLLECTING
        if should_sample:
            context = _SnapshotBuilder(
                snapshot_id=self._next_snapshot_id(),
                forward_index=forward_index,
                timestamp=self._wall_clock(),
            )

        stack = self._contexts.get()
        self._contexts.set((*stack, context))

    def _finish_root_forward(self) -> None:
        """Close the current root-forward context and persist its forward snapshot."""
        stack = self._contexts.get()
        if not stack:
            return

        context = stack[-1]
        self._contexts.set(stack[:-1])
        if not isinstance(context, _SnapshotBuilder):
            return

        targets = context.take_gradient_targets()
        if targets:
            self._register_gradient_hook(context, targets)
        self._write_snapshot(context)

    def _collect_module_output(self, module_name: str, output: object) -> None:
        """Reduce one selected output only while its root context is sampled."""
        stack = self._contexts.get()
        if not stack:
            return
        context = stack[-1]
        if not isinstance(context, _SnapshotBuilder):
            return

        started_at = self._performance_clock()
        call = context.add_module_call(module_name)
        try:
            for leaf in iter_tensor_leaves(output, "output"):
                try:
                    record = self._tensor_record(
                        leaf.tensor,
                        snapshot_id=context.snapshot_id,
                    )
                    context.add_output(module_name, call, leaf.path, leaf.tensor, record)
                except Exception as error:
                    self._handle_error(
                        error,
                        builder=context,
                        module_name=module_name,
                        probe=leaf.path,
                    )
        finally:
            context.add_collection_duration(self._performance_clock() - started_at)

    def _register_gradient_hook(
        self,
        builder: _SnapshotBuilder,
        targets: tuple[_GradientTarget, ...],
    ) -> None:
        """Bind compact gradient collection to the exact sampled autograd graph."""
        fired_lock = threading.Lock()
        fired = False
        handle: RemovableHandle
        bindings = tuple(target.bindings for target in targets)

        def collect_gradients(gradients: Sequence[torch.Tensor | None]) -> None:
            """Record the first backward's available output gradients and rewrite the snapshot."""
            nonlocal fired
            with fired_lock:
                if fired:
                    return
                fired = True

            started_at = self._performance_clock()
            try:
                for target_bindings, gradient in zip(bindings, gradients, strict=True):
                    if gradient is None:
                        continue
                    for binding in target_bindings:
                        try:
                            builder.add_output_gradient(
                                binding,
                                self._tensor_record(
                                    gradient,
                                    snapshot_id=builder.snapshot_id,
                                ),
                            )
                        except Exception as error:
                            self._handle_error(
                                error,
                                builder=builder,
                                module_name=binding.module_name,
                                probe=binding.path,
                            )
                builder.mark_backward_observed()
            finally:
                builder.add_collection_duration(self._performance_clock() - started_at)
                self._discard_gradient_handle(handle)

            self._write_snapshot(builder)

        tensors = tuple(target.tensor for target in targets)
        try:
            # A graph-local hook is the correlation token between one sampled forward and its
            # backward; a module-global backward hook cannot encode that ownership.
            handle = torch.autograd.graph.register_multi_grad_hook(tensors, collect_gradients)
        except Exception as error:
            self._handle_error(
                error,
                builder=builder,
                module_name=Absent("gradient hook registration is run-level"),
                probe="gradient_hook_registration",
            )
            return

        with self._gradient_handles_lock:
            self._gradient_hook_handles.add(handle)

    def _discard_gradient_handle(self, handle: RemovableHandle) -> None:
        """Detach and forget a graph callback after its one supported backward."""
        handle.remove()
        with self._gradient_handles_lock:
            self._gradient_hook_handles.discard(handle)

    def _tensor_record(self, tensor: torch.Tensor, *, snapshot_id: int) -> TensorRecord:
        """Reduce a tensor into metadata and compact CPU-native diagnostics."""
        reduction = reduce_tensor(tensor, self._reducers)
        histogram_reduction = reduce_histograms(
            tensor,
            self._histogram_reducers,
            snapshot_id=snapshot_id,
        )
        dtype = str(tensor.dtype).removeprefix("torch.")
        return TensorRecord(
            shape=tuple(tensor.shape),
            dtype=dtype,
            device=str(tensor.device),
            numel=tensor.numel(),
            stats=reduction.stats,
            unavailable_stats=reduction.unavailable_stats,
            histograms=histogram_reduction.histograms,
            unavailable_histograms=histogram_reduction.unavailable_histograms,
        )

    def _next_forward_index(self) -> int:
        """Allocate a thread-safe index for every root forward, sampled or not."""
        with self._id_lock:
            forward_index = self._forward_index
            self._forward_index += 1
        return forward_index

    def _next_snapshot_id(self) -> int:
        """Allocate a contiguous thread-safe identifier for sampled forwards only."""
        with self._id_lock:
            snapshot_id = self._snapshot_id
            self._snapshot_id += 1
        return snapshot_id

    def _write_snapshot(self, builder: _SnapshotBuilder) -> None:
        """Serialize one consistent builder view under the sink lock."""
        try:
            with self._sink_lock:
                self._sink.write_snapshot(builder.to_record())
        except Exception as error:
            self._handle_error(
                error,
                builder=builder,
                module_name=Absent("snapshot write is run-level"),
                probe="sink",
            )

    def _handle_error(
        self,
        error: Exception,
        *,
        builder: _SnapshotBuilder | None,
        module_name: str | Absent,
        probe: str,
    ) -> None:
        """Persist collection failures when possible and apply the configured policy."""
        if builder is not None:
            builder.add_error(
                ErrorRecord(
                    timestamp=self._wall_clock(),
                    module=module_name,
                    probe=probe,
                    exception_type=type(error).__qualname__,
                    message=str(error),
                )
            )

        if self._error_policy is ErrorPolicy.RAISE:
            raise error
        if self._error_policy is ErrorPolicy.WARN:
            warnings.warn(
                f"TorchInstruments {probe} failed: {type(error).__qualname__}: {error}",
                RuntimeWarning,
                stacklevel=3,
            )
