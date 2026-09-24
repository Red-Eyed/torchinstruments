"""Capture module invocations through PyTorch hooks or forward wrappers."""

from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from types import MethodType
from typing import Protocol

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle

from torchinstruments.records import ExecutionContext, ModuleMode

_MISSING = object()


@dataclass(frozen=True)
class CaptureCallbacks:
    """Expose the observer lifecycle operations needed by capture strategies."""

    start_root: Callable[[], None]
    finish_root: Callable[[], None]
    observe_output: Callable[[str, ExecutionContext, object], None]


class CallCapture(Protocol):
    """Define how an observer discovers root and selected-module invocations."""

    def capture_type(self) -> str:
        """Return a stable identifier suitable for run metadata."""
        ...

    def attach(
        self,
        model: nn.Module,
        selected_modules: Sequence[tuple[str, nn.Module]],
        callbacks: CaptureCallbacks,
    ) -> None:
        """Attach capture behavior to the root and selected modules."""
        ...

    def remove(self) -> None:
        """Remove every capture mutation owned by this strategy."""
        ...


class HookCallCapture:
    """Capture normal ``module(...)`` dispatch with native PyTorch hooks."""

    def __init__(self) -> None:
        """Initialize an unattached collection of removable hook handles."""
        self._handles: list[RemovableHandle] = []

    def capture_type(self) -> str:
        """Identify native hook dispatch in serialized run metadata."""
        return "pytorch_hooks"

    def attach(
        self,
        model: nn.Module,
        selected_modules: Sequence[tuple[str, nn.Module]],
        callbacks: CaptureCallbacks,
    ) -> None:
        """Register selected output hooks and one root lifecycle hook pair."""
        if self._handles:
            raise RuntimeError("hook capture is already attached")

        try:
            for module_name, module in selected_modules:
                self._attach_module(module_name, module, callbacks.observe_output)
            self._handles.append(model.register_forward_pre_hook(self._root_pre_hook(callbacks)))
            self._handles.append(
                model.register_forward_hook(self._root_post_hook(callbacks), always_call=True)
            )
        except BaseException:
            self.remove()
            raise

    def remove(self) -> None:
        """Remove all native hooks registered by this strategy."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _attach_module(
        self,
        name: str,
        module: nn.Module,
        observe: Callable[[str, ExecutionContext, object], None],
    ) -> None:
        """Bind invocation context before execution, including nested calls and failures."""
        contexts: ContextVar[tuple[ExecutionContext, ...]] = ContextVar(
            f"module_context_{id(module)}", default=()
        )

        def start(current: nn.Module, _inputs: tuple[object, ...]) -> None:
            """Snapshot mode before a module can change it inside forward."""
            contexts.set((*contexts.get(), _execution_context(current)))

        def finish(_module: nn.Module, _inputs: tuple[object, ...], output: object) -> None:
            """Restore the surrounding context even after a failed forward."""
            stack = contexts.get()
            if not stack:
                return
            contexts.set(stack[:-1])
            observe(name, stack[-1], output)

        self._handles.append(module.register_forward_pre_hook(start))
        self._handles.append(module.register_forward_hook(finish, always_call=True))

    def _root_pre_hook(
        self,
        callbacks: CaptureCallbacks,
    ) -> Callable[[nn.Module, tuple[object, ...]], None]:
        """Adapt root-context creation to PyTorch's pre-hook signature."""

        def start(_module: nn.Module, _inputs: tuple[object, ...]) -> None:
            """Start one root-forward context before model execution."""
            callbacks.start_root()

        return start

    def _root_post_hook(
        self,
        callbacks: CaptureCallbacks,
    ) -> Callable[[nn.Module, tuple[object, ...], object], None]:
        """Adapt root-context completion to PyTorch's forward-hook signature."""

        def finish(_module: nn.Module, _inputs: tuple[object, ...], _output: object) -> None:
            """Finish one root-forward context after model execution or failure."""
            callbacks.finish_root()

        return finish


@dataclass(frozen=True)
class _ForwardPatch:
    """Remember enough instance state to restore one wrapped forward safely."""

    module: nn.Module
    installed_forward: object
    previous_instance_forward: object


class ForwardCallCapture:
    """Capture both ``module(...)`` and literal ``module.forward(...)`` calls.

    This strategy installs instance-level forward wrappers once. It is intentionally opt-in
    because replacing a Python method is more invasive than registering native PyTorch hooks.
    """

    def __init__(self) -> None:
        """Initialize an unattached collection of reversible forward patches."""
        self._patches: list[_ForwardPatch] = []

    def capture_type(self) -> str:
        """Identify direct-forward-compatible capture in serialized run metadata."""
        return "forward_wrappers"

    def attach(
        self,
        model: nn.Module,
        selected_modules: Sequence[tuple[str, nn.Module]],
        callbacks: CaptureCallbacks,
    ) -> None:
        """Wrap the root and each unique selected child exactly once."""
        if self._patches:
            raise RuntimeError("forward capture is already attached")

        root_name = self._selected_root_name(model, selected_modules)
        try:
            for module_name, module in selected_modules:
                if module is model:
                    continue
                self._wrap_selected_module(module, module_name, callbacks.observe_output)
            self._wrap_root(model, root_name, callbacks)
        except BaseException:
            self.remove()
            raise

    def remove(self) -> None:
        """Restore prior forward attributes without overwriting later caller changes."""
        for patch in reversed(self._patches):
            current_forward = patch.module.__dict__.get("forward", _MISSING)
            if current_forward is not patch.installed_forward:
                continue
            if patch.previous_instance_forward is _MISSING:
                del patch.module.__dict__["forward"]
            else:
                patch.module.__dict__["forward"] = patch.previous_instance_forward
        self._patches.clear()

    def _selected_root_name(
        self,
        model: nn.Module,
        selected_modules: Sequence[tuple[str, nn.Module]],
    ) -> str | object:
        """Return the root's selected name or an internal missing marker."""
        for module_name, module in selected_modules:
            if module is model:
                return module_name
        return _MISSING

    def _wrap_selected_module(
        self,
        module: nn.Module,
        module_name: str,
        observe_output: Callable[[str, ExecutionContext, object], None],
    ) -> None:
        """Wrap one selected child and report its successful output."""
        original_forward = module.forward

        @functools.wraps(original_forward)
        def wrapped(_module: nn.Module, *args: object, **kwargs: object) -> object:
            """Execute the original child forward and report its output once."""
            context = _execution_context(_module)
            output = original_forward(*args, **kwargs)
            observe_output(module_name, context, output)
            return output

        self._install_patch(module, MethodType(wrapped, module))

    def _wrap_root(
        self,
        model: nn.Module,
        selected_root_name: str | object,
        callbacks: CaptureCallbacks,
    ) -> None:
        """Wrap root execution so every direct or dispatched call owns one context."""
        original_forward = model.forward

        @functools.wraps(original_forward)
        def wrapped(_module: nn.Module, *args: object, **kwargs: object) -> object:
            """Run one root forward inside an observer lifecycle context."""
            context = _execution_context(_module)
            callbacks.start_root()
            try:
                output = original_forward(*args, **kwargs)
                if isinstance(selected_root_name, str):
                    callbacks.observe_output(selected_root_name, context, output)
                return output
            finally:
                callbacks.finish_root()

        self._install_patch(model, MethodType(wrapped, model))

    def _install_patch(self, module: nn.Module, installed_forward: object) -> None:
        """Install one wrapper and record the exact instance attribute it replaced."""
        previous_instance_forward = module.__dict__.get("forward", _MISSING)
        module.__dict__["forward"] = installed_forward
        self._patches.append(
            _ForwardPatch(
                module=module,
                installed_forward=installed_forward,
                previous_instance_forward=previous_instance_forward,
            )
        )


def _execution_context(module: nn.Module) -> ExecutionContext:
    """Snapshot module mode separately from autograd recording."""
    return ExecutionContext(
        ModuleMode.TRAIN if module.training else ModuleMode.EVAL, torch.is_grad_enabled()
    )
