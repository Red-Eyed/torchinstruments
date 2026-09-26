"""Capture module invocations through reversible forward interception."""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass
from types import MethodType
from typing import TYPE_CHECKING, Protocol

import torch
from torch import nn

from torchinstruments.records import ExecutionContext, ModuleMode

if TYPE_CHECKING:
    from collections.abc import Sequence

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


@dataclass(frozen=True)
class _ForwardPatch:
    """Remember enough instance state to restore one wrapped forward safely."""

    module: nn.Module
    installed_forward: object
    previous_instance_forward: object


class ForwardCallCapture:
    """Capture normal and direct calls by intercepting each selected forward once."""

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
