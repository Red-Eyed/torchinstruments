"""Bind gradients before in-place mutation and publish at backward completion."""

from collections.abc import Callable
from threading import Lock

import torch
from torch.utils.hooks import RemovableHandle


class FirstBackward:
    """Own one sample's tensor hooks without retaining source tensors."""

    def __init__(self, complete: Callable[[], None]) -> None:
        """Bind completion independently of tensor identity and measurement policy."""
        self._complete = complete
        self._handles: list[RemovableHandle] = []
        self._queued = False
        self._closed = False
        self._lock = Lock()

    def attach(self, tensor: torch.Tensor, consume: Callable[[torch.Tensor], None]) -> None:
        """Register on the current gradient edge before downstream mutation can rebase it."""
        if not tensor.requires_grad:
            return

        def collect(gradient: torch.Tensor) -> None:
            """Consume a gradient without replacing it or extending its lifetime."""
            with self._lock:
                if self._closed:
                    return
                if not self._queued:
                    self._queued = True
                    # PyTorch exposes completion only through the engine callback queue.
                    # Keep this private dependency here and cover backward/autograd.grad in tests.
                    torch.autograd.Variable._execution_engine.queue_callback(self._finish)
            consume(gradient)

        self._handles.append(tensor.register_hook(collect))

    def remove(self) -> None:
        """Disable queued completion and detach every still-live tensor hook."""
        with self._lock:
            self._closed = True
            handles, self._handles = self._handles, []
        for handle in handles:
            handle.remove()

    def _finish(self) -> None:
        """Publish only once, after the engine has visited all used tensor edges."""
        with self._lock:
            if self._closed:
                return
        self.remove()
        self._complete()
