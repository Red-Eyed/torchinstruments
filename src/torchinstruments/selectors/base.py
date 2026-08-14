"""Structural contract for selecting modules during observer attachment."""

from __future__ import annotations

from typing import Protocol

from torch import nn


class ModuleSelector(Protocol):
    """Choose whether one uniquely named module should receive a hook."""

    def __call__(self, name: str, module: nn.Module) -> bool:
        """Return whether ``module`` should be observed under its canonical name."""
        ...
