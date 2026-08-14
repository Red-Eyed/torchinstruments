"""Common framework-agnostic module-selection predicates."""

from __future__ import annotations

from torch import nn

from torchinstruments.selectors.base import ModuleSelector


def leaf_modules() -> ModuleSelector:
    """Select modules without registered child modules."""

    def is_leaf(name: str, module: nn.Module) -> bool:
        """Return whether a module is a leaf; its canonical name is irrelevant."""
        del name
        return next(module.children(), None) is None

    return is_leaf
