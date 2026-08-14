"""Tests for safe and deterministic nested tensor traversal."""

from __future__ import annotations

import torch

from torchinstruments.pytree import iter_tensor_leaves


def test_tensor_tree_ignores_unsupported_values() -> None:
    """Yield supported tensor leaves without failing on arbitrary Python values."""
    value = {
        "logits": torch.ones(2),
        "metadata": object(),
        "nested": [None, (torch.zeros(1), "ignored")],
    }

    leaves = list(iter_tensor_leaves(value, "output"))

    assert [leaf.path for leaf in leaves] == ["output.logits", "output.nested.1.0"]


def test_tensor_tree_sorts_mapping_paths_stably() -> None:
    """Produce deterministic paths independent of mapping insertion order."""
    value = {"z": torch.ones(1), "a": torch.ones(1)}

    paths = [leaf.path for leaf in iter_tensor_leaves(value, "output")]

    assert paths == ["output.a", "output.z"]
