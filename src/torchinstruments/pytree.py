"""Traverse supported Python containers without assuming a single tensor output."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TensorLeaf:
    """Bind a tensor to its stable path inside a nested hook value."""

    path: str
    tensor: torch.Tensor


def iter_tensor_leaves(value: object, prefix: str) -> Iterator[TensorLeaf]:
    """Yield tensor leaves from nested tuples, lists, and mappings.

    Unsupported values and mapping keys are ignored so unusual model outputs cannot break a
    valid training run.
    """
    if isinstance(value, torch.Tensor):
        yield TensorLeaf(path=prefix, tensor=value)
        return

    if isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from iter_tensor_leaves(item, f"{prefix}.{index}")
        return

    if isinstance(value, Mapping):
        for key in sorted(value, key=_mapping_key_order):
            if isinstance(key, (str, int)):
                yield from iter_tensor_leaves(value[key], f"{prefix}.{key}")


def _mapping_key_order(key: object) -> tuple[str, str]:
    """Provide deterministic ordering across heterogeneous mapping-key types."""
    return type(key).__qualname__, repr(key)
