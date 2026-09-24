"""Exact linear quantiles beyond the size supported by torch.quantile."""

import math

import torch


def exact_quartiles(values: torch.Tensor) -> dict[str, torch.Tensor]:
    """Select exact order statistics without sorting or sampling a large tensor."""
    result: dict[str, torch.Tensor] = {}
    for name, probability in (("p25", 0.25), ("median", 0.5), ("p75", 0.75)):
        rank = (values.numel() - 1) * probability
        lower = math.floor(rank)
        upper = math.ceil(rank)
        left = values.kthvalue(lower + 1).values
        right = left if lower == upper else values.kthvalue(upper + 1).values
        result[name] = torch.lerp(left, right, rank - lower)
    return result
