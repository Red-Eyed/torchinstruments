"""Rank discovery and output ownership for optional distributed execution."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import StrEnum

import torch


class RankPolicy(StrEnum):
    """Select which distributed ranks attach model instrumentation."""

    RANK0 = "rank0"
    ALL = "all"


@dataclass(frozen=True)
class RankInfo:
    """Identify one process without requiring distributed initialization."""

    rank: int
    world_size: int

    def __post_init__(self) -> None:
        """Reject identities that cannot describe a valid process group."""
        if self.world_size <= 0:
            raise ValueError("world_size must be greater than zero")
        if self.rank < 0 or self.rank >= self.world_size:
            raise ValueError("rank must be in [0, world_size)")

    @property
    def is_distributed(self) -> bool:
        """Report whether this process belongs to a multi-process run."""
        return self.world_size > 1


def detect_rank() -> RankInfo:
    """Read initialized PyTorch rank state or fall back to torchrun environment variables."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return RankInfo(
            rank=torch.distributed.get_rank(),
            world_size=torch.distributed.get_world_size(),
        )
    return RankInfo(
        rank=_environment_integer("RANK", default=0),
        world_size=_environment_integer("WORLD_SIZE", default=1),
    )


def parse_rank_policy(value: RankPolicy | str) -> RankPolicy:
    """Normalize a public rank policy or reject unsupported values clearly."""
    try:
        return RankPolicy(value)
    except ValueError as error:
        choices = ", ".join(policy.value for policy in RankPolicy)
        raise ValueError(f"rank_policy must be one of: {choices}") from error


def rank_is_enabled(policy: RankPolicy, rank: RankInfo) -> bool:
    """Return whether this process should pay instrumentation cost."""
    return policy is RankPolicy.ALL or rank.rank == 0


def _environment_integer(name: str, *, default: int) -> int:
    """Parse one non-negative distributed environment integer at the process boundary."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from error
    if value < 0:
        raise ValueError(f"{name} must not be negative")
    return value
