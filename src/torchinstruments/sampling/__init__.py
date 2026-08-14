"""Built-in sampling policies and their structural protocol."""

from torchinstruments.sampling.base import SamplingEvent, SamplingPolicy
from torchinstruments.sampling.periodic import AlwaysSampler, EveryNForwardsSampler
from torchinstruments.sampling.timed import TimedSampler

__all__ = [
    "AlwaysSampler",
    "EveryNForwardsSampler",
    "SamplingEvent",
    "SamplingPolicy",
    "TimedSampler",
]
