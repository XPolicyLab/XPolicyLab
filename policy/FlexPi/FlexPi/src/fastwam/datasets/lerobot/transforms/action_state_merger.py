"""Alias — same class names and signatures as FlexPi's."""

from flexpi.datasets.lerobot.transforms.action_state_merger import (  # noqa: F401
    ConcatLeftAlign,
    ScatterToChannels,
)

__all__ = ["ConcatLeftAlign", "ScatterToChannels"]
