"""Alias: `FastWAMProcessor` is FlexPi's `FlexPiProcessor` (renamed, same API).

Verified: the constructor accepts every key a legacy processor block passes;
FlexPi adds three optional params with defaults.
"""

from flexpi.datasets.lerobot.processors.flexpi_processor import FlexPiProcessor

FastWAMProcessor = FlexPiProcessor

__all__ = ["FastWAMProcessor", "FlexPiProcessor"]
