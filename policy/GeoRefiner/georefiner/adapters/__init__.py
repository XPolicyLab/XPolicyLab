"""Adapter interfaces for benchmark-specific conversion.

Full benchmark adapters are intentionally deferred. Current exports only define
converter contracts for future native/canonical action and state conversion.
"""

from georefiner.adapters.benchmark import (
    ActionAdapter,
    ActionAdapterOutput,
    BenchmarkAdapter,
    BenchmarkAdapterOutput,
    CalibrationMLP,
    StateAdapter,
    StateAdapterOutput,
)
from georefiner.adapters.converters import (
    ActionConverterBase,
    IdentityActionConverter,
    IdentityStateConverter,
    StateConverterBase,
)
from georefiner.adapters.views import ViewAdapter, ViewAdapterOutput

__all__ = [
    "ActionAdapter",
    "ActionAdapterOutput",
    "ActionConverterBase",
    "BenchmarkAdapter",
    "BenchmarkAdapterOutput",
    "CalibrationMLP",
    "IdentityActionConverter",
    "IdentityStateConverter",
    "StateAdapter",
    "StateAdapterOutput",
    "StateConverterBase",
    "ViewAdapter",
    "ViewAdapterOutput",
]
