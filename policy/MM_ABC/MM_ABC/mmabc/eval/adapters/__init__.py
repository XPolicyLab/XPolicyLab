"""Benchmark adapters. Importing the package registers the built-in ones."""

from mmabc.eval.adapters.base import BenchmarkAdapter, available, build_adapter, register
from mmabc.eval.adapters.mobile import MobileAdapter

__all__ = [
    "BenchmarkAdapter",
    "MobileAdapter",
    "available",
    "build_adapter",
    "register",
]
