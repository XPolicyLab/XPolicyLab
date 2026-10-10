"""Canonical action/state space: layout, increment transforms, normalisation."""

from mmabc.canonical.layout import CanonicalLayout, Head, Segment, load_layout
from mmabc.canonical.normalize import NormStats, Normalizer, RunningStats, identity_stats
from mmabc.canonical.transforms import EmbodimentSpec, TargetBuilder

__all__ = [
    "CanonicalLayout",
    "EmbodimentSpec",
    "Head",
    "NormStats",
    "Normalizer",
    "RunningStats",
    "Segment",
    "TargetBuilder",
    "identity_stats",
    "load_layout",
]
