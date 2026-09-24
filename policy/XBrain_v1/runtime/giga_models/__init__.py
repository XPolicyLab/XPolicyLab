"""Minimal XBrain-v1 runtime export surface.

The original package-level wildcard imports pull unrelated diffusion and vision
models into a policy process.  The inference distribution exports only the
GigaBrain policy and pipeline used by XBrain-v1 checkpoints.
"""

__version__ = "0.1.0+xbrainv1"

from .models.vla.giga_brain_0 import GigaBrain0Policy
from .pipelines.vla.giga_brain_0 import GigaBrain0Pipeline

__all__ = ["GigaBrain0Pipeline", "GigaBrain0Policy"]
