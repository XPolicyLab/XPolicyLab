"""Instance-local module preparation for Falcon Motubrain models."""

from __future__ import annotations

import inspect

import torch.nn as nn


def _strip_profiling_wrappers(module: nn.Module) -> None:
    """Keep Falcon's NVTX decorators outside compiled inference graphs."""
    for child in module.modules():
        forward = getattr(type(child), "forward", None)
        if forward is not None and hasattr(forward, "__wrapped__"):
            setattr(type(child), "forward", inspect.unwrap(forward))


def prepare_motubrain_model(model: nn.Module) -> None:
    """Prepare a constructed Falcon Motubrain model for inference capture.

    Falcon's model code already imports the portable operators in
    ``soma.inference.models.operators`` directly, so nothing has to be swapped
    in after construction.  What is left is removing the NVTX ``forward``
    decorators, which ``torch.compile`` cannot trace through.
    """
    _strip_profiling_wrappers(model)
