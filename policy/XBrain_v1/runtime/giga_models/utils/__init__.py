"""Small utility surface required by the inference-only XBrain runtime."""

from .action_horizon import (
    downsample_flow_action_tensors,
    flow_action_horizon_indices,
    resolve_flow_action_steps,
)

__all__ = [
    "downsample_flow_action_tensors",
    "flow_action_horizon_indices",
    "resolve_flow_action_steps",
]
