"""Wrap map-style robot data with the ME_U0 model batch contract."""

from .map_style_dataset import build_world_unified_map_dataset


def build_world_unified_dataset(dataset, *, max_action_dim=26, max_state_dim=26):
    if max_action_dim != 26 or max_state_dim != 26:
        raise ValueError("map-style post-training uses 26-D action/state heads")
    return build_world_unified_map_dataset(dataset)
