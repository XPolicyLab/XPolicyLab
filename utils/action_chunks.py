"""Structural checks for policy responses that are ready to execute."""

import numpy as np


def validate_action_chunk(actions):
    """Accept a non-empty sequence of control steps."""
    if isinstance(actions, np.ndarray):
        if actions.ndim != 2 or not all(actions.shape) or actions.dtype.kind not in "iuf":
            raise ValueError("Expected a non-empty numeric (steps, action_dim) action chunk")
        return
    if not isinstance(actions, (list, tuple)) or not actions:
        raise ValueError("Expected a non-empty action chunk")
    for step in actions:
        if isinstance(step, dict) and step:
            continue
        if isinstance(step, (list, tuple, np.ndarray)):
            vector = np.asarray(step)
            if vector.ndim == 1 and vector.size and vector.dtype.kind in "iuf":
                continue
        raise ValueError("Action chunk contains an invalid control step")


def validate_action_batch(chunks, num_envs):
    """Validate all chunk lengths before any environment advances. Return the horizon."""
    if isinstance(chunks, np.ndarray):
        if (chunks.ndim != 3 or chunks.shape[0] != num_envs
                or not all(chunks.shape) or chunks.dtype.kind not in "iuf"):
            raise ValueError("Expected a numeric (active_envs, steps, action_dim) action batch")
        return chunks.shape[1]
    if not isinstance(chunks, (list, tuple)) or not chunks or len(chunks) != num_envs:
        raise ValueError(f"Expected one action chunk per active environment ({num_envs})")
    for chunk in chunks:
        validate_action_chunk(chunk)
    horizon = len(chunks[0])
    if any(len(chunk) != horizon for chunk in chunks):
        raise ValueError("Batched action chunks must have the same length")
    return horizon
