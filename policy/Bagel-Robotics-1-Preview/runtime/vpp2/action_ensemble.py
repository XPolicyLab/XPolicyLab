"""Pose-aware overlap blending for isolated EE16 RoboDojo evaluations."""

from __future__ import annotations

import numpy as np


def _unit_quaternion(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError("Invalid EE quaternion")
    return q / norm


def _slerp(q_old: np.ndarray, q_new: np.ndarray, old_weight: float) -> np.ndarray:
    old = _unit_quaternion(q_old)
    new = _unit_quaternion(q_new)
    dot = float(np.dot(old, new))
    if dot < 0:
        old = -old
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        return _unit_quaternion(old_weight * old + (1 - old_weight) * new)
    angle = np.arccos(dot)
    return (
        np.sin(old_weight * angle) / np.sin(angle) * old
        + np.sin((1 - old_weight) * angle) / np.sin(angle) * new
    )


def _rotation_vector(target: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Shortest rotation taking the current EE orientation to the target."""
    t = _unit_quaternion(target)
    r = _unit_quaternion(reference)
    w = t[0] * r[0] + np.dot(t[1:], r[1:])
    v = -t[0] * r[1:] + r[0] * t[1:] - np.cross(t[1:], r[1:])
    if w < 0:
        w, v = -w, -v
    length = np.linalg.norm(v)
    if length < 1e-8:
        return 2 * v
    return (2 * np.arctan2(length, w) / length) * v


def _motion_feature(action: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Put absolute EE16 commands into a relative pose space for cosine similarity."""
    features = []
    for start in (0, 8):
        features.append((action[start:start + 3] - state[start:start + 3]) / 0.05)
        features.append(
            _rotation_vector(action[start + 3:start + 7], state[start + 3:start + 7]) / 0.5
        )
        features.append(np.array([action[start + 7] - state[start + 7]]))
    return np.concatenate(features)


def _cogact_old_weight(new: np.ndarray, old: np.ndarray, state: np.ndarray,
                       alpha: float) -> float:
    current = _motion_feature(new, state)
    previous = _motion_feature(old, state)
    norms = np.linalg.norm(current) * np.linalg.norm(previous)
    similarity = float(np.dot(current, previous) / norms) if norms > 1e-10 else 0.0
    similarity = float(np.clip(similarity, -1, 1))
    # CogACT Eq. 5, normalized over current (self-similarity 1) and one past plan.
    return float(1 / (1 + np.exp(alpha * (1 - similarity))))


def blend_overlap(new_chunk: np.ndarray, previous_chunk: np.ndarray,
                  state: np.ndarray, *, offset: int, execute: int,
                  mode: str, alpha: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Blend equal absolute future steps; preserve the new plan's gripper timing.

    The previous 32-step plan starts ``offset`` native steps before the new one.
    Only the executed prefix is modified; the full new plan remains the next
    overlap reference. Returns that executed chunk and its per-step old weights.
    """
    new = np.asarray(new_chunk, dtype=np.float64)
    old = np.asarray(previous_chunk, dtype=np.float64)
    state = np.asarray(state, dtype=np.float64)
    if (new.shape != (32, 16) or old.shape != (32, 16) or state.shape != (16,)
            or execute != 16 or offset != 16 or mode not in {"fixed90", "cogact"}
            or not np.isfinite(new).all() or not np.isfinite(old).all()
            or not np.isfinite(state).all() or not np.isfinite(alpha) or alpha < 0):
        raise ValueError("Expected finite EE16 predict-32/execute-16 overlap")
    result = new[:execute].copy()
    weights = np.empty(execute, dtype=np.float64)
    for index in range(execute):
        previous = old[offset + index]
        current = new[index]
        weight = (0.1 if mode == "fixed90" else
                  _cogact_old_weight(current, previous, state, alpha))
        weights[index] = weight
        for start in (0, 8):
            result[index, start:start + 3] = (
                (1 - weight) * current[start:start + 3]
                + weight * previous[start:start + 3]
            )
            result[index, start + 3:start + 7] = _slerp(
                previous[start + 3:start + 7], current[start + 3:start + 7], weight
            )
            # Gripper transitions are categorical; use the fresh observation's command.
            result[index, start + 7] = current[start + 7]
    return result.astype(np.float32), weights
