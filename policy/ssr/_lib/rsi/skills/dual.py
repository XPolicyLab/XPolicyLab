from __future__ import annotations

import numpy as np

GRIP_SCALE = -0.01, 0.044


def grip_for_gap(gap: float) -> float:
    q = gap / 2.0 - 0.0004
    return float(np.clip((q - GRIP_SCALE[0]) / (GRIP_SCALE[1] - GRIP_SCALE[0]), 0.0, 1.0))


def gap_for_grip(g: float) -> float:
    q = g * (GRIP_SCALE[1] - GRIP_SCALE[0]) + GRIP_SCALE[0]
    return max(0.0, 2.0 * (q + 0.0004))


def held_width(g_obs: float, commanded: float = 0.0) -> float:
    rng = GRIP_SCALE[1] - GRIP_SCALE[0]
    if g_obs <= commanded + 0.05:
        return 0.0
    q = g_obs * rng + GRIP_SCALE[0] + 0.2 * rng
    return 2.0 * (q + 0.0004)
