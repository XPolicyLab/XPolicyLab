"""PiperX joint-to-endpose FK adapter for XBrain-v1.

The validated NumPy FK implementation is vendored in this same XBrain_v1
directory. It depends only on NumPy and requires no external SDK checkout.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from XPolicyLab.policy.XBrain_v1.piperx_fk import dual_arm_fk


FK_PATH = Path(__file__).with_name("piperx_fk.py")


def joints14_to_endpose16(joints14):
    """Convert ``(...,14)`` joint actions to ``(...,16)`` endpose actions.

    The FK module returns [left xyz, left WXYZ quat, left gripper,
    right xyz, right WXYZ quat, right gripper]. Grippers are passed through.
    """
    values = np.asarray(joints14, dtype=np.float64)
    if values.ndim < 1 or values.shape[-1] != 14:
        raise ValueError(f"PiperX joint action must have shape (...,14), got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("PiperX joint action contains NaN or infinity")
    endpose = np.asarray(dual_arm_fk(values, quaternion_order="wxyz"), dtype=np.float32)
    if endpose.shape != values.shape[:-1] + (16,) or not np.isfinite(endpose).all():
        raise ValueError(f"PiperX FK returned invalid shape/values: {endpose.shape}")
    return endpose


def fk_source_path():
    """Return the vendored FK source path for startup diagnostics."""
    return str(FK_PATH)


def adjust_piperx_endpose_z(endpose16):
    """Lower in-range left/right Z values while preserving a 0.13 m floor.

    The comparison is independent for every action row and each arm. Values
    outside the inclusive [0.13, 0.17] interval are left unchanged.
    """
    result = np.asarray(endpose16, dtype=np.float32).copy()
    if result.ndim < 1 or result.shape[-1] != 16:
        raise ValueError(f"endpose16 must have shape (...,16), got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError("endpose16 contains NaN or infinity")
    for column in (2, 10):
        z = result[..., column]
        in_range = (z >= 0.13) & (z <= 0.17)
        result[..., column] = np.where(in_range, np.maximum(z - 0.02, 0.13), z)
    return result
