"""Validate IK targets against measured joints before control interpolation."""

import math

import numpy as np


class IKGuard:
    """Reject an entire arm target, rather than clipping individual joints."""

    def __init__(self, max_joint_delta_rad=0.5):
        self.max_joint_delta_rad = float(max_joint_delta_rad)
        if not math.isfinite(self.max_joint_delta_rad) or self.max_joint_delta_rad <= 0:
            raise ValueError("ik_max_joint_delta_rad must be finite and strictly positive")

    def filter_result(self, current_position, ik_result):
        """Return (safe position, rejection reason); None means accepted.

        Differences are intentionally not wrapped modulo 2*pi: the downstream
        controller interpolates raw joint coordinates and would execute a full
        turn even when the endpoints describe the same orientation.
        """
        current = np.asarray(current_position, dtype=np.float64)
        if current.ndim != 1 or current.size == 0 or not np.isfinite(current).all():
            raise ValueError("Cannot hold arm: measured joint positions are invalid")

        if not isinstance(ik_result, dict) or ik_result.get("status") != "Success":
            return current.copy(), "IK solver did not return Success"
        try:
            target = np.asarray(ik_result.get("joint_value"), dtype=np.float64)
        except (TypeError, ValueError, OverflowError):
            return current.copy(), "IK joint values are not numeric"
        if target.shape != current.shape:
            return current.copy(), f"IK joint shape {target.shape} != measured shape {current.shape}"
        if not np.isfinite(target).all():
            return current.copy(), "IK joint values contain NaN or Inf"

        # Subtraction can overflow for finite but corrupt extreme values.
        with np.errstate(over="ignore", invalid="ignore"):
            delta = np.abs(target - current)
        joint_idx = int(np.argmax(delta))
        if delta[joint_idx] > self.max_joint_delta_rad:
            return current.copy(), (
                f"joint[{joint_idx}] delta={delta[joint_idx]:.6g} rad "
                f"> {self.max_joint_delta_rad:.6g} rad "
                f"(current={current[joint_idx]:.6g}, target={target[joint_idx]:.6g})"
            )
        return target, None
