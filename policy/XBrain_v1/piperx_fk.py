"""Offline dual-arm Piper-X FK for RoboDojo joint14 data (NumPy only).

Input (..., 14): [left q1..q6 (rad), left gripper,
                 right q1..q6 (rad), right gripper].
Output (..., 16): [left xyz (m), left quaternion, left gripper,
                  right xyz (m), right quaternion, right gripper].
Each pose is in THAT ARM'S BASE frame; no dual-arm world extrinsic is applied.

The cap_pen_piper_x_lerobot_v21_0831 parquet values use WXYZ quaternions.
Its originally incorrect XYZW metadata labels were corrected to WXYZ on
2026-09-29. Default output is WXYZ to match the stored values and current
metadata. Request quaternion_order="xyzw" for standard SciPy/ROS
ordering, and reorder the recorded targets too when comparing them.

MDH values are copied verbatim from pyAgxArm/api/constants.py, piper_x,
validated 2026-09-29. The original SDK algorithm is preserved; this module
adds batching, two-arm slicing, and quaternion output without CAN imports.
No TCP offset is needed to match the specified dataset. Optional transforms
are for an explicitly defined physical tool, in each flange frame.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# Rows: d (m), a (m), alpha (rad), theta_offset (rad).
PIPERX_MDH = np.array([
    [0.123, 0.0, 0.0, 3.141592653589793],
    [0.0, 0.0, 1.5707963267948966, 0.13578661580515886],
    [0.0, 0.28502999999999995, 0.0, 2.8380798966679794],
    [0.0, 0.27364, 0.0, 0.08063421144213803],
    [0.0, 0.07465999999999999, -1.5707963267948966, 1.5707963267948966],
    [0.03526, 0.0, 1.5707963267948966, 0.0],
], dtype=np.float64)
PIPERX_MDH.setflags(write=False)


def _array(value, width, name):
    array = np.asarray(value, dtype=np.float64)
    if array.ndim < 1 or array.shape[-1] != width:
        raise ValueError(f"{name} must have shape (..., {width}), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _check_order(order):
    if order not in ("wxyz", "xyzw"):
        raise ValueError("quaternion_order must be 'wxyz' or 'xyzw'")


def piperx_flange_matrix(joint_radians):
    """Return (..., 4, 4) base-to-flange transforms from (..., 6) radians.

    Modified DH: Rx(alpha) Tx(a) Rz(q + theta_offset) Tz(d).
    Joint order/sign is the SDK get_joint_angles() convention. Gripper
    dimensions must be excluded. No unit conversion or clipping is done.
    """
    q = _array(joint_radians, 6, "joint_radians")
    transform = np.broadcast_to(np.eye(4), q.shape[:-1] + (4, 4)).copy()
    for i, (d, a, alpha, offset) in enumerate(PIPERX_MDH):
        theta = q[..., i] + offset
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)
        link = np.zeros_like(transform)
        link[..., 0, 0] = ct
        link[..., 0, 1] = -st
        link[..., 0, 3] = a
        link[..., 1, 0] = ca * st
        link[..., 1, 1] = ca * ct
        link[..., 1, 2] = -sa
        link[..., 1, 3] = -sa * d
        link[..., 2, 0] = sa * st
        link[..., 2, 1] = sa * ct
        link[..., 2, 2] = ca
        link[..., 2, 3] = ca * d
        link[..., 3, 3] = 1.0
        transform = transform @ link
    return transform


def _transform_to_pose7(transform, quaternion_order):
    """Use the SDK RPY branch to also match recorded quaternion signs.

    Quaternions q and -q always represent the same rotation. The chosen sign
    can jump at an Euler branch boundary; compare rotations or abs(dot),
    not quaternion components, when assessing physical orientation error.
    """
    r = transform[..., :3, :3]
    pitch = np.arcsin(np.clip(-r[..., 2, 0], -1.0, 1.0))
    singular = np.abs(np.cos(pitch)) < 1e-9
    roll = np.where(singular, 0.0, np.arctan2(r[..., 2, 1], r[..., 2, 2]))
    yaw = np.where(singular, np.arctan2(-r[..., 0, 1], r[..., 1, 1]),
                   np.arctan2(r[..., 1, 0], r[..., 0, 0]))
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    quat = np.stack((cr*cp*cy + sr*sp*sy,
                     sr*cp*cy - cr*sp*sy,
                     cr*sp*cy + sr*cp*sy,
                     cr*cp*sy - sr*sp*cy), axis=-1)
    quat /= np.linalg.norm(quat, axis=-1, keepdims=True)
    if quaternion_order == "xyzw":
        quat = quat[..., [1, 2, 3, 0]]
    return np.concatenate((transform[..., :3, 3], quat), axis=-1)


def _tool_transforms(value):
    if value is None:
        return np.broadcast_to(np.eye(4), (2, 4, 4))
    transforms = np.asarray(value, dtype=np.float64)
    if transforms.shape == (4, 4):
        transforms = np.broadcast_to(transforms, (2, 4, 4))
    if transforms.shape != (2, 4, 4) or not np.isfinite(transforms).all():
        raise ValueError("tcp_transform must be finite (4,4) or (2,4,4), left then right")
    r = transforms[:, :3, :3]
    if (not np.allclose(transforms[:, 3, :], [0, 0, 0, 1], atol=1e-9, rtol=0)
            or not np.allclose(r.swapaxes(-1, -2) @ r, np.eye(3), atol=1e-8, rtol=0)
            or not np.allclose(np.linalg.det(r), 1, atol=1e-8, rtol=0)):
        raise ValueError("tcp_transform must be a rigid transform (SO(3) rotation)")
    return transforms


def dual_arm_fk(joints14, *, quaternion_order="wxyz", tcp_transform=None):
    """Map (...,14) joints to (...,16) endpose, preserving both grippers.

    Inputs are radians and raw gripper scalars; output position is meters.
    Default zero tool offset matches cap_pen Piper-X recorded endpose.
    tcp_transform: optional flange-to-TCP 4x4 shared by both arms, or
    (2,4,4) left/right transforms. Translation units MUST be meters.
    """
    _check_order(quaternion_order)
    q = _array(joints14, 14, "joints14")
    tools = _tool_transforms(tcp_transform)
    left = _transform_to_pose7(piperx_flange_matrix(q[..., :6]) @ tools[0], quaternion_order)
    right = _transform_to_pose7(piperx_flange_matrix(q[..., 7:13]) @ tools[1], quaternion_order)
    return np.concatenate((left, q[..., 6:7], right, q[..., 13:14]), axis=-1)


def reorder_endpose_quaternions(endpose16, *, source_order="wxyz", target_order="xyzw"):
    """Reorder both quaternion blocks without changing xyz or gripper values."""
    _check_order(source_order)
    _check_order(target_order)
    result = _array(endpose16, 16, "endpose16").copy()
    if source_order != target_order:
        order = [1, 2, 3, 0] if source_order == "wxyz" else [3, 0, 1, 2]
        for start in (3, 11):
            result[..., start:start+4] = result[..., start:start+4][..., order]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joints", type=Path, required=True, help="Input .npy (...,14), radians")
    parser.add_argument("--output", type=Path, required=True, help="New output .npy (...,16), meters")
    parser.add_argument("--quaternion-order", choices=("wxyz", "xyzw"), default="wxyz")
    args = parser.parse_args()
    result = dual_arm_fk(np.load(args.joints, allow_pickle=False), quaternion_order=args.quaternion_order)
    # Exclusive creation prevents accidental replacement of input or old results.
    with args.output.open("xb") as stream:
        np.save(stream, result, allow_pickle=False)
    print(f"Saved {result.shape} to {args.output}; xyz=m, quaternion={args.quaternion_order}, TCP offset=0")


if __name__ == "__main__":
    main()
