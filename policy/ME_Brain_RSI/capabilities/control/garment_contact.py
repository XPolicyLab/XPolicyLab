"""Place cloth contact targets above the complete public finger solids."""
from functools import lru_cache

import numpy as np

from b88r42wq.motion import TCP_OFFSET, quat_matrix


TABLE_MARGIN_M = .001
CONTACT_RESERVE_M = .0005
NOMINAL_CONTACT_HEIGHT_M = .0125


@lru_cache(maxsize=1)
def _complete_finger_vertices():
    """Reuse validated complete hulls, including the unknown-opening expansion."""
    from b88r42wq.robot_motion_geometry import _assets
    assets, _, _ = _assets()
    return np.vstack([assets[name][1] for name in ("link7", "link8")])


def finger_lowest_offset(quat):
    """Return the lowest finger Z relative to TCP for the intended measured wrist."""
    local = _complete_finger_vertices() - TCP_OFFSET
    return float(np.min(local @ quat_matrix(quat)[2]))


def contact_height(quat):
    """Keep low fabric contact while retaining the shared positive table margin."""
    return max(NOMINAL_CONTACT_HEIGHT_M,
               TABLE_MARGIN_M + CONTACT_RESERVE_M - finger_lowest_offset(quat))


def clear_low_point(ctx, point, quat, *, arm=None, kind="carry"):
    """Raise a task goal only when its complete fingers would violate table clearance."""
    result = np.asarray(point, float).copy()
    nominal = float(result[2])
    result[2] = max(result[2], ctx.table_z + contact_height(quat))
    if arm is not None:
        ctx.memory.setdefault("garment_contact_geometry", []).append({
            "stage": ctx.stage, "arm": arm, "kind": kind,
            "nominal_tcp_height_m": nominal - ctx.table_z,
            "tcp_height_m": float(result[2] - ctx.table_z),
            "minimum_finger_table_clearance_m": float(result[2] - ctx.table_z + finger_lowest_offset(quat)),
            "required_table_margin_m": TABLE_MARGIN_M,
            "task_reserve_m": CONTACT_RESERVE_M,
            "unknown_jaw_opening_preserved": True})
    return result


def sleeve_goals(ctx, kinematics, arm, source, target, quat, tilt, carry_quat):
    """Use identical geometric targets for sleeve preflight and actual execution."""
    source = clear_low_point(ctx, source, quat)
    goals = [(_lift(source, .075), quat, 12), (source, quat, 12)]
    for fraction in np.linspace(.25, 1., 4):
        point = (1-fraction)*source + fraction*target
        point[2] += .075*np.sin(np.pi*fraction) + .015*fraction
        orientation = carry_quat(kinematics, arm, target, quat, tilt, fraction)
        goals.append((clear_low_point(ctx, point, orientation), orientation, 8))
    retreat = _lift(target, .09)
    retreat[:2] = (target[:2] + np.asarray(ctx.initial_state[f"{arm}_ee_pose"])[:2])/2
    goals.append((retreat, quat, 10))
    return goals


def _lift(point, height):
    result = np.asarray(point, float).copy()
    result[2] += height
    return result
