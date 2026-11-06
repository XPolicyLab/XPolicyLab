from __future__ import annotations
import math
import numpy as np
from skills.shared.geometry.finite_rotation_pose.implementation import validate_pose
from skills.shared.geometry.rigid_contact_transform.implementation import quat_mul_wxyz, quat_inverse_wxyz
from skills.shared.geometry.bounded_planar_increment.implementation import wrap_yaw, yaw_of_quaternion

def quaternion_angle(quaternion, target_quaternion):
    """Full SO(3) geodesic error in [0, pi], invariant under q -> -q."""
    relative = quat_mul_wxyz(quat_inverse_wxyz(quaternion), target_quaternion)
    return 2 * math.atan2(float(np.linalg.norm(relative[1:])), abs(float(relative[0])))

def pose_errors(actual_pose, target_pose):
    """Position/full orientation errors; yaw is diagnostic, never success.

    yaw_rad is None for an undefined planar heading (e.g. a tipped T with
    vertical local X). orientation_rad still measures that full error.
    """
    actual, target = (validate_pose(actual_pose), validate_pose(target_pose))
    delta = actual[:3] - target[:3]
    angle = quaternion_angle(actual[3:], target[3:])
    try:
        yaw = abs(wrap_yaw(yaw_of_quaternion(actual[3:]) - yaw_of_quaternion(target[3:])))
    except ValueError:
        yaw = None
    return {'position_m': float(np.linalg.norm(delta)), 'xy_m': float(np.linalg.norm(delta[:2])), 'z_m': abs(float(delta[2])), 'orientation_rad': angle, 'orientation_deg': math.degrees(angle), 'yaw_rad': yaw, 'yaw_deg': math.degrees(yaw) if yaw is not None else None}
