import math

import numpy as np

from skills.shared.geometry.orientation_quaternion.implementation import mat_to_quat_wxyz


def downward_tool_x(horizontal_direction, tilt_deg=25.0):
    horizontal = np.asarray(horizontal_direction, dtype=np.float64).reshape(-1)
    if horizontal.size != 2 or not np.isfinite(horizontal).all() or np.linalg.norm(horizontal) < 1e-8:
        raise ValueError('A downward tool orientation requires a finite observed horizontal direction')
    horizontal /= np.linalg.norm(horizontal)
    tilt = math.radians(float(tilt_deg))
    tool_axis = np.asarray([horizontal[0] * math.cos(tilt), horizontal[1] * math.cos(tilt), -math.sin(tilt)])
    vertical_axis = np.asarray([0.0, 0.0, 1.0])
    vertical_axis -= tool_axis * float(np.dot(tool_axis, vertical_axis))
    vertical_axis /= max(np.linalg.norm(vertical_axis), 1e-12)
    cross_axis = np.cross(vertical_axis, tool_axis)
    cross_axis /= max(np.linalg.norm(cross_axis), 1e-12)
    return mat_to_quat_wxyz(np.column_stack([tool_axis, cross_axis, vertical_axis]))
