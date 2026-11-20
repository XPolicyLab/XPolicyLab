import math

import numpy as np


def closing_orientation(closing_yaw, tilt_degrees=15, arm='left'):
    if (not isinstance(closing_yaw, (int, float)) or not math.isfinite(closing_yaw)
            or tilt_degrees not in (15, 25, 35, 0) or arm not in ('left', 'right')):
        raise ValueError('Use a current measured closing axis and one original own-gripper tilt')
    closing = np.array([math.cos(closing_yaw), math.sin(closing_yaw), 0.])
    lateral = np.array([-closing[1], closing[0], 0.])
    if (lateral[1] < 0) != (arm == 'right'):
        lateral *= -1
    tilt = math.radians(tilt_degrees)
    forward = math.sin(tilt) * lateral + np.array([0., 0., -math.cos(tilt)])
    closing -= np.dot(closing, forward) * forward
    closing /= np.linalg.norm(closing)
    upward = np.cross(forward, closing)
    return np.column_stack((forward, closing, upward))


def run(payload):
    if not isinstance(payload, dict) or set(payload) != {'closing_yaw', 'tilt_degrees', 'arm'}:
        raise ValueError('Closing orientation accepts only its declared current-axis and own-arm parameters')
    return {'rotation': closing_orientation(**payload), 'robot_actions_issued': 0, 'object_pose_is_GT': False}
