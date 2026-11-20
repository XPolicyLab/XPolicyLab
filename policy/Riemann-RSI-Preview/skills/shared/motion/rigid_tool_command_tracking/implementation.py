import math

import numpy as np

from skills.shared.motion.urdf_pose_kinematics.implementation import matrix_pose, pose_matrix


def checked_pose(value):
    pose = np.asarray(value, dtype=float)
    if (pose.shape != (7,) or not np.isfinite(pose).all() or np.max(np.abs(pose[:3])) > 3
            or not .99 <= np.linalg.norm(pose[3:]) <= 1.01):
        raise ValueError('Rigid tool commands use finite own poses and a qualified public attachment belief')
    return pose


def predicted_tool_pose(observed_wrist, attachment):
    return matrix_pose(pose_matrix(checked_pose(observed_wrist)) @ pose_matrix(checked_pose(attachment)))


def wrist_for_tool_goal(tool_goal, attachment):
    return matrix_pose(pose_matrix(checked_pose(tool_goal)) @ np.linalg.inv(pose_matrix(checked_pose(attachment))))


def tracking_correction(goal, predicted_tool, wrist_command, tolerance, rotation_tolerance_degrees):
    goal, predicted_tool, wrist_command = [checked_pose(value) for value in (goal, predicted_tool, wrist_command)]
    if (any(isinstance(value, bool) or not math.isfinite(float(value)) for value in (tolerance, rotation_tolerance_degrees))
            or not 0 < tolerance <= .04 or not 0 < rotation_tolerance_degrees <= 20):
        raise ValueError('Tracking tolerances must be declared bounded mature execution settings')
    translation = goal[:3] - predicted_tool[:3]
    distance = float(np.linalg.norm(translation))
    cosine = abs(float(np.dot(goal[3:], predicted_tool[3:])
                       / (np.linalg.norm(goal[3:]) * np.linalg.norm(predicted_tool[3:]))))
    angle = math.degrees(2 * math.acos(float(np.clip(cosine, 0., 1.))))
    if distance <= tolerance and angle <= rotation_tolerance_degrees:
        status, command = 'within_tolerance', wrist_command.copy()
    elif distance > .04 or angle > 20:
        status, command = 'diverged', None
    else:
        status = 'correction_available'
        corrected = pose_matrix(wrist_command)
        corrected[:3, 3] += translation
        corrected[:3, :3] = pose_matrix(goal)[:3, :3] @ pose_matrix(predicted_tool)[:3, :3].T @ corrected[:3, :3]
        command = matrix_pose(corrected)
    return {'status': status, 'translation_m': distance, 'rotation_degrees': angle,
            'command_pose': command, 'prediction_is_true_object_pose': False}
