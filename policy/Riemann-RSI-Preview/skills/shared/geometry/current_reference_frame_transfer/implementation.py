import numpy as np

from harness.execution import ExecutionError
from skills.shared.motion.urdf_pose_kinematics.implementation import matrix_pose, pose_matrix


def checked_pose(value):
    try:
        pose = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise ExecutionError('A frame contains only finite public numeric pose components') from error
    if (pose.shape != (7,) or not np.isfinite(pose).all() or np.any(np.abs(pose[:3]) > 5.)
            or not .99 <= np.linalg.norm(pose[3:]) <= 1.01):
        raise ExecutionError('A frame requires bounded public XYZ and normalized own-format WXYZ')
    return pose_matrix(pose)


def run(source_reference, target_reference, tool_contact_command):
    source, target, command = map(checked_pose, (source_reference, target_reference, tool_contact_command))
    relative = target @ np.linalg.inv(source)
    transferred = relative @ command
    pose = matrix_pose(transferred)
    checked_pose(pose)
    return {'pose': pose, 'source': 'current_reference_frame_transport_of_own_contact_command',
        'attachment_is_hypothesis': True, 'true_object_pose_or_asset_geometry_read': False,
        'contact_measurement_claimed': False, 'robot_actions_issued': 0, 'official_success_claimed': False}
