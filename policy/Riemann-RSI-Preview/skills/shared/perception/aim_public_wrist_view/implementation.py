import math

import numpy as np

from harness.execution import ExecutionError
from skills.shared.motion.public_phase_execution.implementation import pose_phase
from skills.shared.motion.upright_transfer.implementation import axis_rotation
from skills.shared.perception.rgb_surface_geometry.implementation import camera_model, pixel_rays, rotation_matrix


def target_pose(api, side, centroid, proposed_pose=None, focus_height=None, maximum_rotation=.8):
    if side not in ('left', 'right'):
        raise ExecutionError('A wrist view uses only one declared own arm')
    if (isinstance(maximum_rotation, (bool, np.bool_)) or maximum_rotation not in (.8, 1.6)
            or maximum_rotation == 1.6 and
            any(float(api.targets()[arm + '_ee_joint_state'][0]) < .9 for arm in ('left', 'right'))):
        raise ExecutionError('A larger before-contact view requires both public opening responses')
    if centroid is None:
        return None
    pixels = np.asarray(centroid, dtype=float)
    if pixels.shape != (2,) or not np.isfinite(pixels).all() or not (0 <= pixels[0] < 640 and 0 <= pixels[1] < 480):
        raise ExecutionError('A wrist view requires a finite centroid inside the current public head image')
    initial = api.targets()[side + '_ee_pose']
    target = np.asarray(initial if proposed_pose is None else proposed_pose, dtype=float).copy()
    if (target.shape != (7,) or not np.isfinite(target).all()
            or abs(np.linalg.norm(target[3:]) - 1.) > .01
            or np.linalg.norm(target[:3] - initial[:3]) > .15):
        raise ExecutionError('A wrist view proposes only a finite own pose with a unit orientation')
    aim_height = initial[2] - .155 if focus_height is None else focus_height
    if (not isinstance(aim_height, (int, float, np.floating)) or isinstance(aim_height, (bool, np.bool_))
            or not np.isfinite(aim_height)):
        raise ExecutionError('A view focus height is only a finite current measurement or own tool reference')
    origin, directions = pixel_rays(api, 'cam_head', [pixels])
    direction = directions[0]
    scale = (aim_height - origin[2]) / direction[2] if abs(direction[2]) > .05 else -1.
    if scale <= 0:
        return None
    focus = origin + direction * scale
    camera, unused_focal = camera_model(api, 'cam_' + side + '_wrist')
    own_rotation = rotation_matrix(target[3:]) @ rotation_matrix(initial[3:]).T
    camera[:3, 3] = target[:3] + own_rotation @ (camera[:3, 3] - initial[:3])
    camera[:3, :3] = own_rotation @ camera[:3, :3]
    desired = focus - camera[:3, 3]
    distance = np.linalg.norm(desired)
    if distance < 1e-8:
        return None
    desired /= distance
    current_direction = -camera[:3, 2]
    axis = np.cross(current_direction, desired)
    angle = math.acos(float(np.clip(current_direction @ desired, -1, 1)))
    if np.linalg.norm(axis) > 1e-8:
        axis /= np.linalg.norm(axis)
        target[3:] = axis_rotation(target[3:], axis, min(angle, maximum_rotation))
    elif angle > maximum_rotation:
        return None
    return target


def phases(api, side, centroid):
    initial = api.targets()[side + '_ee_pose']
    safe_home = np.linalg.norm(initial[:3] - api.home_targets()[side + '_ee_pose'][:3]) <= 0.08
    open_commands = all(api.targets()[arm + '_ee_joint_state'][0] >= 0.9 for arm in ('left', 'right'))
    allowed = centroid is not None and safe_home and open_commands
    target = initial.copy()
    if allowed:
        aimed = target_pose(api, side, centroid)
        allowed = aimed is not None
        if allowed:
            target = aimed
    return [pose_phase('open_at_public_home_before_view', initial, 1, 4, False),
            pose_phase('aim_own_wrist_camera_at_visible_face', target, 1),
            pose_phase('settle_public_view_after_robot_motion', target, 1, 3)], allowed
