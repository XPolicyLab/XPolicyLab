import numpy as np

from harness.execution import ExecutionError
from skills.shared.perception.aim_public_wrist_view.implementation import target_pose
from skills.shared.perception.rgb_surface_geometry.implementation import camera_model, rotation_matrix


def run(api, side, centroid, focus_height, *, use_episode_home=False):
    if side not in ('left', 'right') or type(use_episode_home) is not bool:
        raise ExecutionError('A raised public view uses one own arm and an explicit own reference')
    initial = api.targets()[side + '_ee_pose']
    reference = api.home_targets()[side + '_ee_pose'] if use_episode_home else initial
    camera, unused_focal = camera_model(api, 'cam_' + side + '_wrist')
    own_rotation = rotation_matrix(reference[3:]) @ rotation_matrix(initial[3:]).T
    proposed = reference.copy()
    proposed[:3] += np.asarray([0., 0., .06]) + own_rotation @ camera[:3, 2] * .08
    if np.linalg.norm(proposed[:3] - initial[:3]) > .15:
        return None
    return target_pose(api, side, centroid, proposed, focus_height)
