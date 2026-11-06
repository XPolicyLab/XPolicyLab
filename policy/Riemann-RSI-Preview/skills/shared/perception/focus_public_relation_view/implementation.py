import numpy as np

from skills.shared.perception.raised_public_wrist_view.implementation import run as raised_view
from skills.shared.perception.rgb_surface_geometry.implementation import rotation_matrix
from skills.shared.verification.observed_rigid_transfer.implementation import validate


def run(api, current):
    current = validate(current)
    if (current['phase'] != 'view' or current['scene'] is not None or current['view_attempts'] >= 4
            or current['view_hint'] is None or current['support_memory'] is None
            or not api.own_robot_bases_available()
            or any(float(api.targets()[side + '_ee_joint_state'][0]) < .9 for side in ('left', 'right'))):
        return None
    hint = current['view_hint']
    proposal_index = max(0, current['job']['observation_count'] - 1)
    preferred = 'left' if proposal_index % 2 == 0 else 'right'
    camera_name = 'cam_' + preferred + '_wrist'
    if camera_name not in hint['unqualified_cameras']:
        camera_name = hint['unqualified_cameras'][0]
    side = camera_name.split('_')[1]
    plane = current['support_memory']['height_estimate_m']
    aimed = raised_view(api, side, hint['head_centroid'], plane)
    if aimed is None:
        return None
    tip = aimed[:3] + rotation_matrix(aimed[3:]) @ np.asarray([current['goal']['own_forward_m'], 0., 0.])
    if tip[2] < plane + .06:
        return None
    return {'side': side, 'pose': aimed}
