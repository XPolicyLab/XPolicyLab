"""Validate the old Fold own-robot mounting datum, never a measured surface."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
from skills.shared.motion.declared_pickup_kinematics.implementation import run as own_model
from skills.shared.motion.urdf_pose_kinematics.implementation import pose_matrix

SOURCE = 'public_own_robot_mounting_datum_v45'
URDF_SHA256 = '7aaa308a1a5b6a213d914db86ba09ddd6252f25cb498ac9a148bfa6c7166a423'
MAXIMUM_ROOT_HEIGHT_DISAGREEMENT_M = .005


def validate(value):
    fields = {'source', 'reference_semantics', 'kinematic_model_sha256',
        'current_public_action_epoch', 'public_own_state', 'own_base_poses',
        'mount_z_m', 'root_height_disagreement_m',
        'height_is_observed_surface', 'surface_height_error_bound_m'}
    if (not isinstance(value, dict) or set(value) != fields or value['source'] != SOURCE
            or value['reference_semantics'] != 'v45_mean_current_public_own_root_Z_static_task_datum'
            or value['kinematic_model_sha256'] != URDF_SHA256
            or value['height_is_observed_surface'] is not False
            or value['surface_height_error_bound_m'] is not None
            or type(value['current_public_action_epoch']) is not int
            or not 0 <= value['current_public_action_epoch'] <= 500):
        raise ValueError('A Fold mounting datum is explicitly model-based, not an observed support or contact')
    state = value['public_own_state']
    expected = {s + suffix for s in ('left', 'right') for suffix in ('_arm_joint_state', '_ee_pose')}
    if not isinstance(state, dict) or set(state) != expected:
        raise ValueError('A mounting datum uses only two current public EEF poses and six joints per arm')
    model = own_model()
    root = Path(__file__).resolve().parents[4]
    settings = json.loads((root / 'configs/mature-trajectory.json').read_text())
    if settings['robot_urdf_sha256'] != URDF_SHA256:
        raise ValueError('The declared original Fold robot model differs')
    computed = {}
    for side in ('left', 'right'):
        joints = np.asarray(state[side + '_arm_joint_state'], dtype=float)
        ee = np.asarray(state[side + '_ee_pose'], dtype=float)
        if (joints.shape != (6,) or ee.shape != (7,)
                or not np.isfinite(joints).all() or not np.isfinite(ee).all()
                or not .99 <= np.linalg.norm(ee[3:]) <= 1.01):
            raise ValueError('The mounting datum needs valid current own arm/EEF observations')
        computed[side] = pose_matrix(ee) @ np.linalg.inv(model.eef(joints))
    poses = value['own_base_poses']
    if not isinstance(poses, dict) or set(poses) != {'left', 'right'}:
        raise ValueError('The mounting datum has two independently inferred own roots')
    for side, transform in computed.items():
        reported = np.asarray(poses[side], dtype=float)
        if (reported.shape != (7,) or not np.isfinite(reported).all() or np.max(np.abs(reported[:3])) > 3.
                or not .99 <= np.linalg.norm(reported[3:]) <= 1.01):
            raise ValueError('The mounting datum roots are bounded public robot geometry')
        if not np.allclose(pose_matrix(reported), transform, atol=1e-8, rtol=0):
            raise ValueError('The mounting datum root differs from current public EEF/FK')
    heights = [transform[2, 3] for transform in computed.values()]
    mean, disagreement = float(np.mean(heights)), float(abs(heights[0] - heights[1]))
    if disagreement > MAXIMUM_ROOT_HEIGHT_DISAGREEMENT_M:
        raise ValueError('Independent original X5 mounting datum estimates disagree by more than5mm')
    for name, expected_value in (('mount_z_m', mean), ('root_height_disagreement_m', disagreement)):
        supplied = value[name]
        if type(supplied) not in (int, float) or not np.isfinite(supplied) or abs(supplied - expected_value) > 1e-8:
            raise ValueError('The mounting datum must retain its actual own-robot derivation')
    return deepcopy(value)
