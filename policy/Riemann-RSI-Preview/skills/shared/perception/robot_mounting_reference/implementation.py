import numpy as np

from skills.shared.perception.rgb_surface_geometry.implementation import rotation_matrix


def run(bases):
    if not isinstance(bases, dict) or set(bases) != {'left', 'right'}:
        raise ValueError('Use only the two current own-robot FK-derived public bases')
    poses = [np.asarray(bases[side], dtype=float) for side in ('left', 'right')]
    if any(pose.shape != (7,) or not np.isfinite(pose).all() or np.max(np.abs(pose[:3])) > 3 for pose in poses):
        raise ValueError('An own mounting reference needs finite declared robot poses')
    heights = [float(pose[2]) for pose in poses]
    if (np.ptp(heights) > .005
            or any(float(rotation_matrix(pose[3:])[2, 2]) < .999 for pose in poses)):
        return {'status': 'uncertain', 'height_m': None,
                'reason': 'independent_own_mounts_disagree_or_are_tilted'}
    return {'status': 'observed', 'height_m': float(np.mean(heights)),
            'source': 'current_own_robot_pose_and_declared_URDF',
            'semantics': 'robot_mounting_reference_not_an_object_or_table_measurement',
            'object_support_requires_task_contract': True}
