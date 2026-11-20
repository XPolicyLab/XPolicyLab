"""A conditional rigid payload goal from public source and robot poses."""
import numpy as np
from skills.shared.geometry.native_rigid_pose.implementation import quat_to_mat_wxyz

def run(source_pose, grasp_ee_pose, requested_ee_pose, *, destination_xy, support_top_z, half_extents):
    source, grasp, requested = (np.asarray(v) for v in (source_pose, grasp_ee_pose, requested_ee_pose))
    xy, half = (np.asarray(destination_xy), np.asarray(half_extents))
    if any((v.shape != (7,) or not np.isfinite(v).all() or np.linalg.norm(v[3:]) < 1e-08 for v in (source, grasp, requested))):
        raise ValueError('Finite observed source, own grasp and commanded placement poses required')
    if xy.shape != (2,) or half.shape != (3,) or (not np.isfinite(xy).all()) or (not np.isfinite(half).all()) or np.any(half <= 0) or (not np.isfinite(support_top_z)):
        raise ValueError('Finite public horizontal support and positive declared payload shape required')
    grasp_rotation = quat_to_mat_wxyz(grasp[3:])
    placement_rotation = quat_to_mat_wxyz(requested[3:])
    relative_center = grasp_rotation.T @ (source[:3] - grasp[:3])
    relative_rotation = grasp_rotation.T @ quat_to_mat_wxyz(source[3:])
    payload_rotation = placement_rotation @ relative_rotation
    vertical_half_extent = float(np.abs(payload_rotation[2]) @ half)
    target_center = np.r_[xy, float(support_top_z) + vertical_half_extent]
    result = requested.copy()
    result[:3] = target_center - placement_rotation @ relative_center
    return (result, dict(source='public_source_pose_and_public_preclose_EE_pose', assumption='rigid_pose_if_payload_is_held', attachment_evidence='unknown', physical_finger_measurements_used=False, relative_center_in_EE=relative_center.tolist(), expected_payload_center=target_center.tolist(), oriented_vertical_half_extent_m=vertical_half_extent, requested_EE_position=requested[:3].tolist(), corrected_EE_position=result[:3].tolist(), commanded_orientation_preserved=True))
