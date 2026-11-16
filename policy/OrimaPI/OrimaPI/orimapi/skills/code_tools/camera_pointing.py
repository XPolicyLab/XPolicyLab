"""Geometric wrist-camera pointing hypotheses, not future visibility or motion.

Inputs are current public robot/camera calibration and a current estimated target.
The rigid camera offset is included; the camera is not assumed to sit at the TCP.
No RGB, depth, object assets, motion command or hidden simulator state is read.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from .geometry import vector


def rigid(value):
    value = np.asarray(value, dtype=float)
    if (
        value.shape != (4, 4) or not np.isfinite(value).all()
        or not np.allclose(value[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
        or not np.allclose(value[:3, :3].T @ value[:3, :3], np.eye(3), atol=1e-6, rtol=0)
        or not np.isclose(np.linalg.det(value[:3, :3]), 1, atol=1e-6, rtol=0)
    ):
        raise ValueError("Camera pointing requires finite proper rigid transforms")
    return value.copy()


def align_rotation(first, second, preferred_axis):
    first, second = vector(first), vector(second)
    if min(np.linalg.norm(first), np.linalg.norm(second)) < 1e-12:
        raise ValueError("Pointing directions must be nonzero")
    first = first / np.linalg.norm(first)
    second = second / np.linalg.norm(second)
    cross = np.cross(first, second)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(np.dot(first, second), -1, 1))
    if sine > 1e-10:
        angle = float(np.arctan2(sine, cosine))
        return Rotation.from_rotvec(cross / sine * angle).as_matrix(), angle
    if cosine >= 0:
        return np.eye(3), 0.0
    axis = vector(preferred_axis)
    axis = axis - np.dot(axis, first) * first
    if np.linalg.norm(axis) < 1e-8:
        axis = np.eye(3)[int(np.argmin(np.abs(first)))]
        axis = axis - np.dot(axis, first) * first
    axis /= np.linalg.norm(axis)
    return Rotation.from_rotvec(np.pi * axis).as_matrix(), float(np.pi)


def point_camera(current_tcp, tcp_from_optical, target_world, tcp_offset=(0, 0, 0),
                 target_axis_twist_deg=0.0):
    """Center a static target on the modeled optical axis at a chosen nearby TCP.

    Let c be camera-origin displacement from TCP expressed in optical axes.
    Centering requires target-TCP = R_camera @ (c + [0,0,depth]).
    The norm determines positive optical depth, then one shortest rotation maps
    that vector to the world target displacement. This retains mount translation.
    Other rolls/TCPs may be feasible; failure does not exhaust all viewpoints.
    """
    current_tcp, mount = rigid(current_tcp), rigid(tcp_from_optical)
    target, offset = vector(target_world), vector(tcp_offset)
    if (
        type(target_axis_twist_deg) not in (int, float)
        or not np.isfinite(target_axis_twist_deg)
        or abs(target_axis_twist_deg) > 180
    ):
        raise ValueError("Pointing target-axis twist must be a finite angle within180 degrees")
    if np.linalg.norm(offset) > 0.15 + 1e-9:
        raise ValueError("Camera-pointing TCP offset is bounded to0.15m")
    position = current_tcp[:3, 3] + offset
    displacement = target - position
    distance = float(np.linalg.norm(displacement))
    if not np.isfinite(distance) or not 1e-6 < distance <= 5.0:
        raise ValueError("Pointing target distance must be finite, positive and within5m")
    current_camera = current_tcp @ mount
    camera_local_offset = mount[:3, :3].T @ mount[:3, 3]
    radicand = distance * distance - float(np.dot(camera_local_offset[:2], camera_local_offset[:2]))
    if radicand <= 1e-12:
        raise ValueError("Target is too close for centered pointing at this fixed TCP")
    depth = float(np.sqrt(radicand) - camera_local_offset[2])
    if depth <= 1e-6:
        raise ValueError("No positive modeled optical depth for this pointing hypothesis")
    local_ray_displacement = camera_local_offset + [0, 0, depth]
    current_world_displacement = current_camera[:3, :3] @ local_ray_displacement
    delta, angle = align_rotation(
        current_world_displacement, displacement, current_camera[:3, 0]
    )
    camera_rotation = delta @ current_camera[:3, :3]
    # Rotate the whole rigid camera/TCP pose about TCP-to-target, NOT simply
    # the optical axis: the latter would ignore mount translation and lose aim.
    twist = Rotation.from_rotvec(
        displacement / distance * np.deg2rad(target_axis_twist_deg)
    ).as_matrix()
    camera_rotation = twist @ camera_rotation
    proposed_tcp = np.eye(4)
    proposed_tcp[:3, :3] = camera_rotation @ mount[:3, :3].T
    proposed_tcp[:3, 3] = position
    proposed_tcp = rigid(proposed_tcp)
    proposed_camera = proposed_tcp @ mount
    optical_target = proposed_camera[:3, :3].T @ (target - proposed_camera[:3, 3])
    if optical_target[2] <= 0 or np.linalg.norm(optical_target[:2]) > 1e-7:
        raise ValueError("Camera pointing numerical verification failed")
    return proposed_tcp, {
        "world_from_optical_before": current_camera.tolist(),
        "world_from_optical_proposed": proposed_camera.tolist(),
        "modeled_target_optical_m": optical_target.tolist(),
        "modeled_optical_depth_m": float(optical_target[2]),
        "tcp_offset_m": offset.tolist(),
        "orientation_change_rad": angle,
        "target_axis_twist_deg": float(target_axis_twist_deg),
        "total_orientation_change_rad": float(
            Rotation.from_matrix(camera_rotation @ current_camera[:3, :3].T).magnitude()
        ),
        "rigid_camera_offset_included": True,
        "target_assumed_static": True,
        "future_rgb_acquired": False,
        "occlusion_checked": False,
        "scene_clearance_checked": False,
        "complete_object_coverage_checked": False,
        "scope": "One geometric pointing hypothesis, not future imagery, visibility, full-scene reach or contact certification. Reobserve after checked execution.",
    }
