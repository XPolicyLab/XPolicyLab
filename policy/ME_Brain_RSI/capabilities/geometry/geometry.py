from __future__ import annotations
from collections.abc import Mapping, Sequence
from math import isfinite, sqrt
import numpy as np
from .contract import InsertTubesEnvelopeError
def _numbers(value: object, size: int, name: str) -> list[float]:
    raw = value
    for method_name in ('detach', 'cpu'):
        method = getattr(raw, method_name, None)
        if callable(method):
            raw = method()
    tolist = getattr(raw, 'tolist', None)
    if callable(tolist):
        raw = tolist()
    try:
        result = [float(item) for item in raw]
    except (TypeError, ValueError, OverflowError) as error:
        raise InsertTubesEnvelopeError(f'{name} must be numeric') from error
    if len(result) != size or not all((isfinite(item) for item in result)):
        raise InsertTubesEnvelopeError(f'{name} must contain {size} finite values')
    return result

def _rotation(quaternion: Sequence[float]) -> np.ndarray:
    w, x, y, z = quaternion
    norm = sqrt(w * w + x * x + y * y + z * z)
    if norm <= 1e-12:
        raise InsertTubesEnvelopeError('quaternion must be nonzero')
    w, x, y, z = (item / norm for item in (w, x, y, z))
    return np.asarray([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]], dtype=float)

def _quaternion(matrix: np.ndarray) -> list[float]:
    return _shared_quaternion_from_matrix(matrix)

def semantic_pose_to_ee(robot: Any, pose: Sequence[float]) -> list[float]:
    return _shared_grasp_pose_to_end_effector(robot, pose, rotation_fn=_rotation, quaternion_fn=_quaternion)

def _offset_local_x(pose: Sequence[float], distance: float) -> list[float]:
    result = np.asarray(pose[:3], dtype=float) + _rotation(pose[3:])[:, 0] * float(distance)
    return result.tolist() + [float(item) for item in pose[3:]]
def _ee_pose_to_semantic(robot: Any, ee_pose: Sequence[float]) -> list[float]:
    ee_rotation = _rotation(ee_pose[3:])
    semantic_rotation = ee_rotation @ np.asarray(robot.delta_matrix, dtype=float).reshape(3, 3)
    semantic_position = np.asarray(ee_pose[:3], dtype=float) - semantic_rotation @ np.asarray([0.12 - float(robot.gripper_bias), 0.0, 0.0])
    return semantic_position.tolist() + _quaternion(semantic_rotation)

_FOCAL_PIXELS=640.0*10.0/22.212
def pixel_to_world_on_height(pixel_xy: Sequence[float], *, world_z: float) -> tuple[float, float]:
    """Intersect a head-camera pixel ray with a fixed world-Z plane."""
    if len(pixel_xy) != 2:
        raise InsertTubesEnvelopeError('pixel_xy must contain two values')
    u, v = (float(value) for value in pixel_xy)
    z = float(world_z)
    if not all((isfinite(value) for value in (u, v, z))):
        raise InsertTubesEnvelopeError('pixel ray inputs must be finite')
    ray_x = (u - 320.0) / _FOCAL_PIXELS
    ray_up = -(v - 240.0) / _FOCAL_PIXELS
    ray_y = 0.8660254037844386 * ray_up + 0.5
    ray_z = 0.5 * ray_up - 0.8660254037844386
    distance = (z - 1.308) / ray_z
    return (distance * ray_x, -0.41 + distance * ray_y)

def world_to_pixel_on_height(world_xy: Sequence[float], *, world_z: float) -> tuple[float, float]:
    """Project a world point through the fixed official head camera."""
    if len(world_xy) != 2:
        raise InsertTubesEnvelopeError('world_xy must contain two values')
    world_x, world_y = (float(value) for value in world_xy)
    world_z = float(world_z)
    delta_y = world_y + 0.41
    delta_z = world_z - 1.308
    distance = 0.5 * delta_y - 0.8660254037844386 * delta_z
    ray_up_distance = 0.8660254037844386 * delta_y + 0.5 * delta_z
    if abs(distance) < 1e-09:
        raise InsertTubesEnvelopeError('world point lies on the camera plane')
    return (320.0 + _FOCAL_PIXELS * world_x / distance, 240.0 - _FOCAL_PIXELS * ray_up_distance / distance)

from lxm6wuec_skill_library.pose_skills import quaternion_from_matrix as _shared_quaternion_from_matrix
from lxm6wuec_skill_library.pose_skills import grasp_pose_to_end_effector as _shared_grasp_pose_to_end_effector
