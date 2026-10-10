"""NumPy trajectory selection plus explicit IK approach generation."""

from __future__ import annotations

import random
from itertools import combinations
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from vlm_orchestrator.robodojo_camera import ROBODOJO_DEFAULT_TABLE_HEIGHT_M

_JOINT_ORIGINS = (
    ((0.0, 0.0, 0.0605), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    ((0.02, 0.0, 0.04), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    ((-0.264, 0.0, 0.0), (3.1416, 0.0, 0.0), (0.0, 1.0, 0.0)),
    ((0.245, 0.0, -0.056), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    ((0.06775, 0.0005, -0.0865), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    ((0.02895, 0.0, 0.0865), (-3.1416, 0.0, 0.0), (1.0, 0.0, 0.0)),
)
_LEFT_BASE = np.asarray([-0.3, -0.45, 0.765, 0.707, 0.0, 0.0, 0.707])
_RIGHT_BASE = np.asarray([0.3, -0.45, 0.765, 0.707, 0.0, 0.0, 0.707])
_GRIPPER_OFFSET = np.asarray([0.145, -0.000002, -0.00024363])
DEFAULT_FAR_DISTANCE_M = 0.15
DEFAULT_NEAR_TRAJECTORY_STEPS = 30
DEFAULT_FAR_TRAJECTORY_STEPS = 30
WRIST_TARGET_STANDOFF_M = 0.15
WRIST_TARGET_DISTANCE_TOLERANCE_M = 0.01
WRIST_TARGET_CENTER_TOLERANCE_PX = 10.0


def _rpy(rpy: Sequence[float]) -> np.ndarray:
    roll, pitch, yaw = map(float, rpy)
    cx, sx = np.cos(roll), np.sin(roll)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cz, sz = np.cos(yaw), np.sin(yaw)
    return np.asarray(
        (
            (cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx),
            (sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx),
            (-sy, cy * sx, cy * cx),
        ),
        dtype=np.float64,
    )


def _axis_rotation(axis: Sequence[float], angle: float) -> np.ndarray:
    axis_array = np.asarray(axis, dtype=np.float64)
    axis_array /= max(float(np.linalg.norm(axis_array)), 1e-12)
    x, y, z = axis_array
    cross = np.asarray(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))
    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)


def _pose(pose: Sequence[float]) -> np.ndarray:
    values = np.asarray(pose, dtype=np.float64).reshape(7)
    w, x, y, z = values[3:]
    norm = max(float(np.linalg.norm((w, x, y, z))), 1e-12)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    rotation = np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = values[:3]
    return result


def x5_gripper_pose(
    joints: Sequence[float], base_pose: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Return the X5 gripper center and link-6 rotation in world coordinates."""
    values = np.asarray(joints, dtype=np.float64).reshape(-1)
    if values.size < 6:
        raise ValueError("X5 FK requires six arm joints")
    transform = _pose(base_pose)
    for joint, (translation, rpy, axis) in zip(values[:6], _JOINT_ORIGINS, strict=True):
        step = np.eye(4)
        step[:3, :3] = _rpy(rpy)
        step[:3, 3] = translation
        transform = transform @ step
        rotation = np.eye(4)
        rotation[:3, :3] = _axis_rotation(axis, float(joint))
        transform = transform @ rotation
    offset = np.append(_GRIPPER_OFFSET, 1.0)
    return (transform @ offset)[:3], transform[:3, :3].copy()


def x5_gripper_center(joints: Sequence[float], base_pose: Sequence[float]) -> np.ndarray:
    """Return the X5 gripper center in world coordinates."""
    return x5_gripper_pose(joints, base_pose)[0]


def _arm_joints(step: Any) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(step, Mapping):
        if "left_arm_joint_state" in step and "right_arm_joint_state" in step:
            return (
                np.asarray(step["left_arm_joint_state"], dtype=np.float64).reshape(-1)[:6],
                np.asarray(step["right_arm_joint_state"], dtype=np.float64).reshape(-1)[:6],
            )
        if "arm_joint_state" in step:
            arm = np.asarray(step["arm_joint_state"], dtype=np.float64).reshape(-1)[:6]
            return arm, arm.copy()
        raise ValueError("action step has no arm joint state")
    values = np.asarray(step, dtype=np.float64).reshape(-1)
    if values.size >= 14:
        return values[:6], values[7:13]
    if values.size >= 12:
        return values[:6], values[6:12]
    raise ValueError(f"unsupported action vector shape {values.shape}")


def action_steps(chunk: Any) -> list[Any]:
    if isinstance(chunk, np.ndarray):
        values = chunk if chunk.ndim > 1 else chunk[None, :]
        return [values[index] for index in range(len(values))]
    if isinstance(chunk, (list, tuple)):
        return list(chunk)
    if isinstance(chunk, Mapping):
        return [chunk]
    raise ValueError(f"unsupported action chunk type {type(chunk).__name__}")


def fk_trajectory(chunk: Any) -> dict[str, Any]:
    steps = action_steps(chunk)
    left, right = [], []
    left_joints, right_joints = [], []
    for step in steps:
        left_arm, right_arm = _arm_joints(step)
        left_joints.append(left_arm)
        right_joints.append(right_arm)
        left.append(x5_gripper_center(left_arm, _LEFT_BASE))
        right.append(x5_gripper_center(right_arm, _RIGHT_BASE))
    return {
        "left": np.asarray(left),
        "right": np.asarray(right),
        "left_joints": np.asarray(left_joints),
        "right_joints": np.asarray(right_joints),
    }


def current_ee_pose(current_joints: Any, arm: str) -> tuple[np.ndarray, np.ndarray]:
    """Return the selected arm's current world-frame gripper pose."""
    left_joints, right_joints = _arm_joints(current_joints)
    if arm == "left":
        return x5_gripper_pose(left_joints, _LEFT_BASE)
    if arm == "right":
        return x5_gripper_pose(right_joints, _RIGHT_BASE)
    raise ValueError("target EE pose requires an explicit left or right arm")


def _transform(position: Any, rotation: Any) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    transform[:3, 3] = np.asarray(position, dtype=np.float64).reshape(3)
    return transform


def wrist_centered_ee_target(
    current_joints: Any,
    arm: str,
    target_world: Any,
    current_wrist_camera_to_world: Any,
    preferred_gripper_rotation: Any,
    *,
    standoff_m: float = WRIST_TARGET_STANDOFF_M,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return an EE target whose wrist camera looks at the target point."""
    if not np.isfinite(standoff_m) or standoff_m <= 0:
        raise ValueError("wrist-camera standoff must be a positive finite distance")
    target = np.asarray(target_world, dtype=np.float64).reshape(3)
    current_camera = np.asarray(
        current_wrist_camera_to_world, dtype=np.float64,
    ).reshape(4, 4)
    preferred = np.asarray(preferred_gripper_rotation, dtype=np.float64).reshape(3, 3)
    if not (
        np.isfinite(target).all()
        and np.isfinite(current_camera).all()
        and np.isfinite(preferred).all()
    ):
        raise ValueError("wrist-centered IK requires finite target and camera geometry")

    current_position, current_rotation = current_ee_pose(current_joints, arm)
    current_ee = _transform(current_position, current_rotation)
    try:
        ee_to_camera = np.linalg.inv(current_ee) @ current_camera
    except np.linalg.LinAlgError as exc:
        raise ValueError("current wrist-camera extrinsic is not invertible") from exc

    view_direction = preferred[:, 0].copy()
    view_norm = float(np.linalg.norm(view_direction))
    if view_norm < 1e-8:
        raise ValueError("wrist-camera approach direction is degenerate")
    view_direction /= view_norm
    camera_z = -view_direction
    current_camera_rotation = current_camera[:3, :3]
    camera_x = current_camera_rotation[:, 0].copy()
    camera_x -= np.dot(camera_x, camera_z) * camera_z
    if np.linalg.norm(camera_x) < 1e-8:
        camera_x = current_camera_rotation[:, 1].copy()
        camera_x -= np.dot(camera_x, camera_z) * camera_z
    if np.linalg.norm(camera_x) < 1e-8:
        reference = np.asarray([1.0, 0.0, 0.0])
        if abs(float(np.dot(reference, camera_z))) > 0.9:
            reference = np.asarray([0.0, 1.0, 0.0])
        camera_x = reference - np.dot(reference, camera_z) * camera_z
    camera_x /= np.linalg.norm(camera_x)
    camera_y = np.cross(camera_z, camera_x)
    camera_y /= np.linalg.norm(camera_y)

    ee_to_camera_rotation = ee_to_camera[:3, :3]
    preferred_axis = preferred[:, 1].copy()
    preferred_axis /= max(float(np.linalg.norm(preferred_axis)), 1e-12)
    best = None
    for angle in np.linspace(0.0, 2.0 * np.pi, 721)[:-1]:
        cosine, sine = np.cos(angle), np.sin(angle)
        candidate_x = cosine * camera_x + sine * camera_y
        candidate_y = -sine * camera_x + cosine * camera_y
        candidate_camera_rotation = np.column_stack((candidate_x, candidate_y, camera_z))
        candidate_ee_rotation = candidate_camera_rotation @ ee_to_camera_rotation.T
        closing_axis_alignment = abs(float(np.dot(candidate_ee_rotation[:, 1], preferred_axis)))
        rotation_change = float(np.linalg.norm(
            _orientation_error(current_rotation, candidate_ee_rotation)
        ))
        score = (-round(closing_axis_alignment, 10), rotation_change)
        if best is None or score < best[0]:
            best = (score, candidate_camera_rotation, candidate_ee_rotation, angle)
    assert best is not None
    _, target_camera_rotation, _, roll_angle = best

    target_camera = _transform(
        target - standoff_m * view_direction,
        target_camera_rotation,
    )
    try:
        target_ee = target_camera @ np.linalg.inv(ee_to_camera)
    except np.linalg.LinAlgError as exc:
        raise ValueError("wrist-camera mount transform is not invertible") from exc
    return target_ee[:3, 3], target_ee[:3, :3], {
        "wrist_target_world": target.tolist(),
        "standoff_m": float(standoff_m),
        "view_direction_world": view_direction.tolist(),
        "camera_roll_rad": float(roll_angle),
        "ee_to_wrist_camera": ee_to_camera.tolist(),
        "target_wrist_camera_to_world": target_camera.tolist(),
        "target_ee_to_world": target_ee.tolist(),
    }


def evaluate_wrist_target(
    joints: Any,
    arm: str,
    target_world: Any,
    ee_to_wrist_camera: Any,
    wrist_intrinsic: Any,
    wrist_image_shape: Sequence[int],
    *,
    standoff_m: float = WRIST_TARGET_STANDOFF_M,
    distance_tolerance_m: float = WRIST_TARGET_DISTANCE_TOLERANCE_M,
    center_tolerance_px: float = WRIST_TARGET_CENTER_TOLERANCE_PX,
) -> dict[str, Any]:
    """Measure final wrist-camera visibility, centering, and standoff."""
    height, width = int(wrist_image_shape[0]), int(wrist_image_shape[1])
    if height <= 1 or width <= 1:
        raise ValueError("wrist-camera image must have a valid height and width")
    intrinsic = np.asarray(wrist_intrinsic, dtype=np.float64).reshape(3, 3)
    mount = np.asarray(ee_to_wrist_camera, dtype=np.float64).reshape(4, 4)
    target = np.asarray(target_world, dtype=np.float64).reshape(3)
    if not (np.isfinite(intrinsic).all() and np.isfinite(mount).all() and np.isfinite(target).all()):
        raise ValueError("wrist-camera validation requires finite geometry")

    ee_position, ee_rotation = current_ee_pose(joints, arm)
    camera_to_world = _transform(ee_position, ee_rotation) @ mount
    try:
        target_camera = np.linalg.inv(camera_to_world) @ np.append(target, 1.0)
    except np.linalg.LinAlgError as exc:
        raise ValueError("final wrist-camera pose is not invertible") from exc
    depth = float(-target_camera[2])
    pixel = project_points([target], intrinsic, camera_to_world)[0]
    principal_point = intrinsic[:2, 2]
    center_error = (
        float(np.linalg.norm(pixel - principal_point))
        if np.isfinite(pixel).all() else float("inf")
    )
    distance = float(np.linalg.norm(target - camera_to_world[:3, 3]))
    in_frame = bool(
        np.isfinite(pixel).all()
        and 0.0 <= pixel[0] < width
        and 0.0 <= pixel[1] < height
    )
    visible = bool(depth > 0.0 and in_frame)
    reached = bool(
        visible
        and center_error <= center_tolerance_px
        and abs(distance - standoff_m) <= distance_tolerance_m
    )
    return {
        "final_wrist_camera_to_world": camera_to_world.tolist(),
        "wrist_target_pixel": pixel.tolist(),
        "wrist_principal_point": principal_point.tolist(),
        "wrist_center_error_px": center_error,
        "wrist_target_depth_m": depth,
        "wrist_target_distance_m": distance,
        "wrist_target_in_frame": in_frame,
        "wrist_target_visible": visible,
        "wrist_target_reached": reached,
        "wrist_target_standoff_m": float(standoff_m),
        "wrist_target_distance_tolerance_m": float(distance_tolerance_m),
        "wrist_target_center_tolerance_px": float(center_tolerance_px),
    }


def normalize_points(value: Any, coordinate_range: float = 255.0) -> np.ndarray:
    if isinstance(value, Mapping) and "x" in value and "y" in value:
        value = [value["x"], value["y"]]
    try:
        points = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return np.empty((0, 2))
    if not points.size or points.size % 2:
        return np.empty((0, 2))
    points = points.reshape(-1, 2)
    valid = np.isfinite(points).all(axis=1)
    valid &= (points >= 0).all(axis=1)
    valid &= (points <= coordinate_range).all(axis=1)
    return points[valid]


def project_points(world_points: np.ndarray, intrinsic: Any, camera_to_world: Any) -> np.ndarray:
    points = np.asarray(world_points, dtype=np.float64).reshape(-1, 3)
    camera = np.asarray(camera_to_world, dtype=np.float64).reshape(4, 4)
    homogeneous = np.c_[points, np.ones(len(points))]
    camera_points = homogeneous @ np.linalg.inv(camera).T
    depth = -camera_points[:, 2]
    pixels = np.full((len(points), 2), np.nan)
    matrix = np.asarray(intrinsic, dtype=np.float64).reshape(3, 3)
    valid = depth > 1e-6
    pixels[valid, 0] = matrix[0, 0] * camera_points[valid, 0] / depth[valid] + matrix[0, 2]
    pixels[valid, 1] = matrix[1, 2] - matrix[1, 1] * camera_points[valid, 1] / depth[valid]
    return pixels


def target_pixel_to_world(
    point: Sequence[float], depth_image: Any, intrinsic: Any, camera_to_world: Any,
    image_shape: Sequence[int], coordinate_range: float = 255.0, window_radius: int = 2,
    default_plane_height_m: float | None = None,
) -> np.ndarray:
    """Back-project a normalized image point, with an optional table-plane fallback."""
    depth = None if depth_image is None else np.asarray(depth_image, dtype=np.float64)
    if depth is not None and depth.ndim == 3 and depth.shape[0] == 1:
        depth = depth[0]
    elif depth is not None and depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth is not None and depth.ndim != 2:
        if default_plane_height_m is None:
            raise ValueError(f"head-camera depth must be a single-channel map, got {depth.shape}")
        depth = None
    height, width = int(image_shape[0]), int(image_shape[1])
    x = float(point[0]) * (width - 1) / coordinate_range
    y = float(point[1]) * (height - 1) / coordinate_range
    matrix = np.asarray(intrinsic, dtype=np.float64).reshape(3, 3)
    camera_to_world = np.asarray(camera_to_world, dtype=np.float64).reshape(4, 4)
    distance = None
    if depth is not None:
        depth_height, depth_width = depth.shape
        depth_x = x * (depth_width - 1) / max(width - 1, 1)
        depth_y = y * (depth_height - 1) / max(height - 1, 1)
        ix, iy = int(round(depth_x)), int(round(depth_y))
        values = depth[
            max(0, iy-window_radius):min(depth_height, iy+window_radius+1),
            max(0, ix-window_radius):min(depth_width, ix+window_radius+1),
        ]
        values = values[np.isfinite(values) & (values > 0)]
        if values.size:
            distance = float(np.median(values))
    ray_camera = np.asarray([
        (x - matrix[0, 2]) / matrix[0, 0],
        (matrix[1, 2] - y) / matrix[1, 1],
        -1.0,
    ], dtype=np.float64)
    if distance is not None:
        camera = np.r_[ray_camera * distance, 1.0]
        return (camera_to_world @ camera)[:3]
    if default_plane_height_m is None:
        raise ValueError("target has no valid local depth")
    plane_height = float(default_plane_height_m)
    if not np.isfinite(plane_height):
        raise ValueError("default table height must be finite")
    origin = camera_to_world[:3, 3]
    ray_world = camera_to_world[:3, :3] @ ray_camera
    if abs(ray_world[2]) < 1e-9:
        raise ValueError("target ray is parallel to the default table plane")
    scale = (plane_height - origin[2]) / ray_world[2]
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("default table plane is behind the head camera")
    return origin + scale * ray_world


def target_gripper_rotation(
    approach: str, grasp_axis_points: Any, depth_image: Any, intrinsic: Any,
    camera_to_world: Any, image_shape: Sequence[int], coordinate_range: float = 255.0,
) -> np.ndarray:
    """Build a world-frame gripper rotation from a VLM image-space grasp axis."""
    points = normalize_points(grasp_axis_points, coordinate_range)
    if len(points) != 2 or np.allclose(points[0], points[1]):
        raise ValueError("IK grasp axis requires two distinct image points")
    axis_world = target_pixel_to_world(
        points[1], depth_image, intrinsic, camera_to_world, image_shape, coordinate_range
    ) - target_pixel_to_world(
        points[0], depth_image, intrinsic, camera_to_world, image_shape, coordinate_range
    )
    camera_rotation = np.asarray(camera_to_world, dtype=np.float64).reshape(4, 4)[:3, :3]
    if approach == "top_down":
        approach_world = np.asarray([0.0, 0.0, -1.0])
    elif approach == "front":
        approach_world = -camera_rotation[:, 2]
        approach_world[2] = 0.0
    elif approach == "left_side":
        approach_world = camera_rotation[:, 0].copy()
        approach_world[2] = 0.0
    elif approach == "right_side":
        approach_world = -camera_rotation[:, 0].copy()
        approach_world[2] = 0.0
    else:
        raise ValueError(f"unsupported IK approach: {approach!r}")
    approach_norm = float(np.linalg.norm(approach_world))
    if approach_norm < 1e-6:
        raise ValueError(f"IK approach {approach!r} is degenerate in the world frame")
    tool_x = approach_world / approach_norm
    tool_y = axis_world - np.dot(axis_world, tool_x) * tool_x
    axis_norm = float(np.linalg.norm(tool_y))
    if axis_norm < 1e-6:
        raise ValueError("IK grasp axis is parallel to the approach direction")
    tool_y /= axis_norm
    tool_z = np.cross(tool_x, tool_y)
    tool_z /= np.linalg.norm(tool_z)
    tool_y = np.cross(tool_z, tool_x)
    return np.column_stack((tool_x, tool_y, tool_z))


def _orientation_error(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    delta = target @ current.T
    skew = 0.5 * np.asarray([
        delta[2, 1] - delta[1, 2],
        delta[0, 2] - delta[2, 0],
        delta[1, 0] - delta[0, 1],
    ])
    sine = float(np.linalg.norm(skew))
    cosine = float(np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0))
    angle = float(np.arctan2(sine, cosine))
    if sine > 1e-8:
        return skew * (angle / sine)
    if cosine > 0:
        return np.zeros(3)
    _, vectors = np.linalg.eigh((delta + np.eye(3)) / 2.0)
    axis = vectors[:, -1]
    return axis * np.pi


def _interpolate_rotation(current: np.ndarray, target: np.ndarray, ratio: float) -> np.ndarray:
    rotation_vector = _orientation_error(current, target)
    angle = float(np.linalg.norm(rotation_vector))
    if angle < 1e-12:
        return current.copy()
    return _axis_rotation(rotation_vector, ratio * angle) @ current


def ik_approach_chunk(
    current_joints: Any, current_grippers: Any, arm: str, target_world: Any,
    *, target_rotation: Any = None, steps: int = 10, approach_height_m: float = 0.0,
    max_joint_delta: float = 0.8, tolerance_m: float = 0.01,
    orientation_tolerance_rad: float = 0.2, orientation_weight: float = 0.2,
    grasp_axis_symmetric: bool = False, max_iterations: int = 100,
    max_waypoint_joint_delta: float = 0.35,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Follow Cartesian pose waypoints while preserving the other arm."""
    if arm not in {"left", "right"}:
        raise ValueError("IK approach requires an explicit left or right arm")
    joints = np.asarray(current_joints, dtype=np.float64).reshape(-1)
    grippers = np.asarray(current_grippers, dtype=np.float64).reshape(-1)
    if (joints.size < 12 or grippers.size < 2 or steps <= 0
            or max_iterations <= 0 or max_joint_delta <= 0
            or max_waypoint_joint_delta <= 0):
        raise ValueError("IK approach requires valid state, steps, and iterations")
    arm_slice = slice(0, 6) if arm == "left" else slice(6, 12)
    base = _LEFT_BASE if arm == "left" else _RIGHT_BASE
    start = joints[arm_slice].copy()
    target = np.asarray(target_world, dtype=np.float64).reshape(3).copy()
    target[2] += approach_height_m
    desired_rotation = (
        None if target_rotation is None
        else np.asarray(target_rotation, dtype=np.float64).reshape(3, 3)
    )
    grasp_axis_flipped = False
    input_rotation = None if desired_rotation is None else desired_rotation.copy()
    start_position, start_rotation = x5_gripper_pose(start, base)
    if desired_rotation is not None and grasp_axis_symmetric:
        alternate = desired_rotation @ np.diag([1.0, -1.0, -1.0])
        if (np.linalg.norm(_orientation_error(start_rotation, alternate))
                < np.linalg.norm(_orientation_error(start_rotation, desired_rotation))):
            desired_rotation = alternate
            grasp_axis_flipped = True
    damping, epsilon = 1e-3, 1e-4
    rng = np.random.default_rng(0)
    waypoint_tolerance = tolerance_m
    waypoint_orientation_tolerance = orientation_tolerance_rad
    previous_solution = start.copy()
    solutions = []
    total_converged = total_iterations = 0
    largest_waypoint_delta = 0.0
    waypoint_failure = None

    for waypoint_index, ratio in enumerate(np.linspace(1.0 / steps, 1.0, steps), start=1):
        waypoint_position = start_position + ratio * (target - start_position)
        waypoint_rotation = (
            None if desired_rotation is None
            else _interpolate_rotation(start_rotation, desired_rotation, ratio)
        )
        seeds = [previous_solution]
        seeds.extend(previous_solution + rng.normal(0.0, 0.08, size=(8, 6)))
        converged_solutions = []
        nearest_error = nearest_orientation_error = float("inf")
        for seed in seeds:
            solution = np.clip(np.asarray(seed, dtype=np.float64), -np.pi, np.pi)
            for iteration in range(max_iterations):
                current, current_rotation = x5_gripper_pose(solution, base)
                position_error = waypoint_position - current
                error_norm = float(np.linalg.norm(position_error))
                rotation_error = (
                    np.zeros(3) if waypoint_rotation is None
                    else _orientation_error(current_rotation, waypoint_rotation)
                )
                orientation_error = float(np.linalg.norm(rotation_error))
                if error_norm < nearest_error:
                    nearest_error = error_norm
                    nearest_orientation_error = orientation_error
                if (error_norm <= waypoint_tolerance and (
                        waypoint_rotation is None
                        or orientation_error <= waypoint_orientation_tolerance)):
                    candidate_delta = (
                        solution - previous_solution + np.pi
                    ) % (2*np.pi) - np.pi
                    if np.max(np.abs(candidate_delta)) <= max_waypoint_joint_delta:
                        converged_solutions.append((
                            float(np.max(np.abs(candidate_delta))),
                            float(np.linalg.norm(candidate_delta)),
                            error_norm / waypoint_tolerance + (
                                0.0 if waypoint_rotation is None else
                                orientation_error / waypoint_orientation_tolerance
                            ),
                            previous_solution + candidate_delta,
                            iteration + 1,
                        ))
                    break
                rows = 3 if waypoint_rotation is None else 6
                jacobian = np.empty((rows, 6), dtype=np.float64)
                for joint in range(6):
                    perturbed = solution.copy()
                    perturbed[joint] += epsilon
                    perturbed_position, perturbed_rotation = x5_gripper_pose(perturbed, base)
                    jacobian[:3, joint] = (perturbed_position - current) / epsilon
                    if waypoint_rotation is not None:
                        relative = perturbed_rotation @ current_rotation.T
                        jacobian[3:, joint] = np.asarray([
                            relative[2, 1] - relative[1, 2],
                            relative[0, 2] - relative[2, 0],
                            relative[1, 0] - relative[0, 1],
                        ]) / (2*epsilon)
                residual = position_error
                if waypoint_rotation is not None:
                    jacobian[3:] *= orientation_weight
                    residual = np.concatenate((
                        position_error, orientation_weight * rotation_error,
                    ))
                update = jacobian.T @ np.linalg.solve(
                    jacobian @ jacobian.T + damping*np.eye(rows), residual
                )
                solution = np.clip(solution + np.clip(update, -0.15, 0.15), -np.pi, np.pi)
        if not converged_solutions:
            if not solutions:
                raise ValueError(
                    f"IK waypoint {waypoint_index}/{steps} did not converge without a joint "
                    f"jump: position error {nearest_error:.4f} m, orientation error "
                    f"{nearest_orientation_error:.4f} rad"
                )
            waypoint_failure = {
                "waypoint": waypoint_index,
                "position_error_m": nearest_error,
                "orientation_error_rad": nearest_orientation_error,
            }
            solutions.extend([previous_solution.copy()] * (steps - len(solutions)))
            break
        selected = min(converged_solutions, key=lambda item: item[:3])
        waypoint_delta, _, _, solution, iterations = selected
        total_converged += len(converged_solutions)
        total_iterations += iterations
        largest_waypoint_delta = max(largest_waypoint_delta, waypoint_delta)
        solutions.append(solution.copy())
        previous_solution = solution

    solver_solution = solutions[-1]
    solver_position, solver_rotation = x5_gripper_pose(solver_solution, base)
    solver_error = float(np.linalg.norm(target - solver_position))
    solver_orientation_error = (
        0.0 if desired_rotation is None
        else float(np.linalg.norm(_orientation_error(solver_rotation, desired_rotation)))
    )
    unbounded_delta = (solver_solution - start + np.pi) % (2*np.pi) - np.pi
    largest_delta = float(np.max(np.abs(unbounded_delta)))
    path_scale = min(1.0, max_joint_delta / max(largest_delta, 1e-12))
    limited_by_chunk = path_scale < 1.0
    if limited_by_chunk:
        solutions = [
            start + path_scale * ((solution - start + np.pi) % (2*np.pi) - np.pi)
            for solution in solutions
        ]
    execution_solution = solutions[-1]
    joint_delta = execution_solution - start
    final_position, final_rotation = x5_gripper_pose(execution_solution, base)
    final_error = float(np.linalg.norm(target - final_position))
    final_orientation_error = (
        0.0 if desired_rotation is None
        else float(np.linalg.norm(_orientation_error(final_rotation, desired_rotation)))
    )
    result = np.empty((steps, 14), dtype=np.float64)
    for index, solution in enumerate(solutions):
        pose = joints.copy()
        pose[arm_slice] = solution
        result[index] = np.concatenate((pose[:6], grippers[:1], pose[6:12], grippers[1:2]))
    return result, {
        "arm": arm, "target_world": target.tolist(), "position_error_m": final_error,
        "solver_position_error_m": solver_error,
        "orientation_error_rad": final_orientation_error,
        "solver_orientation_error_rad": solver_orientation_error,
        "grasp_axis_flipped": grasp_axis_flipped,
        "target_rotation": None if desired_rotation is None else desired_rotation.tolist(),
        "input_target_rotation": None if input_rotation is None else input_rotation.tolist(),
        "target_reached": (
            final_error <= tolerance_m
            and (desired_rotation is None or final_orientation_error <= orientation_tolerance_rad)
        ),
        "approach_height_m": approach_height_m, "joint_delta": joint_delta.tolist(),
        "iterations": total_iterations,
        "converged_solution_count": total_converged,
        "selected_max_joint_delta": largest_delta,
        "max_waypoint_joint_delta": largest_waypoint_delta,
        "limited_by_chunk": limited_by_chunk,
        "waypoint_failure": waypoint_failure,
    }


def _geometry_scores(
    chunks: list[Any], points: Any, arm: str | None, intrinsic: Any, camera_to_world: Any,
    image_shape: Sequence[int], coordinate_range: float, trajectory_steps: int,
    execution_steps: int | None = None,
) -> tuple[list[float], list[dict[str, Any]]]:
    targets = normalize_points(points, coordinate_range)
    if not len(targets):
        raise ValueError("target coordinates are empty")
    height, width = int(image_shape[0]), int(image_shape[1])
    if height <= 1 or width <= 1:
        raise ValueError("invalid camera image shape")
    arms = (arm,) if arm in {"left", "right"} else ("left", "right")
    scale = np.asarray((coordinate_range / (width - 1), coordinate_range / (height - 1)))
    scores, details = [], []
    for chunk in chunks:
        compared_steps = action_steps(chunk)[:trajectory_steps]
        trajectory = fk_trajectory(compared_steps)
        best = float("inf")
        projected: dict[str, list[list[float]]] = {}
        for name in arms:
            pixels = project_points(trajectory[name], intrinsic, camera_to_world) * scale
            projected[name] = pixels.tolist()
            valid = np.isfinite(pixels).all(axis=1)
            if valid.any():
                distances = np.linalg.norm(pixels[valid, None] - targets[None], axis=-1).min(axis=1)
                score = float(distances.min())
                if execution_steps is not None:
                    endpoint = pixels[min(execution_steps, len(pixels)) - 1]
                    score += 0.5 * float(np.linalg.norm(endpoint - targets, axis=1).min())
                best = min(best, score)
        scores.append(best)
        details.append({
            "projected": projected,
            "trajectory": {key: value.tolist() for key, value in trajectory.items()},
            "steps_compared": len(compared_steps),
        })
    return scores, details


def _target_distance_from_gripper(
    points: Any,
    arm: str | None,
    current_joints: Any,
    depth_image: Any,
    intrinsic: Any,
    camera_to_world: Any,
    image_shape: Sequence[int],
    coordinate_range: float,
    default_plane_height_m: float | None,
) -> float | None:
    targets = normalize_points(points, coordinate_range)
    joints = np.asarray(current_joints, dtype=np.float64).reshape(-1)
    if not len(targets) or joints.size < 12:
        return None
    if intrinsic is None or camera_to_world is None:
        return None
    arm_joints = {"left": joints[:6], "right": joints[6:12]}
    arms = (arm,) if arm in arm_joints else ("left", "right")
    target_world = []
    for point in targets:
        try:
            target_world.append(target_pixel_to_world(
                point, depth_image, intrinsic, camera_to_world, image_shape,
                coordinate_range,
                default_plane_height_m=default_plane_height_m,
            ))
        except (TypeError, ValueError, np.linalg.LinAlgError):
            continue
    if not target_world:
        return None
    target_world = np.asarray(target_world, dtype=np.float64)
    distances = []
    for name in arms:
        position, _ = x5_gripper_pose(
            arm_joints[name], _LEFT_BASE if name == "left" else _RIGHT_BASE,
        )
        if np.isfinite(position).all():
            distances.append(float(np.linalg.norm(target_world - position, axis=1).min()))
    return min(distances) if distances else None


def select_candidate(
    chunks: list[Any], *, ordered: bool, observe: bool = False, points: Any = None,
    arm: str | None = None, intrinsic: Any = None, camera_to_world: Any = None,
    image_shape: Sequence[int] | None = None, coordinate_range: float = 255.0,
    current_joints: Any = None,
    depth_image: Any = None,
    default_plane_height_m: float | None = ROBODOJO_DEFAULT_TABLE_HEIGHT_M,
    far_distance_m: float = DEFAULT_FAR_DISTANCE_M,
    near_trajectory_steps: int = DEFAULT_NEAR_TRAJECTORY_STEPS,
    far_trajectory_steps: int = DEFAULT_FAR_TRAJECTORY_STEPS,
    similarity_ratio: float = 0.1, rng: random.Random | None = None,
    execution_steps: int | None = None,
) -> tuple[int, dict[str, Any]]:
    """Select an index and diagnostics; never creates or edits an action."""
    if not chunks:
        raise ValueError("candidate chunks must be non-empty")
    chooser = rng or random
    if observe:
        scores = []
        trajectories = []
        for chunk in chunks:
            trajectory = fk_trajectory(chunk)
            trajectories.append({key: value.tolist() for key, value in trajectory.items()})
            joints = np.concatenate((trajectory["left_joints"], trajectory["right_joints"]), axis=1)
            scores.append(float(np.max(np.abs(joints), axis=1).max()) if joints.size else float("inf"))
        index = int(np.argmin(scores))
        return index, {
            "selection_reason": "observe_zero_pose",
            "zero_pose_scores": scores,
            "candidate_pool": [index],
            "trajectories": trajectories,
        }
    fallback_reason = "unordered_or_missing_geometry"
    if ordered and points is not None and intrinsic is not None and camera_to_world is not None and image_shape is not None:
        try:
            target_distance = _target_distance_from_gripper(
                points, arm, current_joints, depth_image, intrinsic, camera_to_world,
                image_shape, coordinate_range,
                default_plane_height_m,
            )
            distance_class = (
                "near"
                if target_distance is not None and target_distance <= far_distance_m
                else "far" if target_distance is not None else "unavailable"
            )
            trajectory_steps = (
                near_trajectory_steps if distance_class == "near" else far_trajectory_steps
            )
            scores, details = _geometry_scores(
                chunks, points, arm, intrinsic, camera_to_world, image_shape,
                coordinate_range, trajectory_steps, execution_steps,
            )
            finite = [index for index, score in enumerate(scores) if np.isfinite(score)]
            if finite:
                best = min(scores[index] for index in finite)
                band = max(abs(best), 1.0) * max(0.0, similarity_ratio)
                pool = [index for index in finite if scores[index] <= best + band]
                index = int(chooser.choice(pool))
                return index, {
                    "selection_reason": "ordered_geometry_random",
                    "geometry_scores": scores,
                    "target_distance_m": target_distance,
                    "distance_class": distance_class,
                    "far_distance_m": far_distance_m,
                    "trajectory_steps_requested": trajectory_steps,
                    "trajectory_steps_compared": [item["steps_compared"] for item in details],
                    "candidate_pool": pool,
                    "projected": details[index]["projected"],
                    "projected_trajectories": [item["projected"] for item in details],
                    "trajectories": [item["trajectory"] for item in details],
                }
            fallback_reason = "no_finite_geometry"
        except (TypeError, ValueError, np.linalg.LinAlgError) as exc:
            fallback_reason = f"geometry_unavailable:{type(exc).__name__}"
    index = int(chooser.choice(list(range(len(chunks)))))
    trajectories = []
    try:
        trajectories = [
            {key: value.tolist() for key, value in fk_trajectory(chunk).items()}
            for chunk in chunks
        ]
    except (TypeError, ValueError, np.linalg.LinAlgError):
        trajectories = []
    return index, {
        "selection_reason": "random_all",
        "fallback_reason": fallback_reason,
        "candidate_pool": list(range(len(chunks))),
        "trajectories": trajectories,
        "geometry_scores": [],
        "projected_trajectories": [],
    }


def group_trajectory_endpoints(projected: list[dict], arm: str = "", radius: float = 15.0) -> list[dict]:
    """Complete-link groups: no pair of arm endpoints exceeds radius."""
    arms = [arm] if arm in {"left", "right"} else ["left", "right"]
    groups, vectors = [], []
    for index, trajectory in enumerate(projected):
        endpoint = {name: np.asarray(trajectory[name][-1]).tolist()
                    for name in arms if name in trajectory and len(trajectory[name])}
        if len(endpoint) != len(arms):
            raise ValueError(f"Missing endpoint for candidate {index}")
        vector = np.asarray([endpoint[name] for name in arms], dtype=float)
        if not np.isfinite(vector).all():
            raise ValueError(f"Non-finite endpoint for candidate {index}")
        for group in groups:
            if all(float(np.linalg.norm(vector - vectors[member], axis=1).max()) <= radius
                   for member in group["candidate_indices"]):
                group["candidate_indices"].append(index)
                group["endpoints"].append(endpoint)
                break
        else:
            groups.append({"group_id": len(groups), "candidate_indices": [index], "endpoints": [endpoint]})
        vectors.append(vector)
    return groups


def average_consistent_chunks(
    chunks: list[Any], candidate_pool: Sequence[int],
    rms_threshold: float = 0.05, max_difference: float = 0.2,
    *, compute_average: bool = True,
) -> tuple[Any, list[int], float | None]:
    """Find a consistent subset and optionally average its action sequences.

    Compare full sequences on a normalized progress grid for unequal lengths.
    When ``compute_average`` is false, the first return value is ``None`` and
    no synthetic mean trajectory is constructed.
    """
    pool = sorted(set(int(index) for index in candidate_pool if 0 <= int(index) < len(chunks)))
    if not pool:
        raise ValueError("candidate pool must be non-empty")
    steps = {index: action_steps(chunks[index]) for index in pool}
    vectors, signatures = {}, {}
    for index in pool:
        rows, schemas = [], []
        for step in steps[index]:
            if isinstance(step, Mapping):
                fields, schema = [], []
                for key in sorted(step):
                    value = np.asarray(step[key])
                    numeric = np.issubdtype(value.dtype, np.number)
                    schema.append((key, value.shape, None if numeric else repr(step[key])))
                    if numeric:
                        fields.append(value.astype(float).reshape(-1))
                rows.append(np.concatenate(fields) if fields else np.empty(0))
                schemas.append(tuple(schema))
            else:
                value = np.asarray(step, dtype=float)
                rows.append(value.reshape(-1))
                schemas.append(value.shape)
        if not rows or not len(rows[0]) or any(schema != schemas[0] for schema in schemas):
            raise ValueError("Action sequence must have a stable non-empty schema")
        array = np.asarray(rows, dtype=float)
        if not np.isfinite(array).all():
            raise ValueError("Action sequence contains non-finite values")
        vectors[index], signatures[index] = array, schemas[0]
    distances, compatible = {}, {}
    for left, right in combinations(pool, 2):
        a, b = vectors[left], vectors[right]
        if signatures[left] != signatures[right] or a.shape[1] != b.shape[1]:
            distances[left, right], compatible[left, right] = float("inf"), False
            continue
        grid = np.linspace(0, 1, max(len(a), len(b)))
        a = np.asarray([np.interp(grid, np.linspace(0, 1, len(a)), column) for column in a.T]).T
        b = np.asarray([np.interp(grid, np.linspace(0, 1, len(b)), column) for column in b.T]).T
        delta = a - b
        distance = float(np.sqrt(np.mean(delta ** 2)))
        distances[left, right] = distance
        compatible[left, right] = distance <= rms_threshold and float(np.abs(delta).max()) <= max_difference
    subset, mean_distance = (pool[0],), None
    # At most eleven candidates: exhaustive search avoids order-dependent chaining.
    for size in range(len(pool), 1, -1):
        valid = [members for members in combinations(pool, size)
                 if all(compatible[pair] for pair in combinations(members, 2))]
        if valid:
            subset = min(valid, key=lambda members: np.mean([distances[pair] for pair in combinations(members, 2)]))
            mean_distance = float(np.mean([distances[pair] for pair in combinations(subset, 2)]))
            break
    if len(subset) == 1:
        return (chunks[subset[0]] if compute_average else None), list(subset), None
    if not compute_average:
        return None, list(subset), mean_distance
    length = min(len(steps[index]) for index in subset)
    if isinstance(steps[subset[0]][0], Mapping):
        averaged = []
        for timestep in range(length):
            merged = dict(steps[subset[0]][timestep])
            for key, value in merged.items():
                if np.issubdtype(np.asarray(value).dtype, np.number):
                    merged[key] = np.mean([np.asarray(steps[index][timestep][key], dtype=float)
                                           for index in subset], axis=0)
            averaged.append(merged)
        return averaged, list(subset), mean_distance
    averaged = np.mean([np.asarray(steps[index][:length], dtype=float) for index in subset], axis=0)
    return averaged, list(subset), mean_distance


def average_most_similar_chunks(
    chunks: list[Any], candidate_pool: Sequence[int],
) -> tuple[Any, list[int], float | None]:
    """Average the closest pair of already-filtered action chunks.

    The returned action keeps the common prefix length of the pair. This is
    intentionally done after geometric filtering so averaging cannot combine
    trajectories aimed at different targets.
    """
    pool = [int(index) for index in candidate_pool if 0 <= int(index) < len(chunks)]
    if not pool:
        raise ValueError("candidate pool must be non-empty")
    if len(pool) == 1:
        return chunks[pool[0]], pool, None

    def vectors(chunk: Any) -> np.ndarray:
        steps = action_steps(chunk)
        rows = []
        for step in steps:
            if isinstance(step, Mapping):
                values = []
                for key in sorted(step):
                    value = np.asarray(step[key])
                    if np.issubdtype(value.dtype, np.number):
                        values.append(value.reshape(-1).astype(np.float64))
                rows.append(np.concatenate(values) if values else np.empty(0))
            else:
                value = np.asarray(step)
                rows.append(value.reshape(-1).astype(np.float64))
        if not rows:
            return np.empty((0, 0))
        width = min(len(row) for row in rows)
        return np.asarray([row[:width] for row in rows], dtype=np.float64)

    representations = {index: vectors(chunks[index]) for index in pool}
    best_pair = (pool[0], pool[1])
    best_distance = float("inf")
    for left_position, left_index in enumerate(pool):
        for right_index in pool[left_position + 1:]:
            left = representations[left_index]
            right = representations[right_index]
            steps = min(len(left), len(right))
            width = min(left.shape[1] if left.ndim == 2 else 0, right.shape[1] if right.ndim == 2 else 0)
            distance = float("inf") if not steps or not width else float(
                np.sqrt(np.mean((left[:steps, :width] - right[:steps, :width]) ** 2))
            )
            if distance < best_distance:
                best_distance = distance
                best_pair = (left_index, right_index)

    left_index, right_index = best_pair
    left_steps = action_steps(chunks[left_index])
    right_steps = action_steps(chunks[right_index])
    steps = min(len(left_steps), len(right_steps))
    left_chunk, right_chunk = left_steps[:steps], right_steps[:steps]
    if isinstance(left_chunk[0], Mapping) and isinstance(right_chunk[0], Mapping):
        averaged = []
        for left_step, right_step in zip(left_chunk, right_chunk, strict=True):
            merged = dict(left_step)
            for key in set(left_step) & set(right_step):
                try:
                    left_value = np.asarray(left_step[key])
                    right_value = np.asarray(right_step[key])
                    if left_value.shape == right_value.shape and np.issubdtype(left_value.dtype, np.number):
                        merged[key] = ((left_value.astype(np.float64) + right_value.astype(np.float64)) / 2).astype(
                            np.result_type(left_value.dtype, right_value.dtype, np.float32), copy=False
                        )
                except (TypeError, ValueError):
                    continue
            averaged.append(merged)
        return averaged, [left_index, right_index], best_distance
    left_array = np.asarray(left_chunk)
    right_array = np.asarray(right_chunk)
    if left_array.shape != right_array.shape:
        return chunks[left_index], [left_index], best_distance
    return ((left_array.astype(np.float64) + right_array.astype(np.float64)) / 2).astype(
        np.result_type(left_array.dtype, right_array.dtype), copy=False
    ), [left_index, right_index], best_distance


__all__ = [
    "DEFAULT_FAR_DISTANCE_M",
    "DEFAULT_FAR_TRAJECTORY_STEPS",
    "DEFAULT_NEAR_TRAJECTORY_STEPS",
    "action_steps",
    "average_most_similar_chunks",
    "average_consistent_chunks",
    "current_ee_pose",
    "fk_trajectory",
    "group_trajectory_endpoints",
    "ik_approach_chunk",
    "normalize_points",
    "project_points",
    "target_gripper_rotation",
    "target_pixel_to_world",
    "select_candidate",
    "x5_gripper_center",
    "x5_gripper_pose",
]
