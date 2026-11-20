---
name: rigid_pose_math
description: "Pure pose and event-value helpers for rigid-object manipulation: convert and multiply scalar-first wxyz quaternions, compute shortest quaternion angle, serialize values, compose/invert seven-value poses, build a grasp wrist pose from a static tool offset, and interpolate poses."
---

# Rigid pose math

## Function

Pure pose and event-value helpers for rigid-object manipulation: convert and multiply scalar-first wxyz quaternions, compute shortest quaternion angle, serialize values, compose/invert seven-value poses, build a grasp wrist pose from a static tool offset, and interpolate poses.

## Inputs

- scalar-first (w,x,y,z) quaternions and seven-value poses
- caller-owned numeric values for jsonable and grasp_pose inputs local_point, tool_quat, finger_offset
- pose pairs for quaternion_angle_deg and interpolate_pose
- numeric duck-typed values accepted by as_numpy

## Outputs

- quat_to_mat_wxyz returns a 3x3 float64 rotation; mat_to_quat_wxyz returns a canonical float32 wxyz quaternion with w>=0
- compose_pose and inverse_pose return float32 seven-value poses; grasp_pose returns a float32 wrist pose
- quaternion_angle_deg returns the shortest angle in degrees; jsonable returns JSON-compatible Python values
- interpolate_pose returns a pose

## Callable interface

Implementation: [implementation.py](implementation.py).

- `quat_to_mat_wxyz(quat: Iterable[float])`
- `mat_to_quat_wxyz(matrix: np.ndarray)`
- `quat_mul_wxyz(q1: Iterable[float], q2: Iterable[float])`
- `quaternion_angle_deg(q_a: Iterable[float], q_b: Iterable[float])`
- `jsonable(value: Any)`
- `compose_pose(a: np.ndarray, b: np.ndarray)`
- `inverse_pose(pose: np.ndarray)`
- `grasp_pose(key_pose: np.ndarray, local_point: np.ndarray, tool_quat: np.ndarray, finger_offset: float)`

## Constraints

- Original float32 conversion, float64 rotation math, normalization, shortest-arc interpolation, error behavior, and copy semantics are preserved
- as_numpy only supports original numeric duck-typing and never obtains a robot, environment, or observation
- grasp_pose uses the supplied static public tool offset and does not measure finger positions or certify grasp retention
- Pose sources must be public RGB fits or explicitly labeled own-plan estimates, not ground-truth object state
