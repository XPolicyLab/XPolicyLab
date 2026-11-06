---
name: native_rigid_pose
description: "Provides pure scalar-first quaternion and 7D pose math: validated normalization, composition, inversion, conversion, interpolation, top-down quaternion construction, and pose comparison. Validated and unchecked motion-algebra helpers are exposed under separate explicit names."
---

# Native rigid pose

## Function

Provides pure scalar-first quaternion and 7D pose math: validated normalization, composition, inversion, conversion, interpolation, top-down quaternion construction, and pose comparison. Validated and unchecked motion-algebra helpers are exposed under separate explicit names.

## Inputs

- pose: finite 7D XYZ/wxyz array for validated helpers
- quaternion: scalar-first 4-vector for normalization/composition/inversion/conversion
- start, goal, fraction for interpolate_pose; closing_axis for top_down_quat
- left, right quaternions for validated_quat_mul_wxyz; a, b poses for compose_pose/inverse_pose

## Outputs

- normalized float64 7D pose or unit quaternion from validated helpers
- float32 quaternions/poses from unchecked quat_mul_wxyz, quat_inverse_wxyz, rotate_vec_by_quat, compose_pose, inverse_pose, interpolate_pose and top_down_quat
- quaternion_angle_deg returns shortest scalar-first quaternion angle in degrees
- _same_quaternion_rotation returns bool

## Callable interface

Implementation: [implementation.py](implementation.py).

- `as_numpy(value: Any)`
- `quat_to_mat_wxyz(quat: Iterable[float])`
- `quat_mul_wxyz(q1: Iterable[float], q2: Iterable[float])`
- `quaternion_angle_deg(q_a: Iterable[float], q_b: Iterable[float])`
- `normalize_quat_wxyz(quaternion)`
- `validate_pose(pose)`
- `validated_quat_mul_wxyz(left, right)`
- `validated_quat_inverse_wxyz(quaternion)`
- `validated_quat_to_mat_wxyz(quaternion)`
- `quat_inverse_wxyz(quat: np.ndarray)`
- `rotate_vec_by_quat(vector: np.ndarray, quat: np.ndarray)`
- `compose_pose(a: np.ndarray, b: np.ndarray)`
- `inverse_pose(pose: np.ndarray)`
- `top_down_quat(closing_axis: np.ndarray, flip: bool=False)`
- `transformed_vertices(pose: np.ndarray, vertices: np.ndarray)`
- `interpolate_pose(start: np.ndarray, goal: np.ndarray, fraction: float)`

## Constraints

- normalize_quat_wxyz rejects zero/NaN/Inf, scales before norm to avoid overflow, and makes the first numerically significant component positive
- validated_quat_mul_wxyz and validated_quat_inverse_wxyz return normalized unit quaternions; quat_inverse_wxyz returns [1,0,0,0] when norm_sq < 1e-12 and quat_to_mat_wxyz returns identity when norm < 1e-12
- top_down_quat raises ValueError when closing_axis has no horizontal component with norm < 1e-6
- _same_quaternion_rotation checks sign/normalization representation only and applies no angular policy gate
