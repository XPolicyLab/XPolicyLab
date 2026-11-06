---
name: rigid_pose_transforms
description: "Converts XYZ-plus-WXYZ poses to 4x4 homogeneous matrices and back, and composes a target pose as T_B @ inv(T_A) @ T_C for real_pose, set_pose, and relative_real_pose. The helpers are stateless and retain the original NumPy/SciPy numerical operations without explicit validation."
---

# Rigid pose transforms

## Function

Converts XYZ-plus-WXYZ poses to 4x4 homogeneous matrices and back, and composes a target pose as T_B @ inv(T_A) @ T_C for real_pose, set_pose, and relative_real_pose. The helpers are stateless and retain the original NumPy/SciPy numerical operations without explicit validation.

## Inputs

- pose: seven values ordered translation_x, translation_y, translation_z, qw, qx, qy, qz
- matrix: 4x4 homogeneous transform
- real_pose, set_pose, relative_real_pose: poses in the same XYZ-plus-WXYZ ordering

## Outputs

- 4x4 homogeneous matrix with rotation from the quaternion and translation in the last column
- pose ndarray ordered x, y, z, qw, qx, qy, qz
- relative_set_pose computed as T_B @ inv(T_A) @ T_C

## Callable interface

Implementation: [implementation.py](implementation.py).

- `pose_to_matrix(pose)`
- `matrix_to_pose(matrix)`
- `calculate_target_pose(real_pose, set_pose, relative_real_pose)`

## Constraints

- No shared planner, cache, command clock, or task class is held by these helpers.
- Uses scipy.spatial.transform.Rotation and NumPy; invalid shapes or non-finite values are not checked here.
- Caller-owned state and clocks remain local.
