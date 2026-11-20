---
name: mature_pose_geometry
description: "Validates and composes finite SE(3) poses and scalar-first quaternions, converts rotations, wraps yaw, selects a nearby revolute-joint IK branch, and reports position/full-orientation/yaw errors. It preserves the pinned mature conveyor rules' numeric normalization and arithmetic order."
---

# Mature pose geometry

## Function

Validates and composes finite SE(3) poses and scalar-first quaternions, converts rotations, wraps yaw, selects a nearby revolute-joint IK branch, and reports position/full-orientation/yaw errors. It preserves the pinned mature conveyor rules' numeric normalization and arithmetic order.

## Inputs

- quaternion, pose, matrix, vector, yaw: finite numeric arrays/scalars with required shapes
- solutions [N,D], current [D], limits [D,2], optional success [N] for closest_joint_solution
- actual_pose and target_pose for pose_errors; parent and child poses for compose_pose
- tip_xyz, tool_quaternion, finger_offset for tool_pose_for_tip

## Outputs

- normalized quaternion, rotation matrix, composed/inverted pose, or tool pose
- wrapped yaw in [-pi, pi), or closest_joint_solution [D] / None
- pose_errors dict with position_m, xy_m, z_m, orientation_rad/deg, yaw_rad/deg, with yaw possibly None

## Callable interface

Implementation: [implementation.py](implementation.py).

- `normalize_quat_wxyz(quaternion)`
- `validate_pose(pose)`
- `quat_mul_wxyz(left, right)`
- `quat_inverse_wxyz(quaternion)`
- `quat_to_mat_wxyz(quaternion)`
- `mat_to_quat_wxyz(matrix)`
- `wrap_yaw(angle)`
- `yaw_of_quaternion(quaternion)`
- `rotate_vec_by_quat(vector, quaternion)`
- `compose_pose(parent, child)`
- `inverse_pose(pose)`
- `tool_pose_for_tip(tip_xyz, tool_quaternion, finger_offset)`
- `closest_joint_solution(solutions, current, limits, success=None)`
- `quaternion_angle(quaternion, target_quaternion)`
- `pose_errors(actual_pose, target_pose)`

## Constraints

- Rejects wrong shapes, nonfinite values, zero quaternions, and non-orthonormal/reflected matrices; inputs are copied.
- yaw_of_quaternion raises ValueError when body-local X is vertical; pose_errors keeps full orientation error and sets yaw to None.
- closest_joint_solution only handles revolute columns, maps each joint to the nearest 2*pi-equivalent inside limits, ranks by max absolute then Euclidean displacement, and does not infer collision safety or impose a motion step cap.
- A 1e-10 rad tolerance only handles limit residue and returned values are clamped; no environment/task/scorer state is read.
