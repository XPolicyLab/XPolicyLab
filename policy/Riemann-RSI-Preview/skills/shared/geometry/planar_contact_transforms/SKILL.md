---
name: planar_contact_transforms
description: "Provides canonical stateless scalar-first quaternion and 7D pose helpers for validated conversion/composition, planar contact transforms, bounded XY/yaw stepping, closest revolute-joint IK branch selection, and pose-error reporting. Caller-owned state and clocks remain local."
---

# Planar contact transforms

## Function

Provides canonical stateless scalar-first quaternion and 7D pose helpers for validated conversion/composition, planar contact transforms, bounded XY/yaw stepping, closest revolute-joint IK branch selection, and pose-error reporting. Caller-owned state and clocks remain local.

## Inputs

- pose/quaternion arrays for U27/U74 helpers; poses are 7D XYZ/wxyz and quaternions are scalar-first
- U74_planar_step current_pose, goal_pose, initial_pose plus max_translation in meters and max_yaw in radians
- U74_closest_joint_solution solutions [N,D], current [D], limits [D,2], optional success [N] bool/0-1
- U74_capture_contact_transform takes measured world object_pose and tool_pose; U53_tool_pose_for_tip takes tip_xyz, tool_quat, finger_offset

## Outputs

- normalized float64 poses/quaternions and rotation matrices from validated U74 helpers
- U74_planar_step returns a 7D pose at initial Z with bounded XY displacement and shortest wrapped yaw step
- U74_closest_joint_solution returns a nearest 2*pi-equivalent revolute branch [D] or None
- U74_capture_contact_transform returns inv(T_world_tool) @ T_world_object; U74_pose_errors returns position_m, xy_m, z_m, orientation_rad/deg and yaw_rad/deg with yaw None when undefined

## Callable interface

Implementation: [implementation.py](implementation.py).

- `U27_as_numpy(value)`
- `U27_mat_to_quat_wxyz(matrix)`
- `U27_quat_mul_wxyz(q1, q2)`
- `U27_quaternion_angle_deg(q_a, q_b)`
- `U74_array(value, shape, name)`
- `U74_scalar(value, name, *, positive=False, nonnegative=False)`
- `U74_normalize_quat_wxyz(quaternion)`
- `U74_validate_pose(pose)`
- `U74_quat_mul_wxyz(left, right)`
- `U74_quat_inverse_wxyz(quaternion)`
- `U74_quat_to_mat_wxyz(quaternion)`
- `U74_quat_axis_angle_wxyz(axis, angle)`
- `U74_yaw_quat(angle)`
- `U74_wrap_yaw(angle)`
- `U74_yaw_of_quaternion(quaternion)`
- `U74_rotate_vec_by_quat(vector, quaternion)`
- `U74_compose_pose(parent, child)`
- `U74_inverse_pose(pose)`
- `U74_capture_contact_transform(object_pose, tool_pose)`
- `U74_tool_pose_for_object(desired_object_pose, object_in_tool)`
- `U74_planar_step(current_pose, goal_pose, initial_pose, *, max_translation=0.02, max_yaw=math.radians(8))`
- `U74_closest_joint_solution(solutions, current, limits, success=None)`
- `U74_quaternion_angle(quaternion, target_quaternion)`
- `U74_pose_errors(actual_pose, target_pose)`
- `U53_quat_inverse_wxyz(quat)`
- `U53_rotate_vec_by_quat(vector, quat)`
- `U53_tool_pose_for_tip(tip_xyz, tool_quat, finger_offset)`
- `U51_yaw_delta(source, target)`

## Constraints

- Validation rejects zero/NaN/Inf quaternions and malformed nonfinite poses; U74_yaw_of_quaternion raises ValueError when body-local X is vertical
- U74_planar_step requires positive max_translation and positive max_yaw <= pi, fixes Z to initial Z, retains initial roll/pitch, and cannot repair a tilted goal
- U74_closest_joint_solution requires all columns to be revolute joints, uses 2*pi equivalents inside limits with 1e-10 residue tolerance, preserves input order on exact ties, and applies no collision, contact or step-cap policy
- No planner instance, cache, command clock or task class is shared; U74_capture_contact_transform uses measured live world poses, not commanded tool pose or guessed offset
