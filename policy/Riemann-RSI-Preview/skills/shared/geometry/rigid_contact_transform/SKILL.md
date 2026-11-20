---
name: rigid_contact_transform
description: "Pure SE(3) quaternion/pose helpers for rigid contact: compose and invert seven-value scalar-first wxyz poses, capture object-in-tool from measured live world poses, and solve a desired tool pose from a desired object pose and captured relative transform."
---

# Rigid contact transform

## Function

Pure SE(3) quaternion/pose helpers for rigid contact: compose and invert seven-value scalar-first wxyz poses, capture object-in-tool from measured live world poses, and solve a desired tool pose from a desired object pose and captured relative transform.

## Inputs

- finite seven-value poses with scalar-first (w,x,y,z) quaternions
- object_pose and tool_pose measured live world poses for capture
- desired_object_pose and object_in_tool for tool_pose_for_object
- rotation quaternions/vectors for quat_mul, quat_inverse, quat_to_mat, and rotate_vec

## Outputs

- compose_pose returns parent*child seven-value pose; inverse_pose returns the inverse pose
- capture_contact_transform returns inv(T_world_tool) @ T_world_object
- tool_pose_for_object returns T_world_object_goal @ inv(object_in_tool)
- quaternion/matrix helpers return normalized quaternions or a 3x3 rotation matrix

## Callable interface

Implementation: [implementation.py](implementation.py).

- `quat_mul_wxyz(left, right)`
- `quat_inverse_wxyz(quaternion)`
- `quat_to_mat_wxyz(quaternion)`
- `rotate_vec_by_quat(vector, quaternion)`
- `compose_pose(parent, child)`
- `inverse_pose(pose)`
- `capture_contact_transform(object_pose, tool_pose)`
- `tool_pose_for_object(desired_object_pose, object_in_tool)`

## Constraints

- Poses are validated and quaternions normalized by finite_rotation_pose helpers
- Capture must be after contact settles using measured poses, not commanded tool pose or guessed offset
- No environment handle, simulator state, layout, object dimensions, or scorer is read
- Numeric output is not object-contact or task-success evidence; caller must attach current public observation provenance
