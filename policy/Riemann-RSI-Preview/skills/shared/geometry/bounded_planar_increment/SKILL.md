---
name: bounded_planar_increment
description: "Computes one bounded planar XY/yaw step from current_pose toward goal_pose on the initial pose's planar manifold. It limits XY displacement by Euclidean norm and yaw by the shortest wrapped world-Z angle, returning a 7-vector pose with the initial Z and roll/pitch."
---

# Bounded planar increment

## Function

Computes one bounded planar XY/yaw step from current_pose toward goal_pose on the initial pose's planar manifold. It limits XY displacement by Euclidean norm and yaw by the shortest wrapped world-Z angle, returning a 7-vector pose with the initial Z and roll/pitch.

## Inputs

- current_pose: finite 7-vector [x, y, z, qw, qx, qy, qz] with normalized WXYZ quaternion
- goal_pose: finite 7-vector [x, y, z, qw, qx, qy, qz] with normalized WXYZ quaternion
- initial_pose: finite 7-vector [x, y, z, qw, qx, qy, qz] with normalized WXYZ quaternion
- max_translation: positive XY Euclidean bound in meters, default 0.02; max_yaw: positive yaw bound in radians, default radians(8), must not exceed pi

## Outputs

- NumPy 7-vector pose [next_x, next_y, initial_z, qw, qx, qy, qz]

## Callable interface

Implementation: [implementation.py](implementation.py).

- `quat_axis_angle_wxyz(axis, angle)`
- `yaw_quat(angle)`
- `wrap_yaw(angle)`
- `yaw_of_quaternion(quaternion)`
- `planar_step(current_pose, goal_pose, initial_pose, *, max_translation=0.02, max_yaw=math.radians(8))`

## Constraints

- Validates all poses and requires positive max_translation and max_yaw; raises ValueError if max_yaw exceeds pi or XY displacement norm overflows.
- Raises ValueError when yaw is undefined because body-local X is vertical.
- Always uses initial Z and retains initial full roll/pitch via q_next = yaw_quat(wrapped(current_yaw + step - initial_yaw)) * initial_quaternion; it cannot repair a tilted goal.
- Numeric output is not object-contact or task-success evidence; the caller must attach current public observation provenance.
