---
name: contact_pose_error
description: "Computes position, XY, Z, full SO(3), and diagnostic yaw errors between actual_pose and target_pose. quaternion_angle returns the full geodesic error in [0, pi] invariant under q -> -q."
---

# Contact pose error

## Function

Computes position, XY, Z, full SO(3), and diagnostic yaw errors between actual_pose and target_pose. quaternion_angle returns the full geodesic error in [0, pi] invariant under q -> -q.

## Inputs

- actual_pose: finite 7-vector [x, y, z, qw, qx, qy, qz]
- target_pose: finite 7-vector [x, y, z, qw, qx, qy, qz]
- quaternion and target_quaternion for quaternion_angle: WXYZ quaternions

## Outputs

- dict with position_m, xy_m, z_m, orientation_rad, orientation_deg, yaw_rad, yaw_deg
- yaw_rad/yaw_deg are None when planar heading is undefined; orientation_rad/orientation_deg still measure full error

## Callable interface

Implementation: [implementation.py](implementation.py).

- `quaternion_angle(quaternion, target_quaternion)`
- `pose_errors(actual_pose, target_pose)`

## Constraints

- Validates poses; yaw is derived from body-local X projected on world XY and raises if body-local X is vertical, which pose_errors converts to None.
- Yaw is diagnostic only and never success; full orientation error remains valid for vertical local X.
- No environment handle, simulator state, layout, object dimensions, or scorer is read.
- Caller must attach current public observation provenance; numeric output is not contact or task-success evidence.
