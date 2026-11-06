---
name: urdf_pose_kinematics
description: "OwnURDFKinematics parses a contained six-joint URDF, computes forward kinematics, infers the robot base from current joints and observed end-effector pose, and solves bounded IK with five seeds and a positive-elbow preference. cartesian interpolates a pose path and accepts it only if the native numerical collision validator accepts the full joint route."
---

# Urdf pose kinematics

## Function

OwnURDFKinematics parses a contained six-joint URDF, computes forward kinematics, infers the robot base from current joints and observed end-effector pose, and solves bounded IK with five seeds and a positive-elbow preference. cartesian interpolates a pose path and accepts it only if the native numerical collision validator accepts the full joint route.

## Inputs

- urdf: non-symlink file path no larger than 2097152 bytes
- preserve_declared_limits: explicit bool; false clips joint limits to [-3.14159, 3.14159]
- joints: six finite own-joint values; observed_pose/target_pose: finite 7-vector XYZ+WXYZ
- base: finite 4x4 matrix; current: six finite joints; validate_path: callable collision validator

## Outputs

- forward(joints) returns a 4x4 end-effector matrix
- observed_base(joints, observed_pose) returns a 4x4 base matrix
- solve(...) returns joints, position_error_m, rotation_error_radians and public_target_pose, or None
- cartesian(...) returns an (N,6) joint path or None

## Callable interface

Implementation: [implementation.py](implementation.py).

- `pose_matrix(pose)`
- `matrix_pose(matrix)`
- `OwnURDFKinematics: __init__(self, urdf, *, preserve_declared_limits=False); forward(self, joints); observed_base(self, joints, observed_pose); solve(self, base, target_pose, current); cartesian(self, base, current, target, validate_path)`

## Constraints

- Parses joints named joint1 through joint6 and raises ValueError on invalid file, mechanism or numerical inputs.
- solve uses residual position plus 0.12 times rotation error, max_nfev=110, position gate <0.0025 m and rotation gate <0.03 rad; positive elbow is preferred.
- cartesian uses linear position and Slerp orientation with spacing distance/0.008 or angle/0.055, at least 3 waypoints, and returns None above 250 waypoints.
- cartesian rejects joint jumps >0.4 and returns the path only when validate_path(path) is truthy; missing validator raises ValueError.
