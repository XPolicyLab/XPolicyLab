---
name: native_timed_cartesian
description: "NativeTimedCartesian(urdf).plan validates a declared timed Cartesian request and solves own-arm joint samples at the original 25 Hz sample count with cubic blend and Slerp orientation. It returns Success samples, or Fail/Budget without positions."
---

# Native timed cartesian

## Function

NativeTimedCartesian(urdf).plan validates a declared timed Cartesian request and solves own-arm joint samples at the original 25 Hz sample count with cubic blend and Slerp orientation. It returns Success samples, or Fail/Budget without positions.

## Inputs

- payload observation with side_ee_pose and side_arm_joint_state, side, pose endpoint, operation profile, remaining_actions in 1..400, and time_scale equal to 1.0
- substeps positive integer used only for dense_waypoints accounting
- own URDF supplied to NativeTimedCartesian

## Outputs

- Success dict with positions float32 (N, 6), dense_waypoints = N * substeps, official_waypoints = N
- Fail dict with empty positions and zero waypoint counts on RuntimeError; Budget dict with empty positions when count exceeds remaining_actions

## Callable interface

Implementation: [implementation.py](implementation.py).

- `native_count(observed, destination, profile)`
- `solve_samples(kinematics, observed, current, destination, profile, remaining_actions)`
- `NativeTimedCartesian: __init__(self, urdf); plan(self, payload, substeps)`

## Constraints

- operation must be in PROFILES; invalid substeps, operation, or time_scale raises ValueError.
- Each returned sample is one official action; no second decimation or added settling action.
- Own IK uses strict tolerances and at most twelve branch subdivisions; branch-jump RuntimeError may trigger subdivision until the budget limit.
- Predicted endpoints are commands, not contact or task-success evidence; no environment, scorer, reward, or real finger measurement access.
