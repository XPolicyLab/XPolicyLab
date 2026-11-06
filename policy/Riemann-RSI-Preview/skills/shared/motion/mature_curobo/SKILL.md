---
name: mature_curobo
description: "MaturePlanner validates a declared trajectory request, infers own robot bases from public joint/EE observations, and returns a resampled six-joint plan via original cuRobo, canonical endpoint replan, timed Cartesian, own-X5 fallback, sampled motion, local descent, or read-only own IK. NativeJointExecutor is adapter-bound and executes phase programs by issuing own arm and normalized gripper commands while validating planner results and remaining-action budgets."
---

# Mature curobo

## Function

MaturePlanner validates a declared trajectory request, infers own robot bases from public joint/EE observations, and returns a resampled six-joint plan via original cuRobo, canonical endpoint replan, timed Cartesian, own-X5 fallback, sampled motion, local descent, or read-only own IK. NativeJointExecutor is adapter-bound and executes phase programs by issuing own arm and normalized gripper commands while validating planner results and remaining-action budgets.

## Inputs

- plan(payload): exact dict with observation, side ('left'/'right'), pose 7-vector, support_plane_m (optional finite scalar or null), remaining_actions int 1-400, time_scale from TRAJECTORY_TIME_SCALES (0.25, 0.8, 0.9, 1.0, 1.12, 1.25, 1.4, 1.8, 2.5, 3.0), operation from the declared enum, and joint_target 6-vector only for joint or sampled_home
- public observation containing own six-joint arm states, one-element normalized gripper states, and seven-element end-effector poses
- MaturePlanner(source) source directory with configs/mature-trajectory.json, robot_yaml, and robot_urdf; NativeJointExecutor(planner, record)
- NativeJointExecutor.action(program, observation, remaining_actions) with a phase program exposing phases, side, targets, preplanned, phase_index, phase_actions, and optional command/trajectory hooks

## Outputs

- plan returns {'status': 'Success'|'Fail'|'Budget', 'positions': Nx6 six-joint positions, 'dense_waypoints': int, 'official_waypoints': int}; non-Success has empty positions and zero official waypoints
- NativeJointExecutor.action returns an own-action dict of left/right arm joint states and one-element normalized gripper commands, or None when finished/failed
- execution records include trajectory status/counts and endpoint command-response events; no official task-success flag is produced

## Callable interface

Implementation: [implementation.py](implementation.py).

- `vector(value, dimension, label)`
- `validate_request(payload)`
- `replay_waypoints(positions, interval, time_scale)`
- `joint_action(observation, side, joints, opening)`
- `validate_result(value, remaining_actions)`
- `canonical_joint_goal(goal, current, lower, upper)`
- `own_arm_joint_limits(urdf)`
- `validate_robot_bases(payload)`
- `NativeJointExecutor: __init__(self, planner, record); action(self, program, observation, remaining_actions)`
- `MaturePlanner: __init__(self, source); reset(self); robot_bases(self, observation); plan(self, payload)`

## Constraints

- validate_request requires exact declared keys, side left/right, finite 7-vector pose with orientation norm 0.99-1.01, whitelisted operation, finite optional plane, remaining_actions int 1-400, and declared time_scale; sampled_* and timed profiles require time_scale 1, and own_ik requires remaining_actions 1, time_scale 1, and no plane
- replay_waypoints accepts a finite Nx6 path with 1..20000 rows, integer interval >=1, and positive time_scale, resamples at stride interval*time_scale, and plan returns Budget when official positions exceed remaining_actions
- validate_result requires exact result fields and status Success/Fail/Budget; non-Success must have no positions or official waypoints; Success positions must be finite Nx6, 1..remaining_actions, and length equal official_waypoints
- canonical_joint_goal chooses per-joint in-limit 2*pi equivalents nearest the current joint or returns None; inferred robot bases require unit orientations and left/right z difference <=0.005 m; gripper opening is a normalized public command in [0,1], not a physical finger measurement or contact signal
