---
name: planned_start_joint_planning
description: "Validates a fixed-field public trajectory request, plans or replays a six-joint path through mature cuRobo/X5/timed/sampled/read-only dense operations, and optionally executes it through NativeJointExecutor by emitting public joint and normalized gripper commands. It returns Success/Fail/Budget with dense and official waypoint counts; read-only dense and own-IK modes return raw plan data without an execution allowance."
---

# Planned start joint planning

## Function

Validates a fixed-field public trajectory request, plans or replays a six-joint path through mature cuRobo/X5/timed/sampled/read-only dense operations, and optionally executes it through NativeJointExecutor by emitting public joint and normalized gripper commands. It returns Success/Fail/Budget with dense and official waypoint counts; read-only dense and own-IK modes return raw plan data without an execution allowance.

## Inputs

- payload dict with exactly observation, side, pose, support_plane_m, remaining_actions, time_scale, operation, and joint_target; side must be left or right
- pose: finite 7-vector position plus unit orientation; joint_target: finite 6-vector only for joint, rule_joint, sampled_home, original_dense_joint, and table_dense_joint
- support_plane_m: optional finite numeric current RGB plane hypothesis or null; remaining_actions: integer 1-400; time_scale must be in TRAJECTORY_TIME_SCALES
- operation: declared pose, rule_pose, rule_joint, canonical_pose, joint, own_ik, continuous_pose, curobo_local_descent, x5, sampled, timed-profile, or read-only dense operation name

## Outputs

- dict with exactly status, positions, dense_waypoints, and official_waypoints
- Success returns finite float32 positions shaped (N,6) with 1<=N<=remaining_actions and N==official_waypoints; Fail/Budget return empty positions and zero official_waypoints
- read-only dense results require official_waypoints==0, dense_waypoints 0..8192 matching raw positions, and return raw positions on Success; own_ik returns one 6-joint row with dense_waypoints=official_waypoints=1
- NativeJointExecutor.action returns a public action dict or None; emitted arm joint states contain six values and gripper commands are normalized [0,1]

## Callable interface

Implementation: [implementation.py](implementation.py).

- `vector(value, dimension, label)`
- `validate_request(payload)`
- `replay_waypoints(positions, interval, time_scale)`
- `joint_action(observation, side, joints, opening)`
- `validate_result(value, remaining_actions)`
- `validate_dense_result(value)`
- `canonical_joint_goal(goal, current, lower, upper)`
- `own_arm_joint_limits(urdf)`
- `validate_robot_bases(payload)`
- `NativeJointExecutor: __init__(self, planner, record); action(self, program, observation, remaining_actions)`
- `MaturePlanner: __init__(self, source); reset(self); robot_bases(self, observation); plan(self, payload)`

## Constraints

- validate_request rejects undeclared or missing fields, non-left/right side, nonfinite vector values, non-unit pose orientation, non-whitelisted operation or time_scale, remaining_actions outside 1..400, and joint_target on pose plans
- read-only own_ik and dense operations require remaining_actions==1, time_scale==1.0, and support_plane_m null; sampled_ operations and timed profiles require time_scale==1.0
- replay_waypoints requires finite (N,6) input with 1<=N<=20000, integer interval>=1, positive finite time_scale, and returns float32 replayed positions; paths exceeding remaining_actions return Budget
- executor preservation of commanded targets, peer commands, and paired gripper commands depends on declared callables/attributes and instance-bound self state; gripper values are validated normalized public commands, not physical finger width or contact
