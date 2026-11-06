---
name: observed_start_joint_planning
description: "Validates a public observed-start trajectory request and delegates planning/replay to the original cuRobo, X5, timed-Cartesian, sampled, and dense backends, returning bounded six-joint waypoints or explicit Fail/Budget. NativeJointExecutor converts accepted paths or sampled phases into public joint action dictionaries and records trace data."
---

# Observed start joint planning

## Function

Validates a public observed-start trajectory request and delegates planning/replay to the original cuRobo, X5, timed-Cartesian, sampled, and dense backends, returning bounded six-joint waypoints or explicit Fail/Budget. NativeJointExecutor converts accepted paths or sampled phases into public joint action dictionaries and records trace data.

## Inputs

- Request dict with exactly observation, side, pose, support_plane_m, remaining_actions, time_scale, operation, joint_target
- side is 'left' or 'right'; pose is a finite 7-vector whose quaternion part has norm 0.99..1.01; support_plane_m is None or a finite int/float RGB plane hypothesis
- operation is from the declared pose/joint/dense/continuous/sampled/x5/timed whitelist; joint_target is a finite 6-vector only for joint, rule_joint, sampled_home, or original_dense_joint
- Executor action takes program, public observation, and remaining_actions; program may provide bound callbacks and api state used by the adapter

## Outputs

- Plan result dict with exactly status in Success/Fail/Budget, positions, dense_waypoints, official_waypoints; own_ik Success returns one joint row with counts 1/1 while executor rejects executing own_ik; status is a planning result only, not official task success
- Success positions are finite six-joint rows with length <= remaining_actions and equal official_waypoints; non-Success results carry no positions or official waypoints
- Read-only dense results require official_waypoints 0 and dense_waypoints 0..8192 with raw positions length matching dense_waypoints
- Executor returns action dict with both arms' public joint states, selected arm's normalized gripper command 0..1, optional preserved peer/gripper commands, or None when finished

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

- validate_request raises ExecutionError for wrong field sets, invalid side/pose/plane/operation, non-whitelisted time_scale, or illegal joint_target; remaining_actions must be int 1..400
- sampled_* and timed-profile operations require time_scale 1.0; own_ik and dense operations require remaining_actions 1, time_scale 1.0, and support_plane_m None
- Planner reuses original cuRobo/X5/timed/sampled implementations and official-bridge resampling, with canonical_joint_goal choosing the nearest in-limit 2*pi equivalent; it does not add another IK solver or replace paths with repeated EE IK
- Executor depends on adapter-provided program/api globals/self, mutates its own key/positions/index/commanded_targets, validates declared peer/gripper callbacks, and treats gripper values as normalized public commands, not physical finger width or contact
