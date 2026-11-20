---
name: paired_joint_execution_planned_start
description: "Defines the planned-start variant of PairedJointProgram, validating paired public phases with pose, joint, rule_joint, or hold operations and executing both arms on one phase clock. It plans each side through api.plan_trajectory, records trajectory_operation, time_scale, and requested_public_pose, aborts on plan failure, and holds shorter paths at their final own command."
---

# Paired joint execution planned start

## Function

Defines the planned-start variant of PairedJointProgram, validating paired public phases with pose, joint, rule_joint, or hold operations and executing both arms on one phase clock. It plans each side through api.plan_trajectory, records trajectory_operation, time_scale, and requested_public_pose, aborts on plan failure, and holds shorter paths at their final own command.

## Inputs

- api: restricted public robot API exposing plan_trajectory, home_joint_states, and _observation state; the program reads api._observation['state'][side + '_arm_joint_state']
- request: dict whose side must be 'left' as the pair feedback anchor
- paired_phases: list of phase dicts exactly {left, right, operation, time_scale}; operation is pose, joint, rule_joint, or hold; both sides must match name, minimum_steps, and require_pose
- options: forwarded to the planned-start PhasedProgram adapter; time_scale must be in the planned-start TRAJECTORY_TIME_SCALES

## Outputs

- action dict with left_arm_joint_state, right_arm_joint_state, left_ee_joint_state, and right_ee_joint_state commands
- record events named mature_joint_trajectory with planner, trajectory_operation, time_scale, requested_public_pose, dense_waypoints, and official_waypoints
- None plus status 'uncertain' and failed=True when either arm plan is not Success
- ExecutionError for an invalid phase contract or if either planned path is empty

## Callable interface

Implementation: [implementation.py](implementation.py).

- `PairedJointProgram: __init__(self, api, request, paired_phases, **options); paired_action(self, observation, remaining_actions, record)`

## Constraints

- PairedJointProgram is an adapter-bound planned-start PhasedProgram subclass; behavior depends on inherited self phase/target state and api._observation globals
- joint_target uses api.home_joint_states()[side] for joint and rule_joint operations
- Both arms must share the phase clock; left is used only as feedback anchor and a right-side PhasedProgram is constructed
- Completion describes public command execution, not official success; no environment object, object pose, layout, contact, reward, or true finger measurement is accepted
