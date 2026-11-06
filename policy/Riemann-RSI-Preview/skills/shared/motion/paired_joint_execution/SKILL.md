---
name: paired_joint_execution
description: "Defines a PairedJointProgram adapter that validates a recorded public pair recipe and executes paired left/right public pose, joint, or hold phases on one bounded phase clock. It plans both arms with api.plan_trajectory, records each plan, aborts the pair on plan failure, and holds shorter paths at their final own command."
---

# Paired joint execution

## Function

Defines a PairedJointProgram adapter that validates a recorded public pair recipe and executes paired left/right public pose, joint, or hold phases on one bounded phase clock. It plans both arms with api.plan_trajectory, records each plan, aborts the pair on plan failure, and holds shorter paths at their final own command.

## Inputs

- api: restricted public robot API exposing plan_trajectory, home_joint_states, and _observation state; the program reads api._observation['state'][side + '_arm_joint_state']
- request: dict whose side must be 'left' as the pair feedback anchor
- paired_phases: list of phase dicts exactly {left, right, operation, time_scale}; operation is pose, joint, or hold; both sides must match name, minimum_steps, and require_pose
- validate_public_pair_recipe(value): dict exactly {start, end, approach, sweep_distance_m, visible_surface_z}; each phase has left/right 7-vector poses with finite unit-ish quaternion and sweep_distance_m in 0.01..0.35 m

## Outputs

- action dict with left_arm_joint_state, right_arm_joint_state, left_ee_joint_state, and right_ee_joint_state commands
- record events named mature_joint_trajectory with status, dense_waypoints, official_waypoints, planner original_curobo_plan_joint or original_curobo_plan_path, and public robot_state_source
- None plus status 'uncertain' and failed=True when either arm plan is not Success
- ExecutionError for an invalid phase contract or if either planned path is empty

## Callable interface

Implementation: [implementation.py](implementation.py).

- `validate_public_pair_recipe(value)`
- `PairedJointProgram: __init__(self, api, request, paired_phases, **options); paired_action(self, observation, remaining_actions, record)`

## Constraints

- PairedJointProgram is an adapter-bound PhasedProgram subclass; behavior depends on inherited self phase/target state and api._observation globals
- Both arms must share the phase clock and bounded planner contract; left is used only as feedback anchor and a right-side PhasedProgram is also constructed
- Shorter planned paths hold their final own command while the other finishes; no arm is silently omitted
- Completion describes public command execution, not official success; no environment object, object pose, layout, contact, reward, or true finger measurement is accepted
