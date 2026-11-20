---
name: paired_mature_waypoint
description: "Defines PairedWaypointAPI and run(api, step) for paired own-arm waypoints: it builds wrist poses, plans and aligns two independent paths with rounded linspace resampling, merges arm/jaw target dictionaries, optionally retries endpoint checks, and executes through the bound callback. It rejects undeclared geometry, budget, callback, and command fields."
---

# Paired mature waypoint

## Function

Defines PairedWaypointAPI and run(api, step) for paired own-arm waypoints: it builds wrist poses, plans and aligns two independent paths with rounded linspace resampling, merges arm/jaw target dictionaries, optionally retries endpoint checks, and executes through the bound callback. It rejects undeclared geometry, budget, callback, and command fields.

## Inputs

- callbacks: exactly wrist_pose, plan_pose, gripper_controls, execute_joint, observed_pose, all callable and bound by the bootstrap to the restricted public runtime
- PairedWaypointAPI(lower_limits, upper_limits, support_height_m, remaining_actions, time_scale): 6-DOF ordered joint bounds, finite support height, int remaining_actions 1..1000, time_scale 0.25..3
- step: exact fields points_m shape (2,3), height_m 0..0.5 relative to support, support_height_m matching API, quaternion_wxyz unit-ish, phase nonempty <=96 chars, hold_closed bool, check_endpoint bool
- poses/sequence/gripper_sequence/execute inputs: own side left/right, pose 7-vector, normalized gripper target 0..1, dense steps 1..20000

## Outputs

- run returns status issued_without_endpoint_check or own_endpoint_qualified with attempts, endpoint_errors_m when checked, contact_confirmed=False, and official_success_claimed=False
- execute returns receipt {actions_executed} after the bound execute_joint callback
- RuntimeError on endpoint error after one retry, unacknowledged execution, or budget overrun
- ValueError on undeclared callbacks/fields, nonfinite geometry, mismatched support height, invalid bounds, or out-of-range normalized commands

## Callable interface

Implementation: [implementation.py](implementation.py).

- `vector(value, size, label)`
- `merge(*sequences)`
- `align(sequences)`
- `PairedWaypointAPI: __init__(self, callbacks, *, lower_limits, upper_limits, support_height_m, remaining_actions, time_scale=1.0); actions_issued(self); support_height_m(self); poses(self, points, height, quaternion); sequence(self, side, pose, hold_closed, gripper_target=0.0); gripper_sequence(self, side, target, steps); validate_commands(self, controls, side, arm); execute(self, controls, phase, gripper_only=False); observed_pose(self, side)`
- `run(api, step)`

## Constraints

- PairedWaypointAPI is adapter-bound; callbacks must be bound to the restricted public runtime, and a Python callback declaration is not a malicious-code isolation sandbox
- Aligns two nonempty own-arm paths to max length with original rounded linspace indices; merge keeps target dictionaries; closed preload uses a constant last jaw command
- Explicit support_height_m must match API geometry; dense paths are capped at 20,000 samples, endpoint retry at 2, execution budget at 1,000 actions, interval 10 with explicit time_scale, and both routes must qualify before either is sent
- Normalized jaw targets are command state, not measured finger width or contact; 25 mm endpoint tolerance is own-wrist qualification, not object contact or official task success; failed execution poisons the API instance
