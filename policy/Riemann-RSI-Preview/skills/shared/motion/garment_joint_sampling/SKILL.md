---
name: garment_joint_sampling
description: "Provides the shared sixteen-command garment sampling clock and preflight helpers for Cartesian wrist paths, joint interpolation, gripper close/release clocks, paired body arcs, and one/two-chunk selection. Functions require caller-supplied api/model context; GarmentOwnModel depends on its imported own_model and module globals."
---

# Garment joint sampling

## Function

Provides the shared sixteen-command garment sampling clock and preflight helpers for Cartesian wrist paths, joint interpolation, gripper close/release clocks, paired body arcs, and one/two-chunk selection. Functions require caller-supplied api/model context; GarmentOwnModel depends on its imported own_model and module globals.

## Inputs

- api exposing own_joint_states(), targets(), and issued_action_count() as used by the adapter
- goals, starts, or displacements dictionaries keyed by left and/or right
- release and cuff booleans for the gripper clock
- position/rotation tolerances or chunk_index/chunk_count where applicable

## Outputs

- list of sixteen deep-copied command dictionaries containing *_arm_joint_state and *_ee_joint_state
- chunk count of 1 or 2 from cuff_carry_chunks or body_fold_chunks
- minimum inter-arm rigid gap in meters from _interarm_rigid_gap
- solved six-joint arrays from GarmentOwnModel.solve

## Callable interface

Implementation: [implementation.py](implementation.py).

- `GarmentOwnModel: __init__(self); eef(self, joints); pad(self, joints, root); roots(self, api); solve(self, target, seed, root, position_tolerance=0.004, rotation_tolerance=0.05)`
- `downward_wrist(contact, closing_axis, tilt_degrees)`
- `current_snapshot(api)`
- `sampled_wrist_path(api, model, goals, opening_goals=None, contact=False, starts=None)`
- `sampled_joint_path(api, goals)`
- `sampled_gripper_clock(api, model, arms, release=False, cuff=False)`
- `sampled_body_arc(api, model, starts, displacements, begin, end, chunk_index, chunk_count)`
- `cuff_carry_chunks(api, model, goals)`
- `body_fold_chunks(api, model, starts, displacements)`

## Constraints

- Requires six finite own joints and declared URDF bounds; enforces own-pad mounting clearance of .07 m above roots, paired wrist separation >= .105 m, and per-sample branch jump <= .30 rad
- Compact hem preflight requires the original .008 m rigid-hull gate as well as IK, branch, and paired-wrist clearance; failed preflight keeps the two-chunk path
- Uses byte-hashed fixed link2..6 arm/housing resources and no finger-joint or scene measurements
- Jaw values are normalized command/observation values, not physical finger width or contact; no solver query budget is imposed, and chunks are not partially issued or resampled
