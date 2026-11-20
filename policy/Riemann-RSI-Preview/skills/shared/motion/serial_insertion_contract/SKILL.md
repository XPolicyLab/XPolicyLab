---
name: serial_insertion_contract
description: "Validates bounded serial insertion source, anchor, route, and progress records and advances the grasp, lift, upright, and seat stage handoffs. It rejects malformed geometry, skipped stages, and forged completion flags."
---

# Serial insertion contract

## Function

Validates bounded serial insertion source, anchor, route, and progress records and advances the grasp, lift, upright, and seat stage handoffs. It rejects malformed geometry, skipped stages, and forged completion flags.

## Inputs

- source: dict exactly {grasp, axis, cap}, each a finite 3-vector with abs values <= 3; axis must be horizontal within 1e-6 and unit norm .99..1.01
- anchor: dict exactly sources, targets, source_surface_m, target_surface_m, support_plane_m, target_radius_m; 2..8 sources, equal-length 3-vector targets, finite scalar geometry abs <= 3, radius .005..0.12 m
- route: dict exactly fingertip_target 3-vector, quaternion unit 4-vector, joint_target 6-vector with abs values <= 64
- progress: dict with order as a 2..8 slot permutation, bounded integer cursor/stage/target_index/final_stage, and boolean inspection_done/recovery_used/settled; stage/source/route must agree

## Outputs

- validated deep-copied source, anchor, route, and progress dicts
- empty_progress(order) returns an initial validated progress record
- advance(value, stage, source, route, target_index) returns the next validated progress record or raises ValueError

## Callable interface

Implementation: [implementation.py](implementation.py).

- `vector(value, size, limit=3.0)`
- `validate_source(value)`
- `validate_anchor(value)`
- `validate_route(value)`
- `empty_progress(order)`
- `validate_progress(value)`
- `advance(value, stage, source=None, route=None, target_index=None)`

## Constraints

- Only one explicitly recorded recovery is permitted; extra stages require an actual recorded recovery and final_stage requires cursor beyond order with stage 0
- upright route is an own-joint endpoint, not a measured object pose; progress and settlement mean commands executed, never contact, grasp, or official task success
- rejects extra or missing fields, malformed geometry, non-unit quaternions/axes, skipped stages, and forged completion flags
- contract holds no environment, scorer, layout, asset dimensions, or task-specific controller
