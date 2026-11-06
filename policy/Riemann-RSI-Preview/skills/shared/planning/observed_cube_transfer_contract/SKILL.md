---
name: observed_cube_transfer_contract
description: "Validates and deep-copies public cube-transfer anchor, progress, route, and feedback records for a three-cube stack episode. The measurements validator checks declared evidence against status and issued action count and returns only recognized field names."
---

# Observed cube transfer contract

## Function

Validates and deep-copies public cube-transfer anchor, progress, route, and feedback records for a three-cube stack episode. The measurements validator checks declared evidence against status and issued action count and returns only recognized field names.

## Inputs

- anchor: dict or None with exact schema riemann-current-public-cube-family-v1, three public visual-slot boxes, finite support_estimate_m, and stack_destination_m
- progress: dict with cursor, stage, route_attempt, axis_rejections, view_attempts, final_stage, and placed_spans bounded as implemented
- route: dict or None with side, visual_slot, poses, lift_poses, span_estimate_m, and the declared route fields
- measurements plus status/actions: feedback dict, command status, and nonnegative issued action count

## Outputs

- deep-copied validated anchor, progress, or route; None where the input is None
- empty_progress() returns zero/empty cursor, stage, route_attempt, axis_rejections, view_attempts, final_stage, and placed_spans
- measurements returns the set of recognized declared fields: public_cube_anchor, public_cube_progress, command_phases_completed, and optional extras

## Callable interface

Implementation: [implementation.py](implementation.py).

- `vector(value, size)`
- `anchor(value)`
- `empty_progress()`
- `progress(value)`
- `route(value)`
- `measurements(value, status, actions)`

## Constraints

- Rejects unknown nested fields, non-finite or out-of-bound vectors, invalid schemas, duplicate visual slots, inconsistent support heights, fragmentary spans, and destinations outside the observed family.
- Progress counters are bounded integers; placed spans are 0.003–0.2 m; completed phases require status completed and positive actions.
- public_gripper_response is a normalized 0–1 signal, never physical finger width or contact; object_grasp_confirmed and is_contact_sensor must be false.
- Uses only current public RGB, delivered own robot feedback, and same-episode own command history; no private layout/scorer records or asset tables.
