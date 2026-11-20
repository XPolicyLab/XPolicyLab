---
name: observed_rigid_transfer_planned_start
description: "Defines closed-schema constructors, validators, history recovery, and execution-receipt checks for planned-start public rigid-transfer progress. It mirrors the observed-start rigid-transfer validation but uses the planned-start support-memory validator and planned-start public action recipe initial_job, with bowl_scene_diagnostics imported from the observed-start implementation."
---

# Observed rigid transfer planned start

## Function

Defines closed-schema constructors, validators, history recovery, and execution-receipt checks for planned-start public rigid-transfer progress. It mirrors the observed-start rigid-transfer validation but uses the planned-start support-memory validator and planned-start public action recipe initial_job, with bowl_scene_diagnostics imported from the observed-start implementation.

## Inputs

- instruction string for initial() and recorded(); initial() fingerprints it with SHA-256 and recorded() resets when the stored fingerprint differs
- public_rigid_transfer_state mapping for validate()/measurements(), matching the riemann-public-rigid-transfer-v3 closed key set and PHASES
- nested goal, scene, anchor, view_hint, support_memory, job, recipe, planned_sources, planned_targets, visited_targets, and last_source_epoch records
- measurements(value, status, action_count) value containing public_rigid_transfer_state, rigid_visual_review, rigid_command_phases_completed, and optional own execution receipts

## Outputs

- initial(instruction) returns a new state dict with phase 'clear', zeroed counters, empty lists, goal/scene/support_memory/view_hint None, and a planned-start initial job
- validate(value) and recorded(memory, instruction) return deep-copied validated state; recorded scans the last 32 memory events for public_rigid_transfer_state and falls back to initial(instruction)
- measurements returns a set of extra receipt keys; the imported bowl_scene_diagnostics returns None
- ValueError is raised for schema, bound, provenance, unit-quaternion, finite/bounded numeric, or receipt violations

## Callable interface

Implementation: [implementation.py](implementation.py).

- `vector(value, size, maximum=4.0)`
- `pose(value)`
- `goal(value)`
- `anchor(value)`
- `scene(value)`
- `initial(instruction)`
- `initial_job()`
- `validate(value)`
- `recorded(memory, instruction)`
- `measurements(value, status, action_count)`

## Constraints

- Requires schema 'riemann-public-rigid-transfer-v3', a 64-character lowercase hex instruction fingerprint, phase in PHASES, and bounded counters (view_leg<=2, item<=32, attempts<=3, home_leg<=2, observed_source_count<=32, view_attempts<=4)
- Pose vectors are 7-element finite bounded arrays with quaternion norm in [0.99,1.01]; colors are 3-vectors bounded to 255; geometry estimates are in meters with declared resolution semantics
- Uses skills.shared.perception.public_support_geometry_context_planned_start.implementation.validate for support_memory and skills.shared.verification.public_action_recipe_planned_start.implementation.initial_job for initial job creation
- Rejects private layouts, asset models, scorer outcomes, contact measurements, official-success claims, non-finite or unbounded values, and out-of-range camera provenance or coverage; visual following and placement review is uncertain public evidence with is_contact_measurement False
