---
name: observed_rigid_transfer_observed_start
description: "Defines closed-schema constructors, validators, history recovery, and execution-receipt checks for public rigid-transfer progress. It validates goal, scene, support-memory, view-hint, job/recipe, planned/visited references, and stage measurements, returning deep copies for state-bearing validators and raising ValueError on disallowed or unbounded data."
---

# Observed rigid transfer observed start

## Function

Defines closed-schema constructors, validators, history recovery, and execution-receipt checks for public rigid-transfer progress. It validates goal, scene, support-memory, view-hint, job/recipe, planned/visited references, and stage measurements, returning deep copies for state-bearing validators and raising ValueError on disallowed or unbounded data.

## Inputs

- instruction string for initial() and recorded(); initial() fingerprints it with SHA-256 and recorded() resets when the stored fingerprint differs
- public_rigid_transfer_state mapping for validate()/measurements(), matching the riemann-public-rigid-transfer-v3 closed key set and PHASES
- nested goal, scene, anchor, view_hint, support_memory, job, recipe, planned_sources, planned_targets, visited_targets, and last_source_epoch records
- measurements(value, status, action_count) value containing public_rigid_transfer_state, rigid_visual_review, rigid_command_phases_completed, and optional own execution receipts

## Outputs

- initial(instruction) returns a new state dict with phase 'clear', zeroed counters, empty lists, goal/scene/support_memory/view_hint None, and an initial job
- validate(value) and recorded(memory, instruction) return deep-copied validated state; recorded scans the last 32 memory events for public_rigid_transfer_state and falls back to initial(instruction)
- measurements returns a set of extra receipt keys; bowl_scene_diagnostics returns None
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
- `bowl_scene_diagnostics(records)`
- `measurements(value, status, action_count)`

## Constraints

- Requires schema 'riemann-public-rigid-transfer-v3', a 64-character lowercase hex instruction fingerprint, phase in PHASES, and bounded counters (view_leg<=2, item<=32, attempts<=3, home_leg<=2, observed_source_count<=32, view_attempts<=4)
- Pose vectors are 7-element finite bounded arrays with quaternion norm in [0.99,1.01]; colors are 3-vectors bounded to 255; geometry estimates are in meters with declared resolution semantics
- Rejects private layouts, asset models, scorer outcomes, contact measurements, official-success claims, non-finite or unbounded values, and out-of-range camera provenance or coverage
- Visual following and placement review is uncertain public evidence with is_contact_measurement False; completed phases require native action acknowledgments and status completed or uncertain; bowl_scene_diagnostics accepts 1-2 scalar admission records with fixed stage/reason sets, bounded qualified_bowl_count, ordered public_action_epoch, and no geometry, command state, or success evidence
