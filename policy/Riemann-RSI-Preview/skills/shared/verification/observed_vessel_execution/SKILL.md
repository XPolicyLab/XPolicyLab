---
name: observed_vessel_execution
description: "Validates public vessel state, scene, rim, route, feedback measurements, and recorded same-episode evidence; state() and recorded() return deep-copied state while measurements() returns selected EXTRA keys. It rejects unknown keys, nonfinite or out-of-bounds vectors, invalid rotations, source/contact/success flags, receipt mismatches, and oversized serialized state with ValueError."
---

# Observed vessel execution

## Function

Validates public vessel state, scene, rim, route, feedback measurements, and recorded same-episode evidence; state() and recorded() return deep-copied state while measurements() returns selected EXTRA keys. It rejects unknown keys, nonfinite or out-of-bounds vectors, invalid rotations, source/contact/success flags, receipt mismatches, and oversized serialized state with ValueError.

## Inputs

- value: mapping with public_vessel_state and command_phases_completed, optionally public_timed_execution and public_vessel_review
- public_vessel_state: exact empty() key set with stage in STAGES, scene/route/rim, bounded counters, optional tilt_poses, ball_candidate, and view_support
- memory: event list where matching ACTIONS events may contain execution_feedback.image_measurements.public_vessel_state
- status: execution status string; actions: number of issued native actions

## Outputs

- Deep-copied validated vessel state from state(), or empty() when recorded() finds no matching event
- Set of selected EXTRA evidence keys from measurements()
- ValueError on schema, bounds, source, flag, receipt, or size violations

## Callable interface

Implementation: [implementation.py](implementation.py).

- `number(value, lower, upper)`
- `vector(value, size)`
- `camera_names(value)`
- `measured_body(value)`
- `measured_opening(value)`
- `scene(value)`
- `compact_scene(value)`
- `rim(value)`
- `route(value)`
- `empty()`
- `state(value)`
- `recorded(memory)`
- `measurements(value, status, actions)`

## Constraints

- State must have exactly empty() keys and a stage in STAGES; booleans are type-checked and cannot represent contact or success
- Vectors are finite with max abs ≤3; 7-vectors require quaternion norm 0.99–1.01; rim rotation must be orthonormal with determinant 1
- Metric records require 2–3 distinct allowed public cameras, source-specific strings, false physical_asset_geometry_used/object_ground_truth_used, and bounded meter/uncertainty/pixel_rms values
- command_phases_completed true requires status 'completed' and actions>0; timed receipt acknowledgments must equal actions; state JSON ≤6400 bytes; review flags is_contact_sensor/object_grasp_confirmed/official_success_claimed must be false
