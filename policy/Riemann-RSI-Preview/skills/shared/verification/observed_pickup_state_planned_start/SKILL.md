---
name: observed_pickup_state_planned_start
description: "Validates a closed same-episode public pickup-progress record, its current visual anchor, route, contact pose, bounded retries, and tool measurement receipt; recorded() recovers the latest valid public_pickup_state from recent memory or returns initial()."
---

# Observed pickup state planned start

## Function

Validates a closed same-episode public pickup-progress record, its current visual anchor, route, contact pose, bounded retries, and tool measurement receipt; recorded() recovers the latest valid public_pickup_state from recent memory or returns initial().

## Inputs

- public_pickup_state dict with exactly stage, attempts, view_attempts, corrections, verify_count, anchor, route, contact_actual, failure_reason
- anchor dict with source current_public_rgb_visual_hull, prompt, center_m, spans_m, height_estimate_m, uncertainty_m, cameras, coverage, head_outline, head_centroid, appearance_rgb, support_memory, resolution_semantics
- route dict with side, pregrasp, contact, lift, pregrasp_joints, contact_joints, lift_joints, width_estimate_m, cost, fraction, tilt_degrees
- measured dict plus status and actions for validate_measurements; memory list of event dicts for recorded

## Outputs

- Deep-copied validated public_pickup_state dict
- initial() state with stage view, zero counters, and None anchor, route, contact_actual, failure_reason
- recorded(memory) returns the latest validated public_pickup_state or initial()
- validate_measurements(measured, status, actions) returns the required measurement field-name set

## Callable interface

Implementation: [implementation.py](implementation.py).

- `vector(value, count, bound, label)`
- `pose(value)`
- `validate_anchor(value)`
- `validate_route(value)`
- `initial()`
- `validate(value)`
- `recorded(memory)`
- `empty_visual()`
- `validate_measurements(measured, status, actions)`

## Constraints

- Closed field sets and ValueError on unknown/private fields, nonfinite or out-of-bound values, non-unit quaternions, invalid stages/reasons, or missing anchor/route after view, pregrasp, or reset stages.
- Counters are ints bounded attempts 0-3, view_attempts 0-6, corrections 0-1, verify_count 0-16; anchor coverage needs 2-3 unique cameras beginning cam_head, coverage >=0.55, height_estimate_m 0.005-0.35, uncertainty_m 0-0.018, and spans_m positive with max(x,y)>=0.004 m.
- Route poses use unit quaternions, joints are bounded 6-vectors, width_estimate_m 0.004-0.082, cost 0-100, fraction 0-1, and tilt_degrees in (0, 15, 25, 35).
- pickup_visual_review must keep is_contact_or_true_finger_sensor and official_success_claimed false; native_command_clock must equal segment_width 16, endpoint_padding_preserved true, trajectory_resampled false, opening_semantics public_normalized_command_not_real_fingers; completed requires status completed/uncertain and actions>=16.
