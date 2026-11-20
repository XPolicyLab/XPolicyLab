---
name: observed_pickup_state_observed_start
description: "Validates and deep-copies closed same-episode pickup progress, current public RGB visual-hull anchors, own robot routes, and public command/uncertainty feedback. The recorded helper scans the last 32 events for allowed pickup skills and returns the latest valid public_pickup_state or initial progress."
---

# Observed pickup state observed start

## Function

Validates and deep-copies closed same-episode pickup progress, current public RGB visual-hull anchors, own robot routes, and public command/uncertainty feedback. The recorded helper scans the last 32 events for allowed pickup skills and returns the latest valid public_pickup_state or initial progress.

## Inputs

- value: closed progress dictionary with stage in STAGES, attempts, view_attempts, corrections, verify_count, anchor, route, contact_actual, and failure_reason
- anchor: current_public_rgb_visual_hull with prompt, center_m, spans_m, height_estimate_m, uncertainty_m, cameras, coverage, 96x2 head_outline, head_centroid, appearance_rgb, support_memory, and resolution_semantics
- route: side, 7-D pregrasp/contact/lift poses, 6-D pregrasp/contact/lift joints, width_estimate_m, cost, fraction, and tilt_degrees
- measurements containing public_pickup_state, pickup_visual_review, native_command_clock, command_phases_completed, plus status and actions

## Outputs

- deep-copied validated progress, anchor, and route dictionaries
- initial() default pickup progress state
- empty_visual() default visual review with unavailable status
- set of required measurement field names and ValueError on invalid schema, range, source, or consistency

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

- STAGES and REASONS are closed; counters are attempts at most 3, view_attempts at most 6, corrections at most 1, and verify_count at most 16; stages outside view, pregrasp, and reset require both anchor and route
- Anchor source must be current_public_rgb_visual_hull and resolution_semantics grid_and_support_resolution_not_hidden_object_error; prompt must be 1-128 characters without control characters; spans_m horizontal maximum at least 0.004 m; height_estimate_m 0.005-0.35 m; uncertainty_m 0-0.018 m; cameras 2-3 unique with cam_head first; coverage 0.55-1; head_outline is 96x2 and bounded inside a 640x480 image; support_memory is validated by public_support_geometry_context
- Route side is left or right; poses are 7-D with quaternion norm 0.99-1.01; joints are 6-D within plus or minus 10; width_estimate_m 0.004-0.082 m; cost 0-100; fraction 0-1; tilt_degrees is one of 0, 15, 25, 35
- Visual review status is unavailable, ambiguous, follows_hand, or still_at_source; is_contact_or_true_finger_sensor and official_success_claimed must be False; native clock requires segment_width 16, endpoint_padding_preserved=True, trajectory_resampled=False, opening_semantics public_normalized_command_not_real_fingers; completed command phases require status completed or uncertain and actions at least 16
