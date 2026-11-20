---
name: observed_liquid_execution_planned_start
description: "Validates and deep-copies same-episode liquid execution progress, current free multiview vessel profiles, own route and attachment hypotheses, and fixed-step tool receipts. The recorded helper scans reversed memory for allowed liquid skills and returns the latest valid public_liquid_state or an empty state."
---

# Observed liquid execution planned start

## Function

Validates and deep-copies same-episode liquid execution progress, current free multiview vessel profiles, own route and attachment hypotheses, and fixed-step tool receipts. The recorded helper scans reversed memory for allowed liquid skills and returns the latest valid public_liquid_state or an empty state.

## Inputs

- state value: closed dictionary with stage in STAGES, scene, side, support_side, route, attachment, tilt_poses, bounded counters, reason, official_success_claimed=False, and optional view_support
- scene value: bottle profile, measured cup opening, optional cup profile, support_estimate_m, support_uncertainty_m, and source current_independently_qualified_liquid_pair
- measurements value containing public_liquid_state, native_execution_receipt, public_liquid_review, plus status and actions
- memory sequence of events with skill, execution_feedback.image_measurements.public_liquid_state

## Outputs

- deep-copied validated state, scene, profile, route, attachment, and measurement inputs
- empty() initial liquid progress state
- set of extra required measurement keys: public_liquid_state, native_execution_receipt, public_liquid_review
- ValueError on schema, range, source, consistency, or bounded-counter violations

## Callable interface

Implementation: [implementation.py](implementation.py).

- `profile(value)`
- `scene(value)`
- `empty()`
- `state(value)`
- `recorded(memory)`
- `measurements(value, status, actions)`

## Constraints

- Body profiles require source current_independent_multiview_free_axial_profile, physical_asset_geometry_used=False, object_ground_truth_used=False, six legacy radii or eight bounded observed radii, center_xy_m length 2, height_estimate_m 0.006-0.35 m, radii 0.001-0.18 m, height_fractions from 0 to 1 with minimum 0.005 spacing, support_estimate_m -3 to 3 m, support_uncertainty_m 0-0.006 m, pixel_rms 0-2.5, and 2-3 unique public cameras
- Scene source must be current_independently_qualified_liquid_pair; bottle support and cup opening height must agree within 0.006 m, and cup opening height above support must be 0.006-0.4 m; cup_profile may be None
- Native receipt requires schema riemann-original-fixed-step-execution-v1, native_chunk_maximum 16, gripper_commands constant_original_normalized_opening, trajectory_sampling_reapplied=False, true_fingers_or_contact_read=False, official_success_claimed=False, bounded counters, actions_executed equal to actions, endpoint_settle_commands equal to 16 times endpoint_settle_intervals, and completed status consuming every prepared native sample once
- Public review requires source current_public_rgb_and_own_command_response; gripper_is_contact_sensor, object_grasp_confirmed, and official_success_claimed must all be False; attachment is a visual movement hypothesis, not measured contact; route poses are 7-D with unit quaternion norm 0.99-1.01 and tilt_poses count is 0, 49, or 97
