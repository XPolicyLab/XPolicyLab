---
name: observed_number_contract
description: "Validates complete independently recognized numeral/target families, public surface and mounting hypotheses, bounded own-command progress, and own route/measurement records for the observed-number contract. Anchor, progress, and route validators return deep copies; measurement validation returns no value and raises ValueError for invalid or private/scorer-like evidence."
---

# Observed number contract

## Function

Validates complete independently recognized numeral/target families, public surface and mounting hypotheses, bounded own-command progress, and own route/measurement records for the observed-number contract. Anchor, progress, and route validators return deep copies; measurement validation returns no value and raises ValueError for invalid or private/scorer-like evidence.

## Inputs

- anchor: dict with exact schema 'riemann-current-number-scene-v1', digits/targets, support_reference_m, mounting_reference, target_geometry_source, observed_target_count, and current_layout_source=False
- progress: dict with integer cursor, stage, count, final_stage, and retry
- route: dict with side, digit, 7-vector approach/grasp/lift/carry/place poses, 3-vector source_tip/target_tip, closing_axis, tilt, branch_score, and joint_limit_margin_rad
- measurements, status, actions: public feedback mapping, status string, and issued-action count used by validate_measurements

## Outputs

- validate_anchor, validate_progress, advance, and validate_route return deep copies of validated anchor/progress/route objects.
- empty_progress returns a fresh progress mapping with cursor 0, stage 0, count set to the argument, final_stage 0, and retry 0.
- validate_measurements returns no value; invalid input raises ValueError.

## Callable interface

Implementation: [implementation.py](implementation.py).

- `validate_anchor(anchor)`
- `empty_progress(count=0)`
- `validate_progress(progress, count)`
- `advance(progress)`
- `validate_route(route)`
- `validate_measurements(measurements, status, actions)`

## Constraints

- Anchor accepts 2-8 equal-length digits/targets, finite support_reference_m, and mounting_reference height matching support_reference_m within 1e-6; current_layout_source must be False and target_geometry_source must be one of the two public RGB circular-surface-family values.
- Digit sources use meters/radians/pixels and restrict cameras to cam_head, cam_left_wrist, cam_right_wrist; estimated_grasp_span_m is in (0, 0.08] and surface z must equal support_reference_m plus span within 1e-6; optional feature intervals must stay above the reference and at most 0.08 above it.
- Progress stages are 0-4 with final_stage 0-2, retry 0-1, cursor 0-count, and stage 0 when cursor equals count; routes restrict side left/right, digit 0-9, unit quaternions with norm 0.99-1.01, closing_axis/tilt whitelists, joint_limit_margin_rad at least 0.035, and finite bounded poses.
- Measurements require exact gripper_signal_semantics 'normalized_public_command_response_not_real_finger_width_or_contact', reject contact-sensor and official-success claims, require relay physical_collision_free_claimed=False, and delegate to imported validators/assign_numbers; failures raise ValueError.
