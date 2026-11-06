---
name: observed_number_contract_observed_start
description: "Validates a complete independently observed numeral/target anchor, bounded own-command progress, and public measurement receipts; it delegates nested inventory, surface, target, head, route, and feedback checks to imported validators. It returns deep copies of anchor/progress and validates measurements without returning a value."
---

# Observed number contract observed start

## Function

Validates a complete independently observed numeral/target anchor, bounded own-command progress, and public measurement receipts; it delegates nested inventory, surface, target, head, route, and feedback checks to imported validators. It returns deep copies of anchor/progress and validates measurements without returning a value.

## Inputs

- anchor: exact riemann-current-number-scene-v1 dict with schema, digits, targets, support_reference_m, mounting_reference, target_geometry_source, observed_target_count, current_layout_source
- progress: dict with integer cursor, stage, count, final_stage, retry
- measurements: dict of public_number_* records, status, and nonnegative actions
- digit source: surface_xyz, estimated_grasp_span_m, yaw_radians, maximum_ray_separation_m, cameras, height_semantics, and optional relocalization or public height-interval fields

## Outputs

- validate_anchor(anchor): deep-copied validated anchor
- validate_progress(progress, count): deep-copied validated progress
- empty_progress(count=0): initial progress dict; advance(progress): deep-copied validated progress with cursor/stage/retry advanced
- validate_measurements(measurements, status, actions): no return value

## Callable interface

Implementation: [implementation.py](implementation.py).

- `validate_anchor(anchor)`
- `empty_progress(count=0)`
- `validate_progress(progress, count)`
- `advance(progress)`
- `validate_measurements(measurements, status, actions)`

## Constraints

- Rejects unknown anchor/progress/measurement fields and non-whitelisted numeral sources; target_geometry_source must be whitelisted and current_layout_source must be False
- estimated_grasp_span_m must be >0 and <=0.08 m; surface_xyz must be finite/bounded and z tied to support_reference_m plus span; relay/site evidence has fixed numeric bounds
- Progress is own-command state only: count must match, 0<=cursor<=count, 0<=stage<5, final_stage<=2, retry<=1; cursor==count requires stage 0
- gripper_signal_semantics must be normalized_public_command_response_not_real_finger_width_or_contact; possible_support_is_contact_sensor must be False; policy checks are not official success
