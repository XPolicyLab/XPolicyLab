---
name: observed_liquid_execution
description: "Validates and deep-copies a finite public liquid skill state containing current free multiview vessel profiles, own command goals, attachment hypothesis and tilt samples. It reconstructs the latest recorded state and checks fixed-step execution receipts and public review without asserting contact or official success."
---

# Observed liquid execution

## Function

Validates and deep-copies a finite public liquid skill state containing current free multiview vessel profiles, own command goals, attachment hypothesis and tilt samples. It reconstructs the latest recorded state and checks fixed-step execution receipts and public review without asserting contact or official success.

## Inputs

- Public liquid state mapping with stage, scene, side and support_side, route, attachment, tilt_poses, bounded attempt counters, actions_seen, reason, official_success_claimed and view_support
- Liquid scene mapping with source current_independently_qualified_liquid_pair, bottle profile, cup_opening, optional cup_profile, support_estimate_m and support_uncertainty_m
- Recent memory event list for recorded(memory)
- Measurement mapping with public_liquid_state, native_execution_receipt and public_liquid_review, plus status and actions for measurements(value, status, actions)

## Outputs

- Deep-copied validated liquid state
- Empty state from empty() with stage view and zero counters
- Recorded state, or the empty state when no matching action record is found
- Set of the three required measurement keys returned by measurements

## Callable interface

Implementation: [implementation.py](implementation.py).

- `profile(value)`
- `scene(value)`
- `empty()`
- `state(value)`
- `recorded(memory)`
- `measurements(value, status, actions)`

## Constraints

- Profiles use source current_independent_multiview_free_axial_profile, forbid physical_asset_geometry_used and object_ground_truth_used, allow six legacy or eight observed radii with ordered normalized height fractions, bound radii to 0.001..0.18 m, and require 2-3 distinct delivered cameras
- Scene source must be current_independently_qualified_liquid_pair; the scene support estimate must agree with the bottle support estimate within 0.006 m, and the cup opening center height above support must be in 0.006..0.4 m
- Attachment source must be current_spout_motion_and_own_wrist_feedback with attachment_is_hypothesis true; state and review keep official_success_claimed false and do not treat gripper values as contact
- Native receipt schema riemann-original-fixed-step-execution-v1 requires native_chunk_maximum 16, constant original normalized gripper opening, no trajectory sampling reapplication, no true finger or contact reads, no official success, bounded counters matching actions, and 16 commands per endpoint settle interval
