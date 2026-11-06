---
name: read_number_scene_observed_start
description: "measure composes current outline reads from three public cameras, a uniquely supported destination row, historical public inventory bounds, numeral-surface triangulation, and a circular target family to return a complete current scene anchor or an explicit uncertain report. Helper functions fresh_source and placement relocalize a remembered digit and test a visual placement hypothesis without claiming official success."
---

# Read number scene observed start

## Function

measure composes current outline reads from three public cameras, a uniquely supported destination row, historical public inventory bounds, numeral-surface triangulation, and a circular target family to return a complete current scene anchor or an explicit uncertain report. Helper functions fresh_source and placement relocalize a remembered digit and test a visual placement hypothesis without claiming official success.

## Inputs

- measure(api, extended=False, head_reference=None, target_reference=None, inventory_reference=None): adapter plus optional retained public head, target, and inventory references
- api: must expose instruction(), own_robot_bases_available(), own_robot_bases(), image(camera), command_memory(), and the shared perception/geometry helpers used by imported modules
- extended: bool passed to read_outlines for each of cam_head, cam_left_wrist, and cam_right_wrist
- inventory_reference: optional prior inventory matching riemann-public-number-inventory-v1 and its declared semantics

## Outputs

- measure: dict with kind 'number_sorting_scene', status 'uncertain' or 'metric_observed', reason on uncertainty, visible head digits/anchors/target count, public_number_inventory, and optional retained references
- on success anchor contains schema 'riemann-current-number-scene-v1', digits, targets sorted by x, support_reference_m, mounting_reference, target_geometry_source, observed_target_count, and current_layout_source false
- fresh_source returns a relocalized copy of a remembered digit surface or None; placement returns status 'observed'/'uncertain', visual_placement_confirmed, horizontal_hypothesis_error_m, upright_visual_outline, camera evidence, and official_success_claimed false
- raises ValueError for invalid inventory schema/count/labels, row length outside 2..8, report labels over 24, non-native head RGB, or nonfinite/missing three-camera numeral-surface inputs; target-capture ValueError is converted to a coded uncertain reason

## Callable interface

Implementation: [implementation.py](implementation.py).

- `validate_inventory(value)`
- `public_inventory(memory=(), row=None, report=None, previous=None)`
- `applicable(instruction)`
- `pad_regions(image)`
- `target_row(regions)`
- `numeral_surfaces(api, reports, reference_height, partial=False)`
- `measure(api, extended=False, head_reference=None, target_reference=None, inventory_reference=None)`
- `fresh_source(api, anchor, digit)`
- `placement(api, anchor, digit, target_index)`

## Constraints

- measure returns uncertain rather than filling missing geometry when instruction is not arrange/number/largest/descending, own FK is unavailable, mounting reference is uncertain, destination row is missing/ambiguous, inventory needs reobservation, numeral/target counts disagree, or target family validation/capture fails
- The declared task mounting contract places fixed upright X5 roots on the source supporting surface; triangulated surfaces use metres above the own mounting reference, per-digit evidence within 0.006 m, height intervals above reference and at most 0.08 m, and surface z spread at most 0.012 m
- public_inventory retains only a bounded historical cardinality lower bound and independently read labels; validate_inventory deep-copies and rejects malformed schema/semantics/minimum_count/observed_labels
- placement uses target z plus estimated_grasp_span_m, checks horizontal distance <= 0.055 m and upright yaw within pi/4 across available current cameras; point semantics are a current-RGB on remembered-surface hypothesis, not object ground truth, and official_success_claimed remains false
