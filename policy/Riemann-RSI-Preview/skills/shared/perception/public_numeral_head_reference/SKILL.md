---
name: public_numeral_head_reference
description: "Validates, captures, retains, and checks compatibility of a bounded public head-image numeral reference containing independent recognized labels, head-image anchors, target centres, and a delivered-image SHA-256. Retention is limited to actual open camera-only commands."
---

# Public numeral head reference

## Function

Validates, captures, retains, and checks compatibility of a bounded public head-image numeral reference containing independent recognized labels, head-image anchors, target centres, and a delivered-image SHA-256. Retention is limited to actual open camera-only commands.

## Inputs

- value dict with schema 'riemann-public-number-head-reference-v1', semantics, digits, target_centres, and image_sha256
- capture(report, row, image) with report digits, row regions containing centroid, and the native head image
- memory event history for retained(memory)
- current report and current row for compatible(value, current, row)

## Outputs

- validate returns a deep-copied validated reference or raises ValueError
- capture returns a validated reference or None when report and row lengths mismatch or validation fails
- retained returns a validated reference from the latest qualifying executed camera-only number_scene_view or number_home inspect event, otherwise None
- compatible returns a bool requiring unchanged target centres within 4 pixels and still-visible current labels within 4 pixels

## Callable interface

Implementation: [implementation.py](implementation.py).

- `validate(value)`
- `capture(report, row, image)`
- `retained(memory)`
- `compatible(value, current, row)`

## Constraints

- digits length must be 2 to 8; labels must be unique ints 0 to 9; every digit camera must be cam_head; yaw_radians must be finite within plus or minus pi; anchors and target centres must be inside 640x480
- image_sha256 must be 64 lowercase hex characters and semantics must equal previous_delivered_head_RGB_during_open_camera_only_commands_not_hidden_state
- retained invalidates after a non-camera command, a number_home purpose other than inspect, or missing settle_public_view_after_robot_motion phase with actions greater than zero
- The reference is a bounded explicit static-scene observation belief, not privileged state or a contact guarantee, and no metric object pose or official result can be included
