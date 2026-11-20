---
name: conditional_payload_placement_rgb
description: "Computes a corrected placement EE pose from an observed source pose, public pre-close grasp EE pose, requested EE pose, destination XY, support top Z, and declared box half-extents. It preserves requested orientation and shifts the EE target so the payload's oriented vertical half-extent rests on the support."
---

# Conditional payload placement rgb

## Function

Computes a corrected placement EE pose from an observed source pose, public pre-close grasp EE pose, requested EE pose, destination XY, support top Z, and declared box half-extents. It preserves requested orientation and shifts the EE target so the payload's oriented vertical half-extent rests on the support.

## Inputs

- source_pose: finite 7-vector observed source object pose with nonzero WXYZ quaternion
- grasp_ee_pose: finite 7-vector public pre-close EE pose with nonzero WXYZ quaternion
- requested_ee_pose: finite 7-vector commanded placement pose with nonzero WXYZ quaternion
- destination_xy: finite 2-vector; support_top_z: finite scalar in meters; half_extents: finite 3-vector of positive payload half-dimensions in meters

## Outputs

- result: 7-vector corrected EE pose with requested orientation preserved
- metadata dict with source label, rigid-pose assumption, attachment_evidence='unknown', physical_finger_measurements_used=False, relative_center_in_EE, expected_payload_center, oriented_vertical_half_extent_m, requested_EE_position, corrected_EE_position, commanded_orientation_preserved=True

## Callable interface

Implementation: [implementation.py](implementation.py).

- `run(source_pose, grasp_ee_pose, requested_ee_pose, *, destination_xy, support_top_z, half_extents)`

## Constraints

- Validates finite shapes and nonzero quaternion norms; raises ValueError for bad poses, destination_xy, support_top_z, or nonpositive half_extents.
- Relative pose is a conditional rigid grasp estimate; it does not prove attachment or contact.
- No environment or score API is read; caller keeps existing grasp feedback and trajectory/terminal checks.
- If the grasp anchor is unavailable, caller must keep the original motion rather than invent a measurement.
