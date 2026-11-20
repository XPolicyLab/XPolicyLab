---
name: garment_native_feedback_boundary
description: "Validates a complete garment phase-feedback packet against the declared garment action schema and the isolated observed_garment_mount_state measurement contract, returning a deep copy. It rejects other action domains and enforces the original two- or sixteen-action garment minimum."
---

# Garment native feedback boundary

## Function

Validates a complete garment phase-feedback packet against the declared garment action schema and the isolated observed_garment_mount_state measurement contract, returning a deep copy. It rejects other action domains and enforces the original two- or sixteen-action garment minimum.

## Inputs

- request: action request validated by harness.runtime.manipulation.validate_action_parameters
- feedback: dict with exact phase-feedback fields including phase_reports, image_measurements, and visual_lift_feedback
- request fields used: skill, side, steps, count

## Outputs

- deepcopy of validated feedback
- Raises harness.execution.ExecutionError for domain/schema/side/status/budget/evidence violations

## Callable interface

Implementation: [implementation.py](implementation.py).

- `validate_feedback(request, feedback)`

## Constraints

- Only skills in observed_garment_mount_state.ACTIONS are accepted; automatic side is allowed only when the declared side enum contains 'auto'
- Feedback cannot claim object grasp/contact or official success; visual_review_required must be true and object_grasp_confirmed/official_success_claimed false
- Gripper fields must be finite normalized values in [0,1] and are not measured fingers; issued/qualified cycles are bounded by request count and are not official button counts
- Image measurements must match observed_garment_mount_state.measurements plus declared flags/angles; after outer validation required_minimum_actions must be 2 or 16
