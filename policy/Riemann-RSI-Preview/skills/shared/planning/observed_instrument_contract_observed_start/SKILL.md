---
name: observed_instrument_contract_observed_start
description: "Validates public instrument anchor, eight-bar target family, tool frame, route, progress, attachment evidence, gripper semantics, and mature execution receipts. It deep-copies accepted evidence and returns the set of declared measurement fields present."
---

# Observed instrument contract observed start

## Function

Validates public instrument anchor, eight-bar target family, tool frame, route, progress, attachment evidence, gripper semantics, and mature execution receipts. It deep-copies accepted evidence and returns the set of declared measurement fields present.

## Inputs

- measurements: dict with optional public_instrument_anchor, public_instrument_progress, public_instrument_route, public_instrument_partial_bars, instrument_attachment_evidence, mature_execution_receipt, gripper_signal_semantics, and command_phases_completed
- status and actions: command status string and nonnegative issued official action count
- anchor schema riemann-public-instrument-anchor-v1 with tool_frame schema riemann-public-spherical-tool-frame-v1, initial_cap_image, and exactly eight bars
- route and attachment evidence: bounded own-command poses, normalized working_orientation, reviewed settings, and public attachment-belief schema

## Outputs

- set of recognized declared measurement field names
- deep-copied validated anchor, bars, route, progress, and attachment evidence when present
- empty_progress() returns {stage: 0, cursor: 0, attempt: 0}

## Callable interface

Implementation: [implementation.py](implementation.py).

- `vector(value, size, maximum=3.0)`
- `cameras(value)`
- `validate_frame(value)`
- `validate_bars(value)`
- `validate_anchor(value)`
- `empty_progress()`
- `validate_progress(value)`
- `validate_route(value)`
- `validate_attachment_evidence(value)`
- `validate_measurements(measurements, status, actions)`

## Constraints

- Rejects unknown fields, non-finite/out-of-range values, non-unique camera lists, invalid schemas, filled-in shaft extent, asset pose, contact measurement, ground truth, or official-success claims.
- Gripper response must be normalized_public_command_response_not_real_finger_width_or_contact; attachment is prediction consistency, not real grasp or contact.
- mature_execution_receipt official_action_acknowledgments must equal actions; completed phases require positive actions and status completed or uncertain.
- Uses imported validate_receipt from mature_feedback_execution_observed_start and checked_pose from rigid_tool_command_tracking_observed_start; no environment access or task controller import.
