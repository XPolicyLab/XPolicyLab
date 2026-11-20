---
name: observed_garment_mount_state
description: "Validates and deep-copies a closed same-instruction public garment state whose anchor support may be public support geometry or a garment mounting datum. It reconstructs the latest recorded garment state from recent action feedback and checks the closed stage measurement receipt."
---

# Observed garment mount state

## Function

Validates and deep-copies a closed same-instruction public garment state whose anchor support may be public support geometry or a garment mounting datum. It reconstructs the latest recorded garment state from recent action feedback and checks the closed stage measurement receipt.

## Inputs

- Garment state mapping with schema riemann-public-garment-state-v1, 64-character lowercase hex instruction_fingerprint, phase, part, chunk, attempts, anchor, route and start_wrists
- Anchor mapping with source current_public_rgb_garment_outline, bounded 2D center and landmarks, orthogonal normalized axes, confidence, height_m, support_uncertainty_m, 96x2 head_outline, appearance_hsv and support_memory
- Recent memory event list and instruction string for recorded(memory, instruction)
- Measurement mapping, status and action_count for measurements(value, status, action_count)

## Outputs

- Deep-copied validated garment state
- Initial state from initial(instruction) with phase view and zero counters
- Recorded state, or the initial state when no matching same-instruction record is found
- Set of the four closed extra measurement keys returned by measurements

## Callable interface

Implementation: [implementation.py](implementation.py).

- `vector(value, length, bound=3.0)`
- `pose(value)`
- `anchor(value)`
- `route(value)`
- `initial(instruction)`
- `validate(value)`
- `recorded(memory, instruction)`
- `measurements(value, status, action_count)`

## Constraints

- State fields are closed; phase must be in the PHASES tuple including prehome_settle, part/chunk/attempts are bounded integers, route arms are one or two distinct own arms, contact_poses have unit quaternions, approach_joints are bounded by 10, and carry/body chunks are 1 or 2
- support_uncertainty_m may be None only when support_memory source equals MOUNTING_DATUM_SOURCE and height_m equals the validated datum mount_z_m within 1e-8; otherwise support_memory is validated by public_support_geometry_context and the datum is not treated as an observed tabletop or contact measurement
- Garment visual review status is limited to uncertain or current_surface_follows_own_pad, gripper_measurement_is_contact must be false, and head/wrist colour fractions are finite in [0,1]
- garment_clock must equal segment_width 16 with trajectory_resampled false and true_finger_measurements_used false; command_phases_completed requires completed or uncertain status and action_count at least 16
