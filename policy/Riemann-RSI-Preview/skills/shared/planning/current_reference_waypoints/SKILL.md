---
name: current_reference_waypoints
description: "Builds an own transfer route from a current measured surface section and a destination reference frame, using current section ranking, closing orientation, wrist-contact pose and reference-frame transfer. It returns a scene with pregrasp, contact, lift, carry, align, place and retract targets, or None when no surface sections are available."
---

# Current reference waypoints

## Function

Builds an own transfer route from a current measured surface section and a destination reference frame, using current section ranking, closing orientation, wrist-contact pose and reference-frame transfer. It returns a scene with pregrasp, contact, lift, carry, align, place and retract targets, or None when no surface sections are available.

## Inputs

- api: bound adapter with targets() and issued_action_count()
- source: dict with surface points_xyz, rgb_samples, height_estimate_m and anchor with reference_pose, center_m and coverage
- destination: target reference pose; target_outline: destination outline data; observed_support: dict with height_estimate_m
- options: forward=.155, attempt=0, side='auto', tilt_degrees=None, approach_height=.055, lift_height=.15

## Outputs

- scene dict with source_anchor, source_reference, target_reference, target_outline, support_memory, side, matching_confidence, observation_epoch and route targets
- None when candidate_sections returns no sections

## Callable interface

Implementation: [implementation.py](implementation.py).

- `run(api, source, destination, target_outline, observed_support, forward=0.155, attempt=0, side='auto', tilt_degrees=None, approach_height=0.055, lift_height=0.15)`

## Constraints

- approach_height and lift_height must be finite and in (0,.2] m
- side='auto' selects the arm whose current ee pose is nearer anchor center; attempt selects sections[attempt % min(3,len(sections))] and default tilt 15 or 25 degrees
- Contact z is max(plane+.002, mid of section min/max z); carry/align add .135 m, place adds .005 m and retract adds .07 m
- Attachment after transfer is a hypothesis to verify; observation retries must not alter the physical grasp-attempt index
