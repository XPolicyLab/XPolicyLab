---
name: public_numeral_surface_reference
description: "Captures, validates, retains, and combines a bounded partial family of previously qualified public numeral surfaces only across acknowledged open camera-only commands. Remembered surfaces enter current anchors only when newly delivered cameras independently recognize the same label and the current ray at the measured height agrees within 8 mm."
---

# Public numeral surface reference

## Function

Captures, validates, retains, and combines a bounded partial family of previously qualified public numeral surfaces only across acknowledged open camera-only commands. Remembered surfaces enter current anchors only when newly delivered cameras independently recognize the same label and the current ray at the measured height agrees within 8 mm.

## Inputs

- Adapter-bound api exposing own_robot_bases_available(), own_robot_bases(), targets(), command_memory(), and public camera projection behavior
- surfaces list, row regions with centroid, and support reference in meters for capture
- current row pixels and current support reference for retained
- reports keyed by cam_head, cam_left_wrist, and cam_right_wrist plus fresh surfaces for combine

## Outputs

- validate returns a detached validated memory dict or raises ValueError
- capture returns validated memory or None when surfaces are empty, row is None, or own robot bases are unavailable
- retained returns compatible pre-contact memory or None
- combine returns (currently qualified surfaces, retained surfaces), sorted by descending digit

## Callable interface

Implementation: [implementation.py](implementation.py).

- `validate(value)`
- `capture(api, surfaces, row, reference)`
- `retained(api, row, reference)`
- `combine(api, reports, fresh, previous)`

## Constraints

- Memory schema is riemann-public-numeral-surface-reference-v1; support_reference_m must be finite with absolute value at most 3; target_centres length is 2 to 8; surfaces length is 1 to target_centres length; both own base poses must be finite 7-vectors in XYZ/WXYZ with near-unit quaternion
- Each surface requires digit, surface_xyz, estimated_grasp_span_m in (0, 0.08], finite yaw within plus or minus pi, public_feature_height_interval_m within reference to reference + 0.08, maximum_ray_separation_m between 0 and 0.004, 2 to 3 unique public cameras, and a declared height_semantics value
- retained requires open grippers with targets at least 0.9, unchanged row pixels within 4 pixels, support reference within 0.005 m, base translation within 0.001 m, and quaternion dot at least 1 - 1e-6; any acknowledged non-camera action ends reuse
- combine clears saved surfaces on same-label disagreement greater than 0.008 m or memory z peak-to-peak greater than 0.012 m; unseen measurements may remain dormant but never fill a current anchor, and no asset geometry or official result is used
