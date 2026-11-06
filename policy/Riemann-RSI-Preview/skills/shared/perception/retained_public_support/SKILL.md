---
name: retained_public_support
description: "validate checks a before-contact support memory record, while validate_geometry accepts the same bounded fields only when the source is either the before-contact source or current independent public RGB support geometry. run calls the current public dense-plane alignment support, updates memory from consistent qualified evidence, or retains previous support only under explicit passive before-contact conditions."
---

# Retained public support

## Function

validate checks a before-contact support memory record, while validate_geometry accepts the same bounded fields only when the source is either the before-contact source or current independent public RGB support geometry. run calls the current public dense-plane alignment support, updates memory from consistent qualified evidence, or retains previous support only under explicit passive before-contact conditions.

## Inputs

- api: adapter object providing current_support(api), targets(), own_robot_bases_available(), and own_robot_bases()
- previous: optional support record accepted by validate, or None
- passive_view: explicit boolean before-contact execution condition

## Outputs

- dict with support, memory, and reason; support status is observed or uncertain and plane may be None
- memory: validated deep-copied support record or None
- reason: implementation string such as current_independent_public_support, no_qualified_before_contact_support, or a contradiction reason

## Callable interface

Implementation: [implementation.py](implementation.py).

- `validate(value)`
- `validate_geometry(value)`
- `validated_support(value, expected_source)`
- `run(api, previous=None, passive_view=False)`

## Constraints

- validate requires exactly height_estimate_m, height_uncertainty_m, camera_alignments, own_base_poses, and source; height_estimate_m must be in [-3, 3] m and height_uncertainty_m in [0, 0.006] m.
- Retained memory requires two alignment receipts for cam_left_wrist and cam_right_wrist passing the unchanged qualified_alignments check; mean accepted height must be within 0.003 m of height_estimate_m, and own base poses are bounded 7-vectors with quaternion norm in [0.99, 1.01].
- run retains previous support only when passive_view is true, own robot bases are available, both normalized gripper command/feedback values are >= 0.9, base positions stay within 0.004 m, quaternion dot is >= 0.999, and current qualified evidence does not differ by more than 0.006 m.
- Normalized gripper values are conservative command/feedback preconditions, not physical finger width, contact, or grasp measurements; the result does not claim official success.
