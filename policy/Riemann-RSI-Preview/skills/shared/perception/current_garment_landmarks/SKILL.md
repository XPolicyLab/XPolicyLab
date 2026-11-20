---
name: current_garment_landmarks
description: "Reads current cam_head RGB and the measured support plane to fit a shipped public garment outline, transform its public local landmarks, or, in a non-flat phase, replace only the two hem landmarks from the current silhouette. It also exposes current-mask contact refinement, wrist-RGB contact backprojection, and a lifted-surface colour check using the own-pad pose."
---

# Current garment landmarks

## Function

Reads current cam_head RGB and the measured support plane to fit a shipped public garment outline, transform its public local landmarks, or, in a non-flat phase, replace only the two hem landmarks from the current silhouette. It also exposes current-mask contact refinement, wrist-RGB contact backprojection, and a lifted-surface colour check using the own-pad pose.

## Inputs

- api adapter exposing image('cam_head'), image('cam_<side>_wrist'), projection/pixel-ray helpers, own_joint_states(), and the retained passive support service
- previous same-episode anchor optionally carrying head_outline (96x2), appearance_hsv, center, axes, height_m, and named landmarks
- phase string: 'flat' selects outline/template reading; any other value selects torso hem re-reading; reuse=True reuses retained flat axes and snaps cuffs, hems, and chests
- segment(image, seed) takes current RGB plus optional seed mask; wrist_contact also takes side ('left'/'right') and a finite nominal 3D point

## Outputs

- read(...) returns {anchor, alternatives, status, support_memory, rgb_geometry_is_hidden_surface}; status is 'observed' or 'not_observed', and torso_read may add torso_read_fallback or current_torso_bounds_local_m
- anchor contains template-derived center, longitudinal_axis, lateral_axis, confidence, eight named landmarks, source='current_public_rgb_garment_outline', head_outline, appearance_hsv, height_m, support_uncertainty_m, and support_memory
- supported_contact returns a 3D point or None; interior_contact returns a shifted 2D/3D point; wrist_contact returns a 3D point plus source/camera/samples/contact metadata
- lifted_surface_feedback returns status ('uncertain' or 'current_surface_follows_own_pad'), head_colour_fraction, wrist_colour_fraction, and gripper_measurement_is_contact=False

## Callable interface

Implementation: [implementation.py](implementation.py).

- `public_outline(points, counts, indices)`
- `public_shapes()`
- `template_mask(api, key, parameters, height)`
- `template_landmarks(api, observed, height)`
- `stable_candidates(candidates, previous)`
- `current_mask(image, outline)`
- `largest(mask, minimum)`
- `segment(image, seed=None)`
- `seeded_masks(image)`
- `outline_supported(api, previous, outline, image)`
- `plane_points(api, camera, pixels, height)`
- `numeric_outline(previous)`
- `retained_appearance_mask(image, previous)`
- `torso_observed(image, previous)`
- `torso_read(api, previous, plane, support, image, empty, retry=True)`
- `read(api, previous=None, reuse=False, phase='flat')`
- `supported_contact(api, anchor, name)`
- `interior_contact(api, anchor, point, direction, reach=0.04, step=0.002)`
- `wrist_contact(api, anchor, side, nominal, interior_axis=None)`
- `lifted_surface_feedback(api, anchor, sides)`

## Constraints

- Flat template fitting requires at least 1200 observed pixels, public-template IoU at least 0.62, and a finite observed-centre backprojection with at least 300 valid samples; otherwise read returns empty or reuses the previous estimate.
- A prompted outline is accepted only when exactly one non-clipped region is returned and it passes outline_supported; reuse snaps only cuffs, hems, and chests within 0.055 m and leaves shoulders from the fitted template.
- Non-flat torso_read preserves the retained flat estimate and updates only left_hem/right_hem; it requires finite retained axes/center and at least 300 silhouette samples and falls back to the retained anchor on incoherent segmentation.
- Contact and lifted checks are bound to the adapter api; supported_contact requires source='current_public_rgb_garment_outline', wrist_contact needs at least 20 valid wrist samples, lifted_surface_feedback's status is a current RGB/pose check rather than an official success verdict, and normalized or commanded gripper readings are not treated as physical contact.
