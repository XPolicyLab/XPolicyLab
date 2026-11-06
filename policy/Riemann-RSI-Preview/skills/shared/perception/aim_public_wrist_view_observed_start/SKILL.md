---
name: aim_public_wrist_view_observed_start
description: "Computes a bounded own-wrist target pose using observed-start pose phases, pointing the declared wrist camera at a current head-image centroid and returning three ordered view phases with an allowed flag. It always caps additional viewing rotation at 0.8 rad and exposes no larger-budget parameter."
---

# Aim public wrist view observed start

## Function

Computes a bounded own-wrist target pose using observed-start pose phases, pointing the declared wrist camera at a current head-image centroid and returning three ordered view phases with an allowed flag. It always caps additional viewing rotation at 0.8 rad and exposes no larger-budget parameter.

## Inputs

- api: public adapter exposing targets(), home_targets(), and camera/pixel-ray helpers
- side: 'left' or 'right'; centroid: finite (x,y) pixel inside 640x480
- proposed_pose: optional shape-(7,) own pose with finite translation and unit orientation, translation within 0.15 m of current pose
- focus_height: optional finite focus height in metres

## Outputs

- target_pose returns a shape-(7,) target pose or None when the ray or rotation cannot be qualified
- phases returns (list of three pose_phase objects, allowed bool)

## Callable interface

Implementation: [implementation.py](implementation.py).

- `target_pose(api, side, centroid, proposed_pose=None, focus_height=None)`
- `phases(api, side, centroid)`

## Constraints

- Motion is allowed only when centroid is present, the chosen wrist is within 0.08 m of its observed home, and both public opening responses are >=0.9.
- Proposed pose must be finite shape (7,) with unit orientation and translation within 0.15 m; focus_height must be finite.
- A ray intersection with non-positive scale or rotation beyond 0.8 rad returns None; the focus is a view hypothesis, not a grasp coordinate.
- Callers must provide their own before-contact and open-response guards; stored layouts are not inputs.
