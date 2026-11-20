---
name: aim_public_wrist_view
description: "Computes a bounded own-wrist target pose that points the declared wrist camera at a current head-image centroid and returns three ordered view phases with an allowed flag. target_pose predicts the wrist camera after an optional proposed own pose and caps additional viewing rotation at 0.8 rad by default or 1.6 rad only under the explicit public-opening guard."
---

# Aim public wrist view

## Function

Computes a bounded own-wrist target pose that points the declared wrist camera at a current head-image centroid and returns three ordered view phases with an allowed flag. target_pose predicts the wrist camera after an optional proposed own pose and caps additional viewing rotation at 0.8 rad by default or 1.6 rad only under the explicit public-opening guard.

## Inputs

- api: public adapter exposing targets(), home_targets(), and camera/pixel-ray helpers
- side: 'left' or 'right'; centroid: finite (x,y) pixel inside 640x480
- proposed_pose: optional shape-(7,) own pose with finite translation and unit orientation, translation within 0.15 m of current pose
- focus_height: optional finite focus height in metres; maximum_rotation: 0.8 or 1.6

## Outputs

- target_pose returns a shape-(7,) target pose or None when the ray or rotation cannot be qualified
- phases returns (list of three pose_phase objects, allowed bool)

## Callable interface

Implementation: [implementation.py](implementation.py).

- `target_pose(api, side, centroid, proposed_pose=None, focus_height=None, maximum_rotation=0.8)`
- `phases(api, side, centroid)`

## Constraints

- The 1.6-rad orientation budget is allowed only when both public opening responses are >=0.9; it is an own-camera aim bound, not an object pose or contact inference.
- The three-phase helper allows motion only when centroid is present, the chosen wrist is within 0.08 m of its observed home, and both public openings are open.
- A ray intersection with non-positive scale or an angle beyond the cap returns None; the focus is a view hypothesis, not a grasp coordinate.
- Callers requesting the larger before-contact view must provide their own before-contact and open-response guards and prove no grasp has started and no held-side continuation exists.
