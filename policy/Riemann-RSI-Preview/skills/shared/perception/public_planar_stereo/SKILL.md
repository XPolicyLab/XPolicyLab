---
name: public_planar_stereo
description: "Class PublicPlanarStereo builds head and wrist camera transforms from public calibration and a fixed URDF camera mounting chain, then estimates a horizontal plane height by plane-induced homography texture matching across current head and wrist RGB. It raises RGBGeometryError when no unique qualified metric plane peak exists."
---

# Public planar stereo

## Function

Class PublicPlanarStereo builds head and wrist camera transforms from public calibration and a fixed URDF camera mounting chain, then estimates a horizontal plane height by plane-induced homography texture matching across current head and wrist RGB. It raises RGBGeometryError when no unique qualified metric plane peak exists.

## Inputs

- calibration dict with head_euler_deg, head_origin_m, wrist_sensor_euler_deg, wrist_sensor_origin_m, head_focal_pixels, and wrist_focal_pixels
- robot_urdf path containing a fixed camera mounting chain ending at link6
- images dict for cam_head, cam_left_wrist, and cam_right_wrist containing current 480x640x3 uint8 RGB
- state with finite 7-vector left_ee_pose and right_ee_pose in XYZ/WXYZ order, a 480x640 bool mask, optional lower and upper bounds in meters, minimum_support=300, and minimum_score=0.28

## Outputs

- estimate returns height_m, score, support_pixels, view, competing_height_margin, rgb_only=True, robot_base_height_used=False, and horizontal_surface_hypothesis=True
- The constructor exposes a mount transform, and cameras(state) returns 4x4 camera transforms for head, left wrist, and right wrist

## Callable interface

Implementation: [implementation.py](implementation.py).

- `RGBGeometryError: exception or data type`
- `PublicPlanarStereo: __init__(self, calibration, robot_urdf); cameras(self, state); estimate(self, images, state, mask, *, lower=None, upper=None, minimum_support=300, minimum_score=0.28)`

## Constraints

- The declared camera mounting chain must be fixed and no longer than 8 joints ending at link6; an unavailable or non-fixed chain raises ValueError
- Default search bounds are from the highest camera z minus 1.0 m to the highest camera z minus 0.03 m; bounds must be finite with low < high and range at most 2.0 m
- Search uses at most 512 coarse heights at 0.004 m spacing and 25 refinement samples per up to two peaks; acceptance requires support_pixels at least 300, score at least 0.28, and competing_height_margin at least 0.045
- It never falls back to robot-base Z and uses no camera depth, scene API, object identity, contact, or evaluator truth; insufficient overlap or lack of a unique texture peak raises RGBGeometryError
