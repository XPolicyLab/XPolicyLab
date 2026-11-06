---
name: calibrated_rgb_rays
description: "Defines Camera(calibration).project(points) for calibrated image projection and Camera(calibration).plane(pixels,height) for public image-ray intersections with a horizontal plane. It uses declared static head calibration and image-index coordinates."
---

# Calibrated rgb rays

## Function

Defines Camera(calibration).project(points) for calibrated image projection and Camera(calibration).plane(pixels,height) for public image-ray intersections with a horizontal plane. It uses declared static head calibration and image-index coordinates.

## Inputs

- calibration: mapping with head_origin_m, head_euler_deg, and head_focal_pixels
- points: world points for Camera.project
- pixels: image-index coordinates and height: caller-observed world z plane for Camera.plane

## Outputs

- project: pixel coordinates using principal point (319.5,239.5)
- plane: world points on the horizontal plane z=height

## Callable interface

Implementation: [implementation.py](implementation.py).

- `Camera: __init__(self, calibration); project(self, points); plane(self, pixels, height)`

## Constraints

- project raises ValueError for projection behind the declared public camera when local z >= -1e-6.
- plane raises ValueError when |r_z| < 1e-7 or the plane hypothesis is behind the camera (d <= 0).
- Continuous optical centres include half pixels; principal point is (319.5,239.5).
- No environment/camera handles or depth; plane height is caller-observed, never a default world/table coordinate.
