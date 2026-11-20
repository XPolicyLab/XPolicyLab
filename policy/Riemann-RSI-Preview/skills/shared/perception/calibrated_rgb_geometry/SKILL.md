---
name: calibrated_rgb_geometry
description: "Provides calibrated RGB projection, plane-ray unprojection, and closed-contour sampling. CalibratedRgbGeometry stores explicit center, rotation, focal, and principal values; sample_closed_contour densifies input hull edges."
---

# Calibrated rgb geometry

## Function

Provides calibrated RGB projection, plane-ray unprojection, and closed-contour sampling. CalibratedRgbGeometry stores explicit center, rotation, focal, and principal values; sample_closed_contour densifies input hull edges.

## Inputs

- center: length-3 camera center, rotation: 3x3 rotation, focal: scalar focal length, principal: length-2 principal point
- points: Nx3 world points for project
- uv: Nx2 image coordinates and z: scalar world-plane height for unproject
- hull: Nx2 closed contour vertices for sample_closed_contour

## Outputs

- project: Nx2 pixel coordinates
- unproject: Nx3 world points on the plane z
- sample_closed_contour: Mx2 points sampled along closed hull edges

## Callable interface

Implementation: [implementation.py](implementation.py).

- `CalibratedRgbGeometry: __init__(self, center, rotation, focal, principal); project(self, points); unproject(self, uv, z)`
- `sample_closed_contour(hull)`

## Constraints

- Constructor raises ValueError unless center shape is (3,), rotation shape is (3,3), and principal length is 2.
- project uses (points-center) @ rotation and divides by -transformed z without an explicit zero-depth guard.
- unproject intersects rays with the plane z=z using transformed x,y,-1.
- Closed contour sampling uses input hull only and at least two samples per edge.
