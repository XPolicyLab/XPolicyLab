---
name: t_silhouette
description: "Recovers a signed T-silhouette from a finite 2-D point set by PCA, comparing transverse spans at the lowest and highest projection quantiles and orienting the axis toward the wider crossbar. The pose wrapper samples up to 1024 image points, projects them onto the cam_head plane at top_height, and returns a yaw pose only when the measured thickness is 3-40 mm."
---

# T silhouette

## Function

Recovers a signed T-silhouette from a finite 2-D point set by PCA, comparing transverse spans at the lowest and highest projection quantiles and orienting the axis toward the wider crossbar. The pose wrapper samples up to 1024 image points, projects them onto the cam_head plane at top_height, and returns a yaw pose only when the measured thickness is 3-40 mm.

## Inputs

- horizontal: finite Nx2 array with at least 30 points
- pixels: finite Nx2 image points
- top_height: measured top-surface height in metres
- support_height: public support-height hypothesis in metres

## Outputs

- signature: dict with center, signed_axis, yaw_radians and width_ratio, or None
- pose: dict with a 7-vector pose, visible_thickness_m, signed_axis and source, or None

## Callable interface

Implementation: [implementation.py](implementation.py).

- `signature(horizontal)`
- `pose(api, pixels, top_height, support_height)`

## Constraints

- Requires at least 30 finite Nx2 points; low/high transverse widths must be positive and their ratio at least 1.4
- Thickness top_height - support_height must be between .003 and .04 m
- pose depends on the adapter api through points_on_plane(api, 'cam_head', ...); it does not use asset half-height, stored pose or layout
- Invalid rays, missing shape asymmetry, ambiguous height or failed projection return None
