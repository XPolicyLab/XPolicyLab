---
name: public_camera_projection
description: "Projects bounded world point sets through the public camera ray model and intersects public pixel rays with a caller-supplied horizontal plane."
---

# Public camera projection

## Function

Projects bounded world point sets through the public camera ray model and intersects public pixel rays with a caller-supplied horizontal plane.

## Inputs

- Public robot API, camera name and one to 1024 finite XYZ points or XY pixels.
- For plane intersection, an explicit finite candidate height.

## Outputs

- N-by-2 projected pixel coordinates or N-by-3 plane intersections, with NaNs for points behind the camera or rays without a forward intersection.

## Callable interface

Implementation: [implementation.py](implementation.py).

- `project_points(api, camera, points)`
- `points_on_plane(api, camera, pixels, height)`

## Constraints

- Raises ValueError for malformed, nonfinite or oversized point sets.
- Camera geometry comes from pixel_rays and the current image dimensions. A supplied plane height is a hypothesis, not an independently measured surface.
