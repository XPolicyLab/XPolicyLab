---
name: public_planar_surface
description: "Qualifies a same-colour planar silhouette by sampling region pixels, projecting wrist rays to a height, triangulating head and wrist centroids, and checking head-mask coverage. It can reject ambiguous candidates without inserting a remembered object height."
---

# Public planar surface

## Function

Qualifies a same-colour planar silhouette by sampling region pixels, projecting wrist rays to a height, triangulating head and wrist centroids, and checking head-mask coverage. It can reject ambiguous candidates without inserting a remembered object height.

## Inputs

- Adapter-bound api exposing image(camera) and public camera projection behavior for points_on_plane, project_points, and triangulate
- head_region with centroid, pixels, and mask; regions(camera) callable for cam_left_wrist and cam_right_wrist
- height in meters for coverage and optional destination_camera defaulting to cam_head

## Outputs

- coverage returns the fraction of projected valid world points inside the dilated destination mask, or 0.0
- centroid_surface returns the chosen stereo dict with point, ray_separation_m, and surface_coverage, or None

## Callable interface

Implementation: [implementation.py](implementation.py).

- `coverage(api, camera, region, head_region, height, *, destination_camera='cam_head')`
- `centroid_surface(api, head_region, regions)`

## Constraints

- Samples up to 512 region pixels and requires at least 16 finite world points plus nonempty projected points; coverage uses a 5x5 dilation and counts inside-image projected points
- A candidate requires coverage at least 0.84; candidates are sorted by descending coverage then ray_separation_m; if top candidate heights differ by more than 0.008 m, mutual head-to-wrist reprojection coverage at least 0.84 is required or the result is None
- Rejection does not insert a remembered object height; a matching centroid is only a qualified planar hypothesis, not contact or object identity ground truth
- Uses only already captured RGB and own-camera calibration, compares at most two wrist images and a bounded region sample, and makes no official success claim
