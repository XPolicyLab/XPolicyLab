---
name: surface_family_correspondence
description: "select_family builds one-to-one head/wrist-view associations from bounded visual candidates, infers common level and size, rejects ambiguous families, and returns family geometry. run triangulates current head/wrist regions through api, filters elevation, projects head surfaces, and returns the family plus surfaces."
---

# Surface family correspondence

## Function

select_family builds one-to-one head/wrist-view associations from bounded visual candidates, infers common level and size, rejects ambiguous families, and returns family geometry. run triangulates current head/wrist regions through api, filters elevation, projects head surfaces, and returns the family plus surfaces.

## Inputs

- select_family: candidates list of dicts with head_index, wrist_index, camera in ('cam_left_wrist','cam_right_wrist'), point 3-vector, size_m, ray_separation_m; head_count 2-8; observed_planar_row bool
- run: api, head_regions list of 2-8 regions with centroid and bbox, detector callable, projector callable, support_plane_m
- run optional: minimum_elevation_m=0.005, maximum_elevation_m=0.3, observed_planar_row=false

## Outputs

- select_family: family dict with height_m, size_m, observed_head_indices, wrist_views, and optional row_correspondence_evidence; or None
- run: same family plus surfaces list, or None

## Callable interface

Implementation: [implementation.py](implementation.py).

- `select_family(candidates, head_count, observed_planar_row=False)`
- `run(api, head_regions, detector, projector, support_plane_m, minimum_elevation_m=0.005, maximum_elevation_m=0.3, observed_planar_row=False)`

## Constraints

- select_family raises ValueError for head_count outside 2-8, >256 candidates, non-bool flag, invalid indices/cameras, non-finite/out-of-bounds points, size outside 0.005-0.12 m, or ray separation outside 0-0.005 m.
- run raises ValueError for invalid support/elevation bounds or >16 wrist detections; rejects unsupported candidates and returns None on missing/ambiguous family or inconsistent projected surface radii.
- Ambiguity thresholds are 0.015 m normally and 0.003 m for partial single-wrist families; partial families require observed planar row, >=max(3, head_count-1) heads, height ptp <=0.002, size ratio <=1.12, and ray separation <=0.0015.
- Result is uncertain visual geometry, not contact, grasp success, or official completion; consumers must recheck current visibility and same-episode geometry.
