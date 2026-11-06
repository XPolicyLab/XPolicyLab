---
name: silhouette_volume
description: "Validates two Boolean masks and two pinhole camera calibrations, performs one bounded two-view silhouette fit against an explicit support height, and returns an observed height above support or UNKNOWN."
---

# Silhouette volume

## Function

Validates two Boolean masks and two pinhole camera calibrations, performs one bounded two-view silhouette fit against an explicit support height, and returns an observed height above support or UNKNOWN.

## Inputs

- head_mask, wrist_mask: explicit public Boolean masks with shape (480,640) and dtype bool
- head_camera, wrist_camera: each a pair (pose, focal), where pose is a finite 4x4 matrix and focal > 0
- support_z: explicit public support-plane height in metres

## Outputs

- {'state':'observed','height_above_support_m': height in metres,'match': float,'views': int,'kind': estimate kind,'height_resolution_m':0.002,'physical_depth_or_pose_claimed':false}
- {'state':'unknown','reason':'two_view_surface_unqualified'}

## Callable interface

Implementation: [implementation.py](implementation.py).

- `run(head_mask, wrist_mask, head_camera, wrist_camera, support_z)`

## Constraints

- Raises ValueError for mask shape/dtype mismatch or non-finite/negative camera calibration.
- Returns unknown if either mask has fewer than 20 true pixels, estimate is None, views < 2, match < 0.85, or height is non-finite or outside 0.004-0.16 m.
- Performs one bounded fit using View/Sight/measure from .geometry; result is a shape hypothesis, not measured depth, possession, or task success.
- Consumers retain original motion on UNKNOWN; initially used only for the sorting car's demonstrated above-object grasp defect.

## Provenance

Generic fitting closure selected from PyRUA-Lean; source attribution is retained in [geometry.py](geometry.py) with its [Apache license](LICENSE).
