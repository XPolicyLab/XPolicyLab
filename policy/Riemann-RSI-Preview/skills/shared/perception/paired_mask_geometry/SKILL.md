---
name: paired_mask_geometry
description: "Associates current head and wrist masks by closest centroid rays under the fixed X5 camera rig, estimating a centre and planar orientation. When qualified head masks are absent, a wrist viewing cone selects current head foreground components."
---

# Paired mask geometry

## Function

Associates current head and wrist masks by closest centroid rays under the fixed X5 camera rig, estimating a centre and planar orientation. When qualified head masks are absent, a wrist viewing cone selects current head foreground components.

## Inputs

- Native head/background RGB, up to eight Boolean semantic masks per view, a wrist camera model and finite support_z calibration.
- Optional explicit head-image exclusion mask. Camera helpers use public wrist EEF poses.

## Outputs

- Observed instance evidence with centre, yaw, ray separation, anisotropy, image centroids, head mask and semantic confidence; or unknown with a reason.

## Callable interface

Implementation: [implementation.py](implementation.py).

- `rotation(q)`
- `camera(name, ee_pose=None)`
- `rays(model, pixels)`
- `project(model, points)`
- `centroid(mask)`
- `pair(head_mask, wrist_mask, wrist_model, support_z)`
- `foreground_candidates(head_rgb, background_rgb, wrist_mask, wrist_model, support_z)`
- `run(head_rgb, background_rgb, head_regions, wrist_regions, wrist_model, support_z, excluded_head_mask=None)`

## Constraints

- Requires forward rays, at most 0.02 m ray separation and estimated height in [support_z-0.01, support_z+0.16].
- Rejects head masks with more than fifteen percent exclusion overlap and ambiguous separated candidates with similar ray residuals.
- Returns physical_pose_claimed=False. The support calibration and foreground/background reference remain caller-supplied assumptions.
