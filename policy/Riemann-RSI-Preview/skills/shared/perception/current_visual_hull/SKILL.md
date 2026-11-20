---
name: current_visual_hull
description: "Reconstructs a conservative current RGB visual hull by intersecting complete current semantic or colour masks from two or three distinct public cameras on an observed support plane. Returns one unique surface summary only when the bounded grid, extent, support, uncertainty, and projection-coverage gates pass."
---

# Current visual hull

## Function

Reconstructs a conservative current RGB visual hull by intersecting complete current semantic or colour masks from two or three distinct public cameras on an observed support plane. Returns one unique surface summary only when the bounded grid, extent, support, uncertainty, and projection-coverage gates pass.

## Inputs

- api: exactly PublicRobotAPI
- regions: dict with exactly cam_head, cam_left_wrist, cam_right_wrist; each value is a list of at most four current regions
- support_height: finite meters with abs <=3; support_uncertainty: finite 0 to 0.006 m
- require_supported: bool; diagnostics: optional append-only list

## Outputs

- surface dict: points_xyz, rgb_samples, center_m, spans_m, height_estimate_m, uncertainty_m, cameras, coverage, head_region, source current_public_rgb_visual_hull
- None when no unique qualified hull remains; multiple distinct hulls are treated as ambiguous
- diagnostics receipt: attempts, gate reasons, grid steps, coverage, and counts with source current_public_visual_hull_gate_diagnostics
- diagnostic_inputs/diagnostic_regions: hashes, public action epoch, own state summary, and bounded region records

## Callable interface

Implementation: [implementation.py](implementation.py).

- `diagnostic_inputs(api)`
- `diagnostic_regions(regions)`
- `region_mask(region)`
- `projected(points, model)`
- `inside(points, mask, model)`
- `projection_coverage(points, mask, model, spacing)`
- `one_hull(api, observations, support_height, support_uncertainty, require_supported, diagnostics=None)`
- `run(api, regions, support_height, support_uncertainty, require_supported=True, diagnostics=None)`

## Constraints

- module-level CAMERAS fixes the three keys; observations require 2-3 distinct cameras with cam_head first and maximum pairwise camera-origin baseline at least 0.02 m
- region outlines must be finite 96x2, not clipped, and sourced from sam3_current_public_rgb_mask or current_public_colour_mask; colour masks must match the current camera RGB SHA-256
- search uses 32 then 38 voxel grids, at most 12 attempts, minimum 16 occupied points, minimum coverage 0.55, uncertainty <=0.018 m, height 0.005-0.35 m, and maximum planar span 0.004-0.5 m
- require_supported rejects lower z clearance greater than max(0.012 m, 2*uncertainty); returned surface_is_true_object_geometry is false and concavity/occluded contact remain uncertain
