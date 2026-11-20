---
name: surface_grasp_sections
description: "Validates a current public RGB-only surface point/color payload and ranks candidate cross-section bands along the principal 2D axis using the retained grasp-selection math."
---

# Surface grasp sections

## Function

Validates a current public RGB-only surface point/color payload and ranks candidate cross-section bands along the principal 2D axis using the retained grasp-selection math.

## Inputs

- payload dict with exactly points_xyz, rgb_samples, support_plane_m, current_height_estimate_m, cameras, current_public_RGB_only, and task_asset_points_used
- points_xyz: finite Nx3 float array with 16<=N<=20000 and max absolute coordinate <=3 m; rgb_samples: matching Nx3 uint8 array
- support_plane_m and current_height_estimate_m: finite scalars with 0<height<0.5 m
- cameras: list with at least two unique names from cam_head, cam_left_wrist, cam_right_wrist

## Outputs

- dict containing sections sorted by quality; each section is (quality, fraction, band, contact_xy, width, minimum_z, maximum_z, level)
- robot_actions_issued set to 0
- task_asset_points_used set to False
- full_pickup_completed set to False

## Callable interface

Implementation: [implementation.py](implementation.py).

- `candidate_sections(points, colors, plane, height)`
- `run(payload)`

## Constraints

- Requires current_public_RGB_only=True and task_asset_points_used=False; rejects mesh catalog, layout, or privileged point-cloud use
- Candidate widths are section widths in meters and must satisfy 0.004<width<=0.082; this is not a measured finger width or contact assertion
- Height bands beyond the base section are added when height>0.065 m, at levels 0.55, 0.78, and 0.32
- The primitive issues no actions; ranking is heuristic and does not indicate official task success
