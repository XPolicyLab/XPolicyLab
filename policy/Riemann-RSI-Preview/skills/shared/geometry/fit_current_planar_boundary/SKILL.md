---
name: fit_current_planar_boundary
description: "Fits a current planar boundary by intersecting public camera rays with candidate planes and requiring reprojection consistency in at least one other actual camera. Returns a chosen plane-frame pose with measured uncertainty or None when inputs, fit, ambiguity, baseline, or uncertainty checks fail."
---

# Fit current planar boundary

## Function

Fits a current planar boundary by intersecting public camera rays with candidate planes and requiring reprojection consistency in at least one other actual camera. Returns a chosen plane-frame pose with measured uncertainty or None when inputs, fit, ambiguity, baseline, or uncertainty checks fail.

## Inputs

- api: public camera/projection API used by triangulate, pixel_rays, and project_points
- observations: list of 2-3 records, each with camera in cam_head/cam_left_wrist/cam_right_wrist, 96x2 finite outline pixels, and optional projective_center
- upright_only: bool restricting normal seeds to [0,0,1] with a tight angular bound
- reference_camera: first record's public camera, default cam_head

## Outputs

- dict with pose [center xyz, wxyz quaternion], normal_xyz, outline_xyz, spans_m, position_uncertainty_m, axis_uncertainty_radians, pixel_rms, cameras, and source/flags
- None on invalid input or rejected/ambiguous/weak/uncertain fit

## Callable interface

Implementation: [implementation.py](implementation.py).

- `tangent_basis(normal)`
- `intersections(origin, rays, normal, point)`
- `run(api, observations, upright_only=False, *, reference_camera='cam_head')`

## Constraints

- Requires 2-3 distinct actual cameras; first record camera must equal reference_camera; outlines must be finite 96x2 pixels inside 640x480 bounds with coordinates >1.
- Returns None rather than raising for invalid observations, failed triangulation, nonintersection, ambiguity, or uncertainty above thresholds.
- Output orientation_is_chosen_geometry_gauge is true and is_true_object_pose/is_contact_measurement/physical_asset_geometry_used are false.
- Uses no template, assumed object dimensions, environment state, layout, or scorer; units are meters and radians where named.
