---
name: public_silhouette_geometry
description: "Provides calibrated public RGB projection, horizontal rim fitting and upright bottle-profile reconstruction with inferred metric dimensions and uncertainty. GeometryObservationError is raised for invalid observations, single-view ambiguity, insufficient independent support, ambiguous metrics or incomplete profiles."
---

# Public silhouette geometry

## Function

Provides calibrated public RGB projection, horizontal rim fitting and upright bottle-profile reconstruction with inferred metric dimensions and uncertainty. GeometryObservationError is raised for invalid observations, single-view ambiguity, insufficient independent support, ambiguous metrics or incomplete profiles.

## Inputs

- CalibratedRGBView(image 480x640x3 uint8 RGB, pose 4x4 finite, focal, exclusion optional 480x640 bool)
- fit_horizontal_rim(views, xy, radius, table_z)
- reconstruct_bottle(views, masks, colour, table_z, plane)
- masks keyed by colour; plane(sample, z) maps image samples to table-plane points

## Outputs

- fit_horizontal_rim returns dict with xy, rim_height, radius, pixel_rms, metric_sigma, views and dimensions_from_RGB
- reconstruct_bottle returns dict with xy, height, body_radius, neck_radius, neck_start, grasp_height, body_axis_spread, body_slice_centres, profile, height_interval and dimensions_from_RGB
- GeometryObservationError on invalid images/poses, single-view ambiguity, insufficient visible support, ambiguous metrics or incomplete profile

## Callable interface

Implementation: [implementation.py](implementation.py).

- `GeometryObservationError: exception or data type`
- `CalibratedRGBView: __init__(self, image, pose, focal, exclusion=None); project(self, points)`
- `bilinear(image, uv)`
- `ring_points(x, y, z, r, count=72)`
- `fit_horizontal_rim(views, xy, radius, table_z)`
- `membership(view, mask, points)`
- `reconstruct_bottle(views, masks, colour, table_z, plane)`

## Constraints

- Rim fit requires at least two selected views, >=58 visible ring samples and >=24 supported samples per selected view, baseline >=.025 m, rank >=4, rms <=2.7, radius sigma <=.012 and xy sigma <=.008
- Bottle reconstruction requires >=100 head silhouette pixels, >=3 body-slice centers with spread <=.012 m, and >=15 positive profile samples
- Search limits are 5 body slices, 197 profile levels from .006 to .40 m in .002 m steps, 11 radius bisection iterations, and 5 rim seeds with max_nfev=130
- Clipped or out-of-frame pixels are unknown, dimensions are inferred from RGB, and no depth, metadata, layout, reward or simulator handle is used
