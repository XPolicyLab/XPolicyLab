---
name: current_vessel_regions
description: "Proposes current whole-image vessel components by HSV appearance thresholds and returns convex-hull outlines and median RGB appearance. The boundary helper resamples a closed contour to 96 equally spaced points."
---

# Current vessel regions

## Function

Proposes current whole-image vessel components by HSV appearance thresholds and returns convex-hull outlines and median RGB appearance. The boundary helper resamples a closed contour to 96 equally spaced points.

## Inputs

- image: current public RGB array shaped 480x640x3 with dtype uint8
- appearance: one of yellow_cup, light_vase, or dark_vase
- contour for boundary(): at least three vertices; total closed length must exceed 1.0

## Outputs

- run(image,appearance): list of at most eight component dicts sorted by descending area
- component dict: outline, bbox, area, centroid, appearance_rgb, source whole_current_rgb_component, and appearance
- boundary(contour,count=96): 96x2 float outline or None

## Callable interface

Implementation: [implementation.py](implementation.py).

- `boundary(contour, count=96)`
- `run(image, appearance)`

## Constraints

- invalid image shape/dtype or undeclared appearance raises ValueError
- components require area 80 to 18% of image, minimum bbox side 6, and bbox aspect ratio at most 5
- appearance ranges are visual priors, not identity, dimensions, or position; no old ROI, crop, layout, asset model, or metric estimate is restored
- an independently qualified multi-view model is required before motion
