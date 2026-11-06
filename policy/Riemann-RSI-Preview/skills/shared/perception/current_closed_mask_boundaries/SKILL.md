---
name: current_closed_mask_boundaries
description: "Accepts a current native 480x640 uint8 RGB image and an unclipped 96-point public region outline, then fits up to four convex four-line closed cycles from Hough segments contained in the filled outline. Candidates are sorted by area times mean per-side Canny-edge support and include an upper-edge agreement flag."
---

# Current closed mask boundaries

## Function

Accepts a current native 480x640 uint8 RGB image and an unclipped 96-point public region outline, then fits up to four convex four-line closed cycles from Hough segments contained in the filled outline. Candidates are sorted by area times mean per-side Canny-edge support and include an upper-edge agreement flag.

## Inputs

- image: current native RGB array with shape (480, 640, 3) and dtype uint8
- region: mapping with outline as finite float-compatible (96, 2) points and clipped exactly False; every outline coordinate must be >1, x < 638 and y < 478

## Outputs

- list of 0 to 4 candidate dicts sorted by edge_support_score descending
- each candidate has outline, centroid/projective_center in pixel coordinates, bbox, area_pixels, edge_support_score, edge_support_per_side, observed_upper_body_edge, source='current_public_closed_mask_line_boundary', clipped=False, boundary_is_current_image_line_fit=True and physical_asset_geometry_used=False
- edge_support_per_side is a list of per-side fractions; observed_upper_body_edge is true when any side has at least 0.85 of samples within 2.5 pixels of the observed upper semantic edge

## Callable interface

Implementation: [implementation.py](implementation.py).

- `intersection(first, second)`
- `run(image, region)`

## Constraints

- Returns [] for wrong image shape/dtype, outline shape/nonfinite values, clipped not False, or outline too close to image bounds.
- A cycle requires four distinct line segments, convex vertices, area between max(40, 8% of parent area) and 92% of parent area, at least 99% containment, min side support >=0.85, mean edge distance <=0.8 pixels, and a diagonal intersection center inside the polygon.
- The boundary and diagonal-intersection center are current image-line fits, not exact object geometry; candidates do not prove an empty receiver or successful placement.
- Accepted upper cycles require actual observed upper semantic edge agreement; a floor or front-face cycle without it is rejected. The general contour path runs first and these candidates are used only when it yields no qualified current planar frame.
