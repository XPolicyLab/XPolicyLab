---
name: elliptical_edge_regions
description: "Finds compact, unclipped elliptical edge regions in current RGB by fitting Canny contours, checking angular coverage and boundary residual, then deduplicating nearby fits."
---

# Elliptical edge regions

## Function

Finds compact, unclipped elliptical edge regions in current RGB by fitting Canny contours, checking angular coverage and boundary residual, then deduplicating nearby fits.

## Inputs

- Current uint8 RGB image of shape (480, 640, 3).

## Outputs

- Up to 64 region dictionaries containing mask, contour pixels, area, bounding box, centroid and boundary error.

## Callable interface

Implementation: [implementation.py](implementation.py).

- `run(image)`

## Constraints

- Examines at most 512 contours; accepted axes are 10-96 pixels with aspect ratio at most 3 and at least 18 of 24 occupied angular bins.
- Rejects clipped contours, excessive fit residual and out-of-range areas. Ellipse detection alone establishes no object identity, opening or metric pose.
