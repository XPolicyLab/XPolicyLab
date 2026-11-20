---
name: current_enclosed_rgb_edges
description: "Extracts closed current contours inside an unclipped public RGB region using an Otsu threshold on grayscale samples and a Canny edge mask, returning up to eight bounded contours sorted by pixel area. It does not label compartments or assert emptiness or insertion."
---

# Current enclosed rgb edges

## Function

Extracts closed current contours inside an unclipped public RGB region using an Otsu threshold on grayscale samples and a Canny edge mask, returning up to eight bounded contours sorted by pixel area. It does not label compartments or assert emptiness or insertion.

## Inputs

- image: current native RGB array with shape (480, 640, 3) and dtype uint8
- region: mapping with outline as finite float-compatible (96, 2) points and clipped exactly False; every outline coordinate must be >1, x < 638 and y < 478

## Outputs

- list of 0 to 8 candidate dicts sorted by area_pixels descending
- each candidate has outline, centroid in pixel coordinates, bbox, area_pixels, source='current_public_enclosed_rgb_edges' and clipped=False

## Callable interface

Implementation: [implementation.py](implementation.py).

- `run(image, region)`

## Constraints

- Returns [] if image/region validation fails, fewer than 80 contained grayscale samples exist, or sample intensity range is <18.
- Contours must have at least 12 points, area 12 to 0.85 of parent area, min bounding side >=3, coordinates inside bounds, and no more than 1% of contour area outside the parent mask.
- Duplicates within 3 pixels centroid and 15% area are suppressed; outputs are closed contours only, not occupancy or insertion claims.
- Metric pose and occupancy require separate current-view qualification; clipped, flat-colour and unresolved regions remain unqualified.
