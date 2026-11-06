---
name: rgb_components
description: "blobs labels a caller-provided binary mask with 8-connectivity, filters components below min_area, and returns observed pixel regions and centres sorted by visible area. It reports only connected-component measurements from the supplied mask."
---

# Rgb components

## Function

blobs labels a caller-provided binary mask with 8-connectivity, filters components below min_area, and returns observed pixel regions and centres sorted by visible area. It reports only connected-component measurements from the supplied mask.

## Inputs

- mask: caller-provided public binary image mask
- min_area: minimum component area in pixels, default 20

## Outputs

- list of dicts with area in pixels, uv centroid as an array, bbox as (x, y, w, h), and pix as an array of (u, v) pixel coordinates
- components sorted by descending area

## Callable interface

Implementation: [implementation.py](implementation.py).

- `blobs(mask, min_area=20)`

## Constraints

- Uses cv2.connectedComponentsWithStats with connectivity=8 and skips background label 0.
- No task zones, object coordinates, or predefined layout are introduced.

## Provenance

Connected-component function extracted from GO-3 PR214.
