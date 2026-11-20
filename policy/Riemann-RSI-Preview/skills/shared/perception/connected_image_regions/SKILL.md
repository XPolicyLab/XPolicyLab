---
name: connected_image_regions
description: "run(mask, minimum_area=20, maximum_regions=64) returns bounded eight-connected foreground components from an actual image mask. It filters by area, sorts by descending area then label, and limits the returned regions."
---

# Connected image regions

## Function

run(mask, minimum_area=20, maximum_regions=64) returns bounded eight-connected foreground components from an actual image mask. It filters by area, sorts by descending area then label, and limits the returned regions.

## Inputs

- mask: 2D finite array of size 1 to 307200 with dtype kind b, u, i, or f
- minimum_area: integer >= 1
- maximum_regions: integer from 1 to 128

## Outputs

- list of dicts with mask, pixels, bbox, area, and centroid

## Callable interface

Implementation: [implementation.py](implementation.py).

- `run(mask, minimum_area=20, maximum_regions=64)`

## Constraints

- Raises ValueError for invalid mask shape, size, dtype, non-finite values, or invalid minimum_area/maximum_regions.
- Binary foreground is value > 0; labels are eight-connected.
- No opening, closing, invented foreground, scene crop, or physical size reconstruction occurs.
- Results are image measurements, not object identity or poses.
