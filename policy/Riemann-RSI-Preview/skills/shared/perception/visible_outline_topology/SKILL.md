---
name: visible_outline_topology
description: "Compares enclosed background holes in the current image with the top three independently ranked outline templates after applying the supplied raster-to-image and fitted transforms. It reports hole areas, per-candidate hole-count agreement, and whether topology resolves a close candidate margin."
---

# Visible outline topology

## Function

Compares enclosed background holes in the current image with the top three independently ranked outline templates after applying the supplied raster-to-image and fitted transforms. It reports hole areas, per-candidate hole-count agreement, and whether topology resolves a close candidate margin.

## Inputs

- observed: current 2-D finite image/mask with 1..307200 elements
- labels: unique outline labels, 2..64 entries
- outlines: template array matching labels
- ranking: complete independent ranking records with label, outline_to_observed and similarity; optional raster_to_image 3x3 and clipped bool

## Outputs

- dict with observed_hole_areas_pixels and candidates containing label, hole_areas_pixels and hole_count_matches
- resolves_close_candidates bool, physical_dimensions_used=False and interpretation string

## Callable interface

Implementation: [implementation.py](implementation.py).

- `hole_areas(mask)`
- `run(observed, labels, outlines, ranking, raster_to_image=None, clipped=False)`

## Constraints

- Raises ValueError unless the current image, unique labels, complete ranking, finite transforms and condition number <=1e9 are valid
- Only top three ranking records are projected; each needs a finite 3x3 outline_to_observed and finite similarity
- Resolution requires not clipped, top similarity >=.90, at least 140 foreground pixels, winner hole-count match, runner-up mismatch and minimum extra hole area >=5 pixels
- Areas are current-image pixels; the scale-free image transform is not a physical pose, and hole agreement is not contact or hidden-occlusion proof
