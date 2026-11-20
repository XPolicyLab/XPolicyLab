---
name: observed_row_geometry
description: "Enumerates point subsets of the requested size and fits image rows whose axis agrees with a supplied tangent and whose spacing and residual meet visible-width bounds."
---

# Observed row geometry

## Function

Enumerates point subsets of the requested size and fits image rows whose axis agrees with a supplied tangent and whose spacing and residual meet visible-width bounds.

## Inputs

- Two to sixteen finite 2D centres, their positive observed diameters, a nonzero 2D tangent and required_count from two to eight.

## Outputs

- All qualifying rows, each with ordered original point indices, centroid, axis, normal and maximum residual in pixels.

## Callable interface

Implementation: [implementation.py](implementation.py).

- `run(centres, diameters, tangent, required_count)`

## Constraints

- Rejects row axes more than twenty degrees from the supplied tangent. Residual must not exceed max(1 pixel, one quarter of median width).
- Gaps must exceed half the median width and the largest gap must not exceed 1.7 times the smallest. Multiple rows remain multiple hypotheses.
