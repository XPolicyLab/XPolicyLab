---
name: visible_reference_correspondence
description: "Returns the index of exactly one finite candidate point that lies within at least one explicit public reference radius in either native pixels or metres. Zero matches and multiple local matches remain uncertain and return None."
---

# Visible reference correspondence

## Function

Returns the index of exactly one finite candidate point that lies within at least one explicit public reference radius in either native pixels or metres. Zero matches and multiple local matches remain uncertain and return None.

## Inputs

- candidate_points: up to 64 finite Nx2 or Nx3 points
- public_references: 1..8 finite reference points in the same dimensionality
- maximum_distances: positive per-reference radii
- units: 'native_pixels' or 'metres'

## Outputs

- int index of the unique matching candidate, or None

## Callable interface

Implementation: [implementation.py](implementation.py).

- `run(candidate_points, public_references, maximum_distances, units)`

## Constraints

- native_pixels uses 2-D points bounded to x<640 and y<480, radii <=32; metres uses 3-D points with |coordinate|<=3 and radii <=.2
- References must be finite and bounded; invalid dimensions, radii, shapes or nonfinite references raise ValueError
- Nonfinite or out-of-image candidates cannot match
- Returned index is visual association evidence only; the caller keeps qualification thresholds and decides task operation
