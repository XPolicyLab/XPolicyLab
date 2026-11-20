---
name: descending_number_assignment
description: "Validates equally sized current digit and target families, then pairs digits in descending numeric order with targets sorted by ascending measured x-coordinate. It returns explicit source and target surface coordinates plus target indices."
---

# Descending number assignment

## Function

Validates equally sized current digit and target families, then pairs digits in descending numeric order with targets sorted by ascending measured x-coordinate. It returns explicit source and target surface coordinates plus target indices.

## Inputs

- digits: list/tuple of 1–10 dicts with exactly digit (int 0–9, unique) and surface_xyz (finite shape (3,))
- targets: equally sized list/tuple convertible to a finite float array of shape (n, 3)

## Outputs

- list of dicts with digit, source_surface_xyz, target_surface_xyz (lists of three numbers), and target_index
- pairs are ordered by descending digit and ascending target x-coordinate

## Callable interface

Implementation: [implementation.py](implementation.py).

- `run(digits, targets)`

## Constraints

- Raises ValueError for invalid types/shapes, non-finite values, duplicate digits, out-of-range digits, mismatched family lengths, or non-distinct target x positions separated by 1e-8 or less.
- Does not supply missing target counts, pad spacing, nominal positions, digit identities, or layouts.
- Pure numeric planning only; not recognition, manipulation, progress, or official-success validation.
- Does not mutate inputs; returns new lists.
