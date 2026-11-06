---
name: mature_rule_sampling_observed_start
description: "Selects a rounded, evenly spaced subset of a finite six-joint dense path using speedup, retaining both endpoints when speedup > 1 and length > 2. Returns a copied numpy array of joint commands for rule_pose or rule_joint sampling; the official bridge then samples that subset at its unchanged observation interval."
---

# Mature rule sampling observed start

## Function

Selects a rounded, evenly spaced subset of a finite six-joint dense path using speedup, retaining both endpoints when speedup > 1 and length > 2. Returns a copied numpy array of joint commands for rule_pose or rule_joint sampling; the official bridge then samples that subset at its unchanged observation interval.

## Inputs

- positions: finite 2-D numpy-compatible array with shape (N, 6), 1 <= N <= 20000
- speedup: finite non-bool scalar in [0.1, 4], dimensionless

## Outputs

- dense[indices].copy() subset of six-joint positions
- unchanged dense.copy() when speedup <= 1 or N <= 2

## Callable interface

Implementation: [implementation.py](implementation.py).

- `speed_up_controls(positions, speedup)`

## Constraints

- raises ValueError for wrong shape, non-finite values, bool/non-finite/out-of-range speedup
- count=max(2, ceil(N/speedup)); indices are unique rint(linspace(0, N-1, count))
- does not accept task ID, object state, environment handle, or evaluator feedback
- multiplying bridge stride by speedup is not equivalent and must not replace this sequence
