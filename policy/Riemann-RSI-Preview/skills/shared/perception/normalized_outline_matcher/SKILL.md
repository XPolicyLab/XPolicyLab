---
name: normalized_outline_matcher
description: "Ranks every label in an explicit outline dictionary against a normalized current silhouette, searching rotations and small raster translations. Confirms a label only when both similarity and runner-up margin pass."
---

# Normalized outline matcher

## Function

Ranks every label in an explicit outline dictionary against a normalized current silhouette, searching rotations and small raster translations. Confirms a label only when both similarity and runner-up margin pass.

## Inputs

- Finite bounded two-dimensional observed mask; 2-64 unique integer/string labels with corresponding finite outline arrays.
- One to 121 unique finite angles in [-180, 180] degrees; default -90 through 90 in three-degree increments.

## Outputs

- Confirmed or uncertain status, optional label/yaw, similarity, margin and the complete per-label ranking.
- Per-candidate template-to-observed raster transform; normalization helpers also return a 128-by-128 mask and its transform.

## Callable interface

Implementation: [implementation.py](implementation.py).

- `normalize_with_transform(mask)`
- `normalize(mask)`
- `dictionary_bank(labels, shape, dtype, data, angles)`
- `run(observed, labels, outlines, angles=DEFAULT_ANGLES)`

## Constraints

- Uses 24-pixel raster padding, similarity threshold 0.885 and margin threshold 0.012; too little foreground remains uncertain.
- Scores are outline similarity rather than probabilities, and transforms are image-raster correspondences rather than metric object poses.
- Caches at most two template-bank argument sets; cached masks and transforms are read-only.
