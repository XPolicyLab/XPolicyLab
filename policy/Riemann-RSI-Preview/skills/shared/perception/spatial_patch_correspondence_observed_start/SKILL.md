---
name: spatial_patch_correspondence_observed_start
description: "Runs the same bounded patch-correspondence measurement but first keeps a qualified original grid; if none qualifies, it evaluates three half-grid shifted fallbacks and returns the best qualified fallback by selection_score."
---

# Spatial patch correspondence observed start

## Function

Runs the same bounded patch-correspondence measurement but first keeps a qualified original grid; if none qualifies, it evaluates three half-grid shifted fallbacks and returns the best qualified fallback by selection_score.

## Inputs

- source_high, destination_high: finite current high-frequency grayscale arrays with shape (240,320)
- valid: Boolean overlap mask with the same shape

## Outputs

- measure: qualified metrics dict or None
- run: (correlation, current_matching_pixels) or None

## Callable interface

Implementation: [implementation.py](implementation.py).

- `measure(source_high, destination_high, valid)`
- `run(source_high, destination_high, valid)`

## Constraints

- Raises ValueError unless both fields are (240,320), valid is same-shape bool, and both fields are finite.
- Uses _measure_grid with the same eligibility, inlier, coverage, and spatial-distribution thresholds as spatial_patch_correspondence.
- Fallbacks shift by (0,8), (8,0), and (8,8) only after the original grid returns None; ties are resolved by Python max over selection_score.
- Result remains a correspondence metric, not pose/contact/success; no rank interface is exposed.
