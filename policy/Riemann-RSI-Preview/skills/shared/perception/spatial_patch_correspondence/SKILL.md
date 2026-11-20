---
name: spatial_patch_correspondence
description: "Computes a fixed 15x20 grid of 16x16 photometrically normalized patch correlations over two current 240x320 high-frequency grayscale fields and their valid overlap, then returns coverage-weighted metrics and qualification. rank returns metrics even when unqualified; measure and run return only qualified results."
---

# Spatial patch correspondence

## Function

Computes a fixed 15x20 grid of 16x16 photometrically normalized patch correlations over two current 240x320 high-frequency grayscale fields and their valid overlap, then returns coverage-weighted metrics and qualification. rank returns metrics even when unqualified; measure and run return only qualified results.

## Inputs

- source_high, destination_high: finite current high-frequency grayscale arrays with shape (240,320)
- valid: Boolean overlap mask with the same shape

## Outputs

- patch_metrics: (metrics dict, qualified bool) or None when no inlier pixels
- metrics: correlation, current_matching_pixels, textured_overlap_pixels, selection_score
- rank: metrics or None; measure: qualified metrics or None; run: (correlation, current_matching_pixels) or None

## Callable interface

Implementation: [implementation.py](implementation.py).

- `patch_metrics(source_high, destination_high, valid)`
- `rank(source_high, destination_high, valid)`
- `measure(source_high, destination_high, valid)`
- `run(source_high, destination_high, valid)`

## Constraints

- Raises ValueError unless both fields are (240,320), valid is same-shape bool, and both fields are finite.
- Eligible patches require >=128 valid pixels and RMS energy >=0.8 in both fields; inliers additionally require correlation >=0.60.
- Qualification requires >=6000 matching pixels, row/column extent >=4, and >=4 coarse spatial bins; correlation is scaled by min(1, pixels/12000) and selection_score weights agreement by matching fraction.
- rank is a search hint only; measure/run keep final gates; caller must fit a unique height and retain independent-camera agreement and uncertainty gates; not pose/contact/success.
