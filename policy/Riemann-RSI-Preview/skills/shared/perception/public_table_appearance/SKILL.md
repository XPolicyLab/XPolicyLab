---
name: public_table_appearance
description: "TableAppearance stores public-image region and robot masks, fits a robust quadratic LAB a/b background model over selected pixels, and produces a foreground residual mask. fit returns None when the selected subsampled region has fewer than 500 pixels."
---

# Public table appearance

## Function

TableAppearance stores public-image region and robot masks, fits a robust quadratic LAB a/b background model over selected pixels, and produces a foreground residual mask. fit returns None when the selected subsampled region has fewer than 500 pixels.

## Inputs

- region: array-like public-image mask, converted with np.asarray(region, bool)
- robot_mask: array-like own-robot mask; must have the same shape as region
- fit(rgb, ignore=None, k_fit=7, iters=5, trim=0.7, block=2, abs_cap=18.0): RGB image, optional ignore mask, and fit controls
- foreground(rgb, fit, k_ab=2.0, k_b=5.0, open_px=3, close_px=11): RGB image, fit tuple from fit(), and threshold/morphology controls

## Outputs

- constructor: TableAppearance instance; raises ValueError('Public image masks differ') if mask shapes differ
- fit: None or ((coef, sigma, spread, (h, w)), (ys, xs, block))
- foreground: (boolean foreground mask, sigma)
- sigma is the empirical robust residual scale; spread is the 90th percentile residual

## Callable interface

Implementation: [implementation.py](implementation.py).

- `TableAppearance: __init__(self, region, robot_mask); fit(self, rgb, ignore=None, k_fit=7, iters=5, trim=0.7, block=2, abs_cap=18.0); foreground(self, rgb, fit, k_ab=2.0, k_b=5.0, open_px=3, close_px=11)`

## Constraints

- fit median-blurs with k_fit, converts RGB to LAB, uses only region & ~robot_mask & ~ignore, subsamples by block, and requires at least 500 selected pixels
- fit performs at most iters least-squares passes on a normalized quadratic [1,u,v,u^2,uv,v^2] basis for LAB a/b and clips the trimming threshold to [3.0, abs_cap]
- foreground predicts LAB from the fit, thresholds d_ab > k_ab*sigma or d_b > k_b*sigma, then opens and closes with open_px/close_px kernels
- Does not estimate physical table geometry; notes say only three required GO-3 colour-background functions were extracted, with no pack-task module, fixed table/world coordinates, or scene handles.

## Provenance

Extracted three generic GO-3 colour-background functions; no pack-task module or fixed table/world coordinates are imported.
