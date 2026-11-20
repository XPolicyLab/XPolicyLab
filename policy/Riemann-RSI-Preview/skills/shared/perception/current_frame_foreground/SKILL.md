---
name: current_frame_foreground
description: "row_median_candidates proposes unlabelled compact foreground components by thresholding each pixel against its row median in a fixed ROI; prefer_primary_detections removes appearance-fallback detections that match a nearby mature primary detection of the same category."
---

# Current frame foreground

## Function

row_median_candidates proposes unlabelled compact foreground components by thresholding each pixel against its row median in a fixed ROI; prefer_primary_detections removes appearance-fallback detections that match a nearby mature primary detection of the same category.

## Inputs

- row_median_candidates(rgb, roi, patch_size=64, threshold=26, min_area=70, max_area=7000): rgb is converted to uint8 and truncated to the first three channels; roi is boolean-like with image height and width
- prefer_primary_detections(detections, radius=20): detections are mappings with category, centroid and optional appearance_fallback

## Outputs

- row_median_candidates returns a list of dicts with centroid, area, bbox, pixels, top_uv, patch and appearance_fallback=True
- centroid is float64 [u, v] in pixels; area and bbox are pixel counts/coordinates; pixels is Nx2 float64 [column, row]; top_uv is the mean column of the top two rows and that top row; patch is a patch_size-square uint8 RGB copy
- prefer_primary_detections returns the input detections filtered so fallback entries are dropped when a non-fallback entry of the same category has centroid within radius pixels

## Callable interface

Implementation: [implementation.py](implementation.py).

- `row_median_candidates(rgb, roi, patch_size=64, threshold=26, min_area=70, max_area=7000)`
- `prefer_primary_detections(detections, radius=20)`

## Constraints

- Background is the per-row median RGB; mask uses max absolute channel difference > threshold, 3x3 elliptical opening, 5x5 elliptical closing, ROI zeroing, and 8-connected components.
- Components are rejected unless min_area <= area <= max_area, not within 2 pixels of any image border, and a centered patch_size patch fits in the image.
- Proposals are unlabelled; the task's existing classifier must reject background and require at least 0.9 confidence. Background edges and shadows can also propose components.
- No reset layout, object pose, depth, stored empty image, scorer, object metadata or simulator state is used; pose estimation, reference memory and tracking remain the task's mature implementation.
