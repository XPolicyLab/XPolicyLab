---
name: read_number_outline
description: "run reads one native public RGB frame, proposes bounded material/connected regions, rectifies each candidate on a parallel virtual plane for shape matching only, identifies a numeral from the scale-free dictionary, and filters nested background matches. It returns confirmed current numeral outline records plus explicit uncertain candidates, never metric poses or physical dimensions."
---

# Read number outline

## Function

run reads one native public RGB frame, proposes bounded material/connected regions, rectifies each candidate on a parallel virtual plane for shape matching only, identifies a numeral from the scale-free dictionary, and filters nested background matches. It returns confirmed current numeral outline records plus explicit uncertain candidates, never metric poses or physical dimensions.

## Inputs

- api: adapter exposing image(camera) and used by pixel_rays and points_on_plane
- camera: public camera name; implementation expects current native RGB shape exactly (480, 640, 3)
- extended: bool passed to material_outline_proposals to add material variants
- placed: bool selecting outline angle ranges: -90..90 step 3 when true, else -60..60 step 3

## Outputs

- dict with kind 'current_numeral_outlines', camera, status ('observed' only when confirmed digits are nonempty and labels are unique, otherwise 'uncertain'), digits, uncertain_candidates, dictionary_labels, and false conditioning/metric flags
- each confirmed digit includes digit, camera, anchor_image, anchor_image_normalized, yaw_radians, similarity, margin, bbox, area, and anchor/rectification semantics
- uncertain candidates carry reasons such as current_view_cannot_rectify_horizontal_outline, independent_outline_match_ambiguous, visual_anchor_outside_current_image, or enclosed_background_not_independent_foreground
- raises ValueError for a non-native frame or invalid dictionary metadata/hash/shape/field checks

## Callable interface

Implementation: [implementation.py](implementation.py).

- `dictionary()`
- `identify(mask, placed=False, *, observed_mask=None, raster_to_image=None, clipped=False)`
- `region_overlap(first, second)`
- `candidate_regions(image, extended=False, minimum_area=20)`
- `foreground_matches(recognized)`
- `rectify_outline(api, camera, region)`
- `run(api, camera, extended=False, placed=False)`

## Constraints

- Candidate proposals are limited to 24, use minimum area 45 for cam_head and 70 otherwise, reject regions over 4% of the image, tiny boxes, and border-touching boxes, and share one overlap-group budget
- Rectification uses pixel_rays and points_on_plane at origin[2] - 1 m, rejects non-backward rays, nonfinite spans/transforms, and transform condition number > 1e9; the plane is scale-free and not a measured surface
- dictionary() is lru_cached with maxsize 1, verifies dictionary.json and outline-dictionary.npz sha256, labels 0..9, 320x320 outlines, finite normalized anchors, and makes returned arrays read-only
- Similarity is not a calibrated probability; visible_outline_topology can resolve close candidates, and enclosed-background suppression requires an enclosing recognized match with at least the same similarity and margin and at least 98% overlap with its enclosed background
