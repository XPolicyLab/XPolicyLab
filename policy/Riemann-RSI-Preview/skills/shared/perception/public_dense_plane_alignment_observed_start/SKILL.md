---
name: public_dense_plane_alignment_observed_start
description: "Adapter-bound helper compares the current head and own wrist RGB views under independently reconstructed camera transforms, searching heights around the observed own mounting height for a horizontal support-plane hypothesis using whole-image visible texture. If the original current texture alignment is not qualified, it retries with a color-independent spatial patch correspondence fallback."
---

# Public dense plane alignment observed start

## Function

Adapter-bound helper compares the current head and own wrist RGB views under independently reconstructed camera transforms, searching heights around the observed own mounting height for a horizontal support-plane hypothesis using whole-image visible texture. If the original current texture alignment is not qualified, it retries with a color-independent spatial patch correspondence fallback.

## Inputs

- Adapter-bound api exposing own_robot_bases_available(), own_robot_bases(), and image(camera) for cam_head, cam_left_wrist, and cam_right_wrist
- Current native public RGB images shaped (480, 640, 3) with dtype uint8
- color_independent boolean mode for the fallback path; non-bool raises ValueError

## Outputs

- Dict with status ('observed' or 'uncertain'), plane or None, camera_alignment records, qualification, reason, and official_success_claimed=False
- When qualified, plane contains normal [0, 0, 1], distance_m as the mean accepted height, height_uncertainty_m at least 0.002, and semantics horizontal_support_hypothesis_from_current_cross_view_texture_alignment
- camera_alignment records include camera, height_estimate_m, correlation, current_matching_pixels, height_uncertainty_m, peak_unique, and qualified

## Callable interface

Implementation: [implementation.py](implementation.py).

- `image_fields(image, color_independent=False)`
- `plane_projection(api, camera, support_height)`
- `ranked_correlation(source, destination, warp, robust=False)`
- `correlation(source, destination, warp, robust=False)`
- `current_alignment(api, color_independent=False)`
- `run(api)`

## Constraints

- Requires own camera reference availability and observed own mounting reference; otherwise returns uncertain with reason own_camera_reference_unavailable or own_mounts_disagree
- Uses 320x240 reduced images, a coarse height search of mounting height plus or minus 0.15 m in 31 steps, a fine plus or minus 0.012 m search in 25 steps, and rejects correlations with fewer than 1500 valid pixels
- Does not load old scene coordinates, physical object dimensions, layout metadata, or evaluator outputs; it does not change rendering or request additional observations
- A qualified alignment is not official success; photometric correlation is not a metric point-cloud residual, depth sensor, object pose, collision-free route, or contact claim
