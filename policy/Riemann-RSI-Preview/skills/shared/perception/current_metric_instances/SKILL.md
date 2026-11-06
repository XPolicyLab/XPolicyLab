---
name: current_metric_instances
description: "Associates current head semantic regions with wrist regions using projected support-height hypotheses and a geometric-plus-colour cost, then builds conservative visual-hull instances. Fits enclosed RGB-edge or closed-mask boundaries across independent cameras with bounded attempts."
---

# Current metric instances

## Function

Associates current head semantic regions with wrist regions using projected support-height hypotheses and a geometric-plus-colour cost, then builds conservative visual-hull instances. Fits enclosed RGB-edge or closed-mask boundaries across independent cameras with bounded attempts.

## Inputs

- api: restricted public robot API providing current images, projections, and camera models
- description: one bounded string of 1-160 printable characters; support_height and support_uncertainty are meters
- regions: dict keyed by cam_head, cam_left_wrist, cam_right_wrist, each containing current semantic region lists
- diagnostics: optional list, or InstanceAdmissionDiagnostics subclass, to collect gate and cost records

## Outputs

- instances(...): list of items containing anchor snapshot, surface, and per-camera associated regions
- anchor snapshot: center_m, spans_m, height_estimate_m, uncertainty_m, cameras, coverage, appearance_rgb, chosen yaw gauge, and source current_public_rgb_visual_hull
- openings(...): deduplicated fitted boundary frames, or an empty list
- diagnostics: appended per-head admission summary and optional receipt

## Callable interface

Implementation: [implementation.py](implementation.py).

- `InstanceAdmissionDiagnostics: exception or data type`
- `appearance(image, region)`
- `snapshot(surface, description)`
- `instances(api, description, support_height, support_uncertainty, require_supported=True, diagnostics=None)`
- `fitted_openings(api, regions, extractor, upright_only, balance_independent_views=False)`
- `openings(api, regions, upright_only=False)`
- `outside(region, boundary)`

## Constraints

- module-level CAMERAS fixes the three camera keys; at most 12 head regions are considered; clipped heads are skipped and head rays nearly parallel to support are rejected
- wrist association cost is minimum projected-pixel distance plus 0.03 times RGB median distance, retaining up to two regions per wrist camera
- instance centers within 0.015 m are duplicates; snapshot marks orientation_is_chosen_geometry_gauge true and surface_is_true_object_geometry false
- opening fits use at most 12 attempts, pose deduplication within 0.008 m, and balanced closed-cycle groups use a 4-attempt budget; closed-cycle agreement requires observed_upper_body_edge true in all used views
