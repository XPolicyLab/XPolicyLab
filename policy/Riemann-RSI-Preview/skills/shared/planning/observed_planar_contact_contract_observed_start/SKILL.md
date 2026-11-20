---
name: observed_planar_contact_contract_observed_start
description: "Validates exact public planar T-anchor, route, and own contact progress schemas using checked_pose from the observed-start geometry module and returns deep copies. active_destination selects the route staging pose when present, otherwise the observed destination pose."
---

# Observed planar contact contract observed start

## Function

Validates exact public planar T-anchor, route, and own contact progress schemas using checked_pose from the observed-start geometry module and returns deep copies. active_destination selects the route staging pose when present, otherwise the observed destination pose.

## Inputs

- anchor: exact dict with version=1, source, colour, initial_pose, destination_pose, support_plane_m, top_plane_m, maximum_ray_separation_m, initial_head_area
- route: exact dict with side, turn, contact_wrist, high_wrist, low_joints, high_joints, joint_margin_rad, and optional staging_destination_pose
- progress: exact dict with version, stage, side, updates, route, shape_in_wrist, predicted_pose, last_observed_wrist, pose_source, independent_visual_confirmed, recontacts, issued_actions
- observed and current: anchor and progress values passed to active_destination

## Outputs

- anchor(value): deep-copied validated anchor
- route(value): deep-copied validated route
- progress(value): deep-copied validated progress
- active_destination(observed,current): checked_pose of selected destination or staging pose; initial_progress() returns the declared zero progress dict

## Callable interface

Implementation: [implementation.py](implementation.py).

- `anchor(value)`
- `route(value)`
- `active_destination(observed, current)`
- `initial_progress()`
- `progress(value)`

## Constraints

- Uses checked_pose from skills.shared.geometry.planar_rigid_contact_observed_start.implementation
- Anchor source and colour are whitelisted; initial_head_area is int in [40,46080]; top_plane_m - support_plane_m in [0.003,0.04] m and maximum_ray_separation_m in [0,0.004] m
- Route high_wrist is 0.10 m above contact_wrist with identical XY/orientation; six finite joints within pi; joint_margin_rad in [0.08,pi]; optional staging Z is low Z minus 0.154 m
- Progress stage 0..6, updates 0..100, recontacts 0..1, issued_actions 0..640; stages >=2 require route/side, stages >=4 require poses; pose_source is unavailable/current_public_planar_observation/own_wrist_contact_hypothesis_not_object_measurement; numerical tracker is not measured contact and native success/score is evaluator-only
