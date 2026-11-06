---
name: raised_public_wrist_view
description: "run builds a bounded proposed own wrist view by lifting from an own reference pose, retreating along the declared wrist-camera optical axis, and passing a centroid/focus target to target_pose. It returns only a proposed pose or None when the translation bound is exceeded."
---

# Raised public wrist view

## Function

run builds a bounded proposed own wrist view by lifting from an own reference pose, retreating along the declared wrist-camera optical axis, and passing a centroid/focus target to target_pose. It returns only a proposed pose or None when the translation bound is exceeded.

## Inputs

- api: adapter exposing targets() and home_targets(); also passed to camera_model and target_pose
- side: 'left' or 'right'; selects '<side>_ee_pose' and 'cam_<side>_wrist'
- centroid and focus_height: passed unchanged to target_pose after the proposed pose is formed
- use_episode_home: keyword-only bool; when true, reference is api.home_targets()[side + '_ee_pose'], otherwise the current initial pose

## Outputs

- proposed own wrist pose returned by target_pose, or None if ||proposed[:3] - initial[:3]|| > 0.15 m
- raises ExecutionError for an invalid side or a non-bool use_episode_home
- no object geometry, action, depth, contact, finger-joint position, or scorer data is returned

## Callable interface

Implementation: [implementation.py](implementation.py).

- `run(api, side, centroid, focus_height, *, use_episode_home=False)`

## Constraints

- Lift is [0, 0, 0.06] m plus own_rotation @ camera[:3, 2] * 0.08 m, with own_rotation = rotation_matrix(reference[3:]) @ rotation_matrix(initial[3:]).T
- camera_model(api, 'cam_' + side + '_wrist') supplies the camera matrix; the unused focal return is ignored
- The optional episode-home reference uses only the API's retained own home pose and is intended to avoid repeated retreat accumulation
- Caller must independently qualify the current focus reference, current open-command state, and own tool clearance; a clear proposed view does not guarantee a visible goal or later geometry qualification.
