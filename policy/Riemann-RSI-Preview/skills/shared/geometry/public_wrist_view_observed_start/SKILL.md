---
name: public_wrist_view_observed_start
description: "Observed-start variant of the public wrist-view preflight using observed-start URDF kinematics and matrix pose conversions; reuses camera_target, joint_origins, and validate_evidence from the base skill. Selects among the same four profiles by projected overlap, preference, joint travel, and index."
---

# Public wrist view observed start

## Function

Observed-start variant of the public wrist-view preflight using observed-start URDF kinematics and matrix pose conversions; reuses camera_target, joint_origins, and validate_evidence from the base skill. Selects among the same four profiles by projected overlap, preference, joint travel, and index.

## Inputs

- api providing own_robot_bases_available(), own_robot_bases(), own_joint_states(), targets(), and cam_head/cam_<side>_wrist calibration
- side in {'left','right'}, focus as 3 finite values with max abs <=3, distance in [0.2,0.45] m, plane_height finite
- protected_pixels as 1-48 finite pixel anchors in 640x480; optional preferred_profile among outward_oblique, vertical, outward_quarter_turn, front_oblique
- margin_m default 0.07 with allowed range [0.01,0.08] in projected_overlap

## Outputs

- run returns {'pose': 4x4 camera target or None, 'evidence': validated evidence dict}
- evidence status is 'view_candidate_selected' with profile and projected_overlap_count, or 'uncertain' with reason 'no_declared_camera_profile_is_own_IK_reachable'
- projected_overlap returns an integer count of protected pixels estimated overlapped by projected own-arm/gripper segments
- camera_target returns a 4x4 wrist-camera target transform

## Callable interface

Implementation: [implementation.py](implementation.py).

- `own_kinematics()`
- `gripper_joint_envelope()`
- `gripper_segments(kinematics, base, joints)`
- `projected_overlap(api, chains, protected_pixels, plane_height, margin_m=0.07)`
- `run(api, side, focus, distance, protected_pixels, plane_height, preferred_profile=None)`

## Constraints

- Uses observed-start OwnURDFKinematics, matrix_pose, and pose_matrix while sharing the base camera_target, joint_origins, and validate_evidence
- Requires own-robot FK capability; missing capability or invalid side raises ValueError
- Four fixed profiles only; preferred_profile outside them raises ValueError; projected overlap is a risk estimate, not a visibility or contact sensor
- Evidence forbids scene state injection and collision bypass; selection does not replace numerical collision validation or authorize manipulation
