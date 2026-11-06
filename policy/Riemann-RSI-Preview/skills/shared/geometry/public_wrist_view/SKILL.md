---
name: public_wrist_view
description: "Runs an own-URDF wrist-view preflight: projects own-arm joint origins and the full declared gripper envelope into cam_head for four fixed camera profiles and selects the reachable candidate with least projected overlap, preferred-profile mismatch, joint travel, then profile index. Returns a target pose and bounded evidence, or an uncertain no-reachable-profile outcome."
---

# Public wrist view

## Function

Runs an own-URDF wrist-view preflight: projects own-arm joint origins and the full declared gripper envelope into cam_head for four fixed camera profiles and selects the reachable candidate with least projected overlap, preferred-profile mismatch, joint travel, then profile index. Returns a target pose and bounded evidence, or an uncertain no-reachable-profile outcome.

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
- `joint_origins(kinematics, base, joints)`
- `gripper_joint_envelope()`
- `gripper_segments(kinematics, base, joints)`
- `projected_overlap(api, chains, protected_pixels, plane_height, margin_m=0.07)`
- `camera_target(focus, distance, offset, yaw)`
- `validate_evidence(value)`
- `run(api, side, focus, distance, protected_pixels, plane_height, preferred_profile=None)`

## Constraints

- Requires own-robot FK capability; missing capability or invalid side raises ValueError
- Four fixed profiles only; preferred_profile outside them raises ValueError
- Projected overlap is a risk estimate, not a true visibility or contact sensor; the 70 mm margin is view-planning clearance, and the full declared gripper envelope is included without treating normalized feedback as a real finger position
- Evidence forbids scene state injection and collision bypass; selection does not replace numerical collision validation or authorize manipulation; caller must settle the real camera and qualify current RGB geometry
