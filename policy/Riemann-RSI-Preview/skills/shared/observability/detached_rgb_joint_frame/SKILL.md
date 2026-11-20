---
name: detached_rgb_joint_frame
description: "Defines a frozen Observation that validates and copies six own-robot state keys plus three current RGB images, and validates rule actions against a bound observation. read_finger_joints always rejects precise finger access."
---

# Detached rgb joint frame

## Function

Defines a frozen Observation that validates and copies six own-robot state keys plus three current RGB images, and validates rule actions against a bound observation. read_finger_joints always rejects precise finger access.

## Inputs

- state mapping with left/right _arm_joint_state (6 floats), _ee_pose (7 floats), and _ee_joint_state (1 float)
- vision mapping with cam_head, cam_left_wrist, cam_right_wrist each 480x640x3 uint8
- count integer official control index in [0, Observation.HORIZON=10000]
- command mapping for validate_bound_action containing both arms' _arm_joint_state and _ee_joint_state keys

## Outputs

- Observation with immutable MappingProxyType state/images, physical_step=count, and per-camera SHA-256 evidence
- validate_bound_action returns a dict of finite float lists with normalized jaw commands in [0,1]
- read_finger_joints raises PermissionError

## Callable interface

Implementation: [implementation.py](implementation.py).

- `Observation: create(cls, state, vision, count, summary='')`
- `validate_bound_action(command, observation)`
- `read_finger_joints(*args, **kwargs)`

## Constraints

- Copies state and image arrays and marks them read-only; image evidence is a SHA-256 hex digest per camera
- Raises ValueError for non-int/out-of-range count, wrong state shapes or nonfinite values, and images not 480x640x3 uint8
- Jaw values are normalized commands, never physical finger widths or contact
- No layout, reward, object pose, depth, or finger measurement enters the frame; precise finger positions/contact are not public observations
