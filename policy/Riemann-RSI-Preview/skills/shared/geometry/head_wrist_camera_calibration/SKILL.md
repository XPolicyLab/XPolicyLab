---
name: head_wrist_camera_calibration
description: "Defines Camera and CameraRig for current public head/wrist RGB geometry and builds a finger occlusion mask from explicit opening hypotheses. Camera.project maps 3D points to pixels; Camera.pixels back-projects pixels to a caller-declared plane z; CameraRig.current builds head and wrist cameras from fixed calibration and current EEF state."
---

# Head wrist camera calibration

## Function

Defines Camera and CameraRig for current public head/wrist RGB geometry and builds a finger occlusion mask from explicit opening hypotheses. Camera.project maps 3D points to pixels; Camera.pixels back-projects pixels to a caller-declared plane z; CameraRig.current builds head and wrist cameras from fixed calibration and current EEF state.

## Inputs

- urdf: path to shipped public X5 URDF; fingers: optional path to fingers.npz
- Camera.project(points) and Camera.pixels(uv, plane_z)
- CameraRig.current(obs) with obs.state left_ee_pose/right_ee_pose
- finger_occlusion_mask(camera, state, plane_z, rig)

## Outputs

- Camera object with center, rotation, focal_px, and project/pixels results with NaN where invalid
- current returns dict of cam_head and cam_left_wrist/cam_right_wrist Camera objects
- mask returns boolean 480x640 union and per-arm masks

## Callable interface

Implementation: [implementation.py](implementation.py).

- `Camera: __init__(self, name, transform, focal_px); project(self, points); pixels(self, uv, plane_z)`
- `CameraRig: __init__(self, urdf, fingers=None); current(self, obs)`
- `finger_occlusion_mask(camera, state, plane_z, rig)`

## Constraints

- Camera methods are bound to instance center/rotation/focal_px; wrist transforms use only fixed hand_to_camera_mount and camera_joint URDF ancestors.
- URDF origins use extrinsic xyz Euler angles; RoboDojo camera-config orientations use intrinsic XYZ angles.
- finger_occlusion_mask unions opening_hypotheses and dilates each arm mask; it is not a measured two-finger pose and URDF limits do not measure contact.
- Camera poses are not obtained from the simulator; project/pixels return NaN for behind-camera or invalid intersections.
