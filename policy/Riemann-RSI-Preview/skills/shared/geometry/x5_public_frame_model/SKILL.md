---
name: x5_public_frame_model
description: "Provides fixed X5 rotation, quaternion, matrix, and pose conversions, head-camera projection and backprojection, URDF parsing with six-joint FK, robot-frame recovery, and bounded IK. The geometry constants and joint handling are the public X5 model, with only public URDF and public own-arm state accepted."
---

# X5 public frame model

## Function

Provides fixed X5 rotation, quaternion, matrix, and pose conversions, head-camera projection and backprojection, URDF parsing with six-joint FK, robot-frame recovery, and bounded IK. The geometry constants and joint handling are the public X5 model, with only public URDF and public own-arm state accepted.

## Inputs

- rotation(q): finite WXYZ quaternion with norm > 1e-8; matrix(pose): finite XYZ-plus-WXYZ seven-vector
- project(points): 3D points in the fixed camera frame; backproject(uv, plane_point, plane_normal=(0, 0, 1)): length-2 pixel coordinates and a plane
- X5Kinematics(urdf): path to a URDF containing joint1 through joint6 with origin, axis, and limit elements; fk(joints) takes six joint values
- recover(state): state with left/right_ee_pose and left/right_arm_joint_state; RobotFrame.ik(arm, target_pose, current, *, axis_only=False, max_axis_deg=8) uses arm in ARMS

## Outputs

- rotation returns a 3x3 matrix; quaternion returns WXYZ; matrix returns 4x4; pose returns XYZ-plus-WXYZ
- project returns pixel coordinates with 320/240 principal point; backproject returns 3D points on the supplied plane
- X5Kinematics.fk returns a 4x4 transform or, with all_links=True, a dict containing base_link and link1 through link6
- recover returns a RobotFrame; RobotFrame.ik returns a record with joints, target_pose, position_error_m, rotation_error_rad, axis_error_deg, and possibly valid, or None

## Callable interface

Implementation: [implementation.py](implementation.py).

- `rotation(q)`
- `quaternion(r)`
- `matrix(pose)`
- `pose(t)`
- `project(points)`
- `backproject(uv, plane_point, plane_normal=(0, 0, 1))`
- `X5Kinematics: __init__(self, urdf); fk(self, joints, *, all_links=False); recover(self, state)`
- `RobotFrame: __init__(self, kinematics, bases, mounting_z); point_on_plane(self, relative_z=0); pixels(self, uv, relative_z=0); root_pose(self, arm); ik(self, arm, target_pose, current, *, axis_only=False, max_axis_deg=8)`

## Constraints

- Uses fixed constants FOCAL_PX = 640.0 * 10.0 / 22.212, CAMERA_CENTER [0, -0.41, 1.308], and a 30-degree x-axis camera rotation; projection and backprojection raise ValueError for points behind the camera, parallel rays, or planes behind rays.
- URDF joint limits are clipped to [-3.14159, 3.14159]; recover raises ValueError if arm mounting heights differ by more than 0.005 or a mounting normal is not within 0.999 of +z.
- IK uses scipy.optimize.least_squares with multiple seeds, bounds, and max_nfev=110; accepts position error below 0.0025 m and either axis error below max_axis_deg+0.1 or full rotation error below 0.03 rad, preferring nonnegative joint 3 when immediately valid.
- Fixed graph dispatch only; no per-action model selection, learned fallback, GT, hidden retry, or environment capability lookup; CPU parity is not native performance verification.
