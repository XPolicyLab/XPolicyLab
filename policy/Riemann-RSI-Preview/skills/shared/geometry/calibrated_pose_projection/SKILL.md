---
name: calibrated_pose_projection
description: "Provides pure pose matrix/7-vector conversion, SE(3) interpolation and pose-error calculation plus fixed U02 head-camera and wrist-camera projection utilities. Constants and arithmetic are fixed to the U02 liquid-vessel workflows."
---

# Calibrated pose projection

## Function

Provides pure pose matrix/7-vector conversion, SE(3) interpolation and pose-error calculation plus fixed U02 head-camera and wrist-camera projection utilities. Constants and arithmetic are fixed to the U02 liquid-vessel workflows.

## Inputs

- pose: 7-vector [x, y, z, qw, qx, qy, qz] or 4x4 transform, depending on helper
- first, second: 4x4 transforms; fraction: interpolation fraction; points: Nx3 coordinates
- kinematics, observation, roots, arm: own-robot state inputs for U02_wrist_camera
- pixels: Nx2 image coordinates; plane_z: workspace plane Z in meters

## Outputs

- 4x4 homogeneous transform or 7-vector pose
- Interpolated 4x4 transform; pose error tuple (translation_m, rotation_rad)
- Nx3 transformed or plane-intersected points; Nx2 projected pixels; wrist camera 4x4 transform

## Callable interface

Implementation: [implementation.py](implementation.py).

- `U02_matrix(pose)`
- `U02_pose(transform)`
- `U02_interpolate(first, second, fraction)`
- `U02_pose_error(first, second)`
- `U02_transform_points(points, transform)`
- `U02_project(points)`
- `U02_wrist_camera(kinematics, observation, roots, arm)`
- `U02_project_wrist(points, camera)`
- `U02_pixels_to_plane(pixels, plane_z)`

## Constraints

- U02 fixed head camera uses 30-degree x rotation, position [0, -0.41, 1.308], focal length 640*10/22.212, and 320/240 principal point; wrist camera uses link 'camera' with 1 mm z offset and XYZ Euler [0, -80, -90] degrees.
- Quaternion conversion uses SciPy xyzw internally and WXYZ in the 7-vector API; no mutable episode state or evaluator is stored.
- U02_project_wrist raises ValueError if any point has camera-frame z >= -0.005; U02_pixels_to_plane raises ValueError if a fixed-camera ray does not intersect the plane or scale is nonfinite.
- Inputs are explicit public own-robot state and declared camera calibration; outputs are pure geometry, not task success.
