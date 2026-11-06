---
name: garment_camera_projection
description: "Builds the garment pinhole camera model from current native RGB, EEF poses, and the hashed fixed X5 URDF; projects 3D points and back-projects bounded pixels to rays or caller-declared plane heights. It supplies model geometry only, not surface, visibility, contact, or task-success evidence."
---

# Garment camera projection

## Function

Builds the garment pinhole camera model from current native RGB, EEF poses, and the hashed fixed X5 URDF; projects 3D points and back-projects bounded pixels to rays or caller-declared plane heights. It supplies model geometry only, not surface, visibility, contact, or task-success evidence.

## Inputs

- api: image(camera) and targets() provider for current RGB and EEF poses
- camera: cam_head, cam_left_wrist, or cam_right_wrist
- pixels: finite Nx2 pixel array with N from 1 to 1024, or up to 307200 for pixel_rays
- points or height: finite Nx3 points for project_points or scalar plane height for points_on_plane

## Outputs

- pixel_rays returns optical origin and unit rays in world coordinates
- project_points returns Nx2 pixels with NaN for points behind the camera
- points_on_plane returns Nx3 points with NaN for invalid or nonpositive intersections
- camera_model returns a 4x4 camera-to-world transform and focal length in pixels

## Callable interface

Implementation: [implementation.py](implementation.py).

- `wrist_optical_from_link6()`
- `camera_model(api, camera)`
- `pixel_rays(api, camera, pixels)`
- `project_points(api, camera, points)`
- `points_on_plane(api, camera, pixels, height)`

## Constraints

- Requires api.image(camera).shape == (480, 640, 3); rejects other cameras or image shapes.
- Requires the declared fixed X5 URDF to be non-symlink and match its recorded SHA-256; wrist camera reads only fixed hand_to_camera_mount and camera_joint ancestors.
- Plane height is a caller-declared geometric reference, not observed support; no visibility/contact/folding evidence is produced.
- Wrist transform is cached and marked read-only; invalid shapes, nonfinite values, or out-of-image pixels raise ValueError.
