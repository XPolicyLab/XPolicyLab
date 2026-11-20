---
name: rgb_surface_geometry
description: "Builds metric hypotheses from current RGB using the declared static X5 camera rig: detects bright colour faces, triangulates centroids, estimates face-on-plane geometry, and assembles scene_geometry targets with a same-episode support-plane hypothesis. Exported helpers cover camera rays, triangulation, height intervals, and face-on-plane fitting."
---

# Rgb surface geometry

## Function

Builds metric hypotheses from current RGB using the declared static X5 camera rig: detects bright colour faces, triangulates centroids, estimates face-on-plane geometry, and assembles scene_geometry targets with a same-episode support-plane hypothesis. Exported helpers cover camera rays, triangulation, height intervals, and face-on-plane fitting.

## Inputs

- api: adapter object providing image(camera) and targets() with link-six end-effector poses
- colours: bounded tuple of declared colour names for scene_geometry
- remembered: optional same-episode dict with support_plane_m and raised_colours
- camera, pixels, face, height, and estimator arguments for exported helpers

## Outputs

- scene_geometry returns status observed or uncertain, support_plane_m, targets, calibration declared_static_x5_rig, and an interpretation string
- triangulate returns point, ray_separation_m, ray_angle_radians, and cameras, or None
- triangulated_height_interval returns min and max height in metres, or None
- face_on_plane returns point, edge_m, yaw_radians, and face_area_pixels, or None

## Callable interface

Implementation: [implementation.py](implementation.py).

- `rotation_matrix(quaternion)`
- `euler_matrix(angles)`
- `fixed_transform(translation, rotation)`
- `camera_model(api, camera)`
- `pixel_rays(api, camera, pixels)`
- `bright_face(image, colour)`
- `triangulate(api, first_camera, first_pixel, second_camera, second_pixel)`
- `triangulated_height_interval(api, first_camera, first_pixel, second_camera, second_pixel, estimator=None)`
- `face_on_plane(api, camera, face, height)`
- `scene_geometry(api, colours, remembered=None)`

## Constraints

- camera_model supports only native 640x480 RGB for cam_head, cam_left_wrist, and cam_right_wrist; other cameras raise ExecutionError.
- triangulate rejects origins less than 0.02 m apart, ray angles below 8 degrees, non-positive distances, or residual greater than 0.004 m; pixel_rays require finite in-image pixels.
- face_on_plane rejects non-positive distances and rectangles with min edge < 0.01 m, max edge > 0.10 m, or aspect ratio > 1.6; triangulated_height_interval propagates half-pixel offsets and returns None if any perturbation is invalid.
- Estimates are current RGB cube-face and same-episode support hypotheses, not scene ground truth; no depth, object state, layout, teacher, reward, or camera-manager access is allowed, and official task success is not proven.
