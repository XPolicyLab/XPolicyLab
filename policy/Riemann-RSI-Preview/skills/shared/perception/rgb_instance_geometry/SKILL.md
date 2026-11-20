---
name: rgb_instance_geometry
description: "bright_faces detects bright saturated colour components and selects faces whose hue matches the requested declared colour; surface_for_instance and contact_for_instance associate current head-image centroids with current wrist-image candidates through triangulation and plane geometry. choose_consistent selects a unique metric hypothesis when corroborated by both wrist cameras, otherwise applies the conservative single-camera ambiguity check."
---

# Rgb instance geometry

## Function

bright_faces detects bright saturated colour components and selects faces whose hue matches the requested declared colour; surface_for_instance and contact_for_instance associate current head-image centroids with current wrist-image candidates through triangulation and plane geometry. choose_consistent selects a unique metric hypothesis when corroborated by both wrist cameras, otherwise applies the conservative single-camera ambiguity check.

## Inputs

- api: adapter object providing image(camera) and the calibration-dependent triangulate and face_on_plane dependencies
- head_face or head_region: current head-image face or region containing a centroid
- colour: one of FACE_HUES keys blue, green, red, yellow, orange, or cyan
- regions: callable returning wrist-image region dicts for contact_for_instance

## Outputs

- surface_for_instance and contact_for_instance return a chosen candidate dict with point, ray_separation_m, ray_angle_radians, cameras, and optional surface fields, or None
- bright_faces returns face dicts with pixels, centroid, area, brightness_threshold, and whole_area, sorted by centroid x when at most 16 faces are found
- choose_consistent returns None for conflicting uncorroborated hypotheses or competing corroborated hypotheses

## Callable interface

Implementation: [implementation.py](implementation.py).

- `bright_faces(image, colour)`
- `choose_consistent(candidates)`
- `surface_for_instance(api, head_face, colour)`
- `contact_for_instance(api, head_region, regions)`

## Constraints

- bright_faces accepts saturation >= 105, value >= 180, area from 30 to 3% of image pixels, aspect ratio <= 2.2, interior bounding boxes, and hue nearest FACE_HUES.
- choose_consistent prefers a unique candidate corroborated by both cam_left_wrist and cam_right_wrist within 0.015 m; without corroboration it rejects candidates within 0.003 m ray separation but more than 0.015 m apart.
- These are hypotheses from declared RGB and own-camera calibration, not simulator object poses; clipped views or unresolved correspondence require another bounded public view.
- The functions are bound to the adapter-provided api and imported helpers; standalone invocation is not guaranteed.
