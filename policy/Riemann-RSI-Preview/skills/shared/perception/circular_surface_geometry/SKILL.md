---
name: circular_surface_geometry
description: "Fits a circular visible-surface hypothesis from current public image pixels projected onto a caller-qualified plane using points_on_plane. It convex-hulls the region pixels, computes a minimum enclosing circle, and returns a bounded metric point and radius or None."
---

# Circular surface geometry

## Function

Fits a circular visible-surface hypothesis from current public image pixels projected onto a caller-qualified plane using points_on_plane. It convex-hulls the region pixels, computes a minimum enclosing circle, and returns a bounded metric point and radius or None.

## Inputs

- api: adapter object passed to points_on_plane
- camera: camera accepted by points_on_plane
- region: mapping with pixels Nx2 and area
- height: caller-qualified plane height

## Outputs

- dict with point [x,y,height], radius_m, and face_area_pixels, or None

## Callable interface

Implementation: [implementation.py](implementation.py).

- `run(api, camera, region, height)`

## Constraints

- Raises ValueError unless pixels are a finite Nx2 array with 3 <= len(pixels) <= 307200.
- Caps the convex hull to at most 1024 sampled hull points.
- Returns None if projection is None or non-finite, or if radius is outside .005 to .12 metres.
- Appearance-independent; cannot access environment, layout, depth, or reward.
