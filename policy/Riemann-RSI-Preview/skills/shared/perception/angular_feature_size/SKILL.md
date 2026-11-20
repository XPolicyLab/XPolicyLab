---
name: angular_feature_size
description: "Estimates an approximate metric visual feature size from a current region's centroid and major bbox extent, public camera rays, and a triangulated point. It offsets the centroid along the wider image axis, measures the half-angle between the two offset rays, and multiplies the point-to-ray-origin distance by sin(angle)."
---

# Angular feature size

## Function

Estimates an approximate metric visual feature size from a current region's centroid and major bbox extent, public camera rays, and a triangulated point. It offsets the centroid along the wider image axis, measures the half-angle between the two offset rays, and multiplies the point-to-ray-origin distance by sin(angle).

## Inputs

- api: adapter object passed to pixel_rays
- camera: camera accepted by pixel_rays
- region: mapping with centroid (2,) and bbox width,height at indices 2 and 3
- point: finite length-3 triangulated point

## Outputs

- float: approximate feature size in distance units of point and ray origin

## Callable interface

Implementation: [implementation.py](implementation.py).

- `run(api, camera, region, point)`

## Constraints

- Raises ValueError unless point shape is (3,), centroid shape is (2,), point is finite, and min(width,height)>0.
- Uses max(width,height)/2 pixel offset along the wider bbox image axis.
- Approximate visual size, not an asset dimension, contact measurement, or guaranteed circular horizontal surface.
- Can reject inconsistent stereo correspondences without known object height or a fixed image window.
