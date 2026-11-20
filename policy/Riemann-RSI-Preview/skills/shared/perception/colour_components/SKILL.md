---
name: colour_components
description: "Extracts bounded colour regions from a current public RGB observation for a declared camera and colour. It thresholds OpenCV HSV, applies ellipse open/close morphology, runs eight-connected components, and returns up to six largest regions above a minimum area."
---

# Colour components

## Function

Extracts bounded colour regions from a current public RGB observation for a declared camera and colour. It thresholds OpenCV HSV, applies ellipse open/close morphology, runs eight-connected components, and returns up to six largest regions above a minimum area.

## Inputs

- observation: public observation with vision[camera] as RGB array or dict with color
- request: mapping with camera and colour key in COLOURS
- previous: optional prior value accepted but unused by run

## Outputs

- dict with kind, status observed/not_observed, camera, colour, components, and interpretation
- components: list of centroid_image and box_image normalized by image width/height, plus area_fraction

## Callable interface

Implementation: [implementation.py](implementation.py).

- `hue_distance(value, reference)`
- `components(mask, roi)`
- `rgb_image(observation, camera)`
- `colour_components(image, colour)`
- `image_geometry(component, image_shape)`
- `run(observation, request, previous=None)`

## Constraints

- COLOURS hue references: blue 112, green 60, red 0, yellow 28, orange 14, cyan 88 on OpenCV HSV hue scale.
- Mask uses hue distance <= 11, saturation >= 75, and value >= 45.
- Minimum area is max(20,int(height*width*0.0004)); returns at most six largest components.
- rgb_image scales floating images with max <= 1 by 255, requires finite RGB, and clips the first three channels to uint8; regions are current-image only, not object identity, world poses, or grasp poses.
