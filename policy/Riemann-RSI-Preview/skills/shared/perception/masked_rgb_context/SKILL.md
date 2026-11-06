---
name: masked_rgb_context
description: "Returns a copy of the supplied public reference background with pixels selected by a Boolean mask replaced by the observed image, isolating that region. It validates matched uint8 three-channel images and a Boolean image mask before compositing."
---

# Masked rgb context

## Function

Returns a copy of the supplied public reference background with pixels selected by a Boolean mask replaced by the observed image, isolating that region. It validates matched uint8 three-channel images and a Boolean image mask before compositing.

## Inputs

- image: uint8[H,W,3] public RGB image.
- background: uint8[H,W,3] matching the image shape.
- mask: bool[H,W] matching the image height and width.

## Outputs

- new RGB image with image[mask] retained and surrounding context from background.
- ValueError when dtypes, channel count, shapes or mask type do not match.

## Callable interface

Implementation: [implementation.py](implementation.py).

- `run(image, background, mask)`

## Constraints

- Neither input image is modified.
- Receives image arrays only; no task, environment, pose, layout, scorer or contact information is received.
- The result is a classifier input, not a claim that hidden pixels match the reference; visibility/occlusion evidence remains separate from class probabilities.
- Action budget is zero.
