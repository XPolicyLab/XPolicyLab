---
name: glyph_template_bank
description: "Loads exact public v23 digit records and masks from digits.json and digits.npz, then returns rotated template masks filtered by family. rotated defaults to angles -60..60 step 3, caches by (family, angles), and uses nearest-neighbor warping with the negative physical-yaw convention."
---

# Glyph template bank

## Function

Loads exact public v23 digit records and masks from digits.json and digits.npz, then returns rotated template masks filtered by family. rotated defaults to angles -60..60 step 3, caches by (family, angles), and uses nearest-neighbor warping with the negative physical-yaw convention.

## Inputs

- directory containing digits.json and digits.npz.
- rotated(family, angles=None); angles defaults to range(-60, 61, 3).

## Outputs

- records loaded from digits.json and masks loaded from digits.npz.
- list of (record, angle, mask) tuples cached per (family, angles).
- RASTER_SIZE=192 and RASTER_SCALE=2400.0 constants.

## Callable interface

Implementation: [implementation.py](implementation.py).

- `TemplateBank: __init__(self, directory); rotated(self, family, angles=None)`

## Constraints

- Uses stored external templates; never rebuilds silhouettes from USD or imports official assets online.
- Mask rotation uses cv2.warpAffine with INTER_NEAREST; a positive OpenCV angle is negative physical yaw.
- family filtering skips records whose family differs.
