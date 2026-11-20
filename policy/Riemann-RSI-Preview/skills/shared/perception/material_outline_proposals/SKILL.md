---
name: material_outline_proposals
description: "Builds colour-threshold foreground masks for public material appearances and returns deduplicated immutable outline proposals. material_mask returns the primary mask; run adds independent warm, purple, and brightness-band alternatives."
---

# Material outline proposals

## Function

Builds colour-threshold foreground masks for public material appearances and returns deduplicated immutable outline proposals. material_mask returns the primary mask; run adds independent warm, purple, and brightness-band alternatives.

## Inputs

- Current RGB image as a uint8 HxWx3 array with 1 to 307200 total pixels.
- extended bool selecting whether material_mask includes purple and salmon thresholds in the primary mask.

## Outputs

- material_mask returns a uint8 primary foreground mask from blue/red/orange thresholds, with purple and salmon added only when extended is true.
- run returns a tuple of at most thirteen immutable boolean masks, deduplicated by array equality and excluding empty proposals.

## Callable interface

Implementation: [implementation.py](implementation.py).

- `material_mask(image, extended=False)`
- `run(image, extended=False)`

## Constraints

- material_mask raises ValueError unless the input is an actual uint8 three-channel RGB array within the pixel bound and extended is an explicit bool.
- run includes the primary mask, five warm-colour delta proposals, one purple proposal, and two brightness-band proposals for each floor value 140, 180, and 220.
- Each returned proposal has flags.writeable set to False.
- A material mask is not an object detector or calibrated probability; foreground gaps and holes are preserved, and unrelated coloured clutter still needs independently supported outline evidence.
