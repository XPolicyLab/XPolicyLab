---
name: glyph_normalization
description: "Crops a glyph foreground, rescales it into a square with an eight-pixel size allowance, thresholds the resized crop and centres it on a black canvas."
---

# Glyph normalization

## Function

Crops a glyph foreground, rescales it into a square with an eight-pixel size allowance, thresholds the resized crop and centres it on a black canvas.

## Inputs

- A two-dimensional mask converted to uint8, normally with foreground intensity 255, and output size (default 64).

## Outputs

- A size-by-size uint8 image with values 0 or 255.

## Callable interface

Implementation: [implementation.py](implementation.py).

- `normalize_glyph(mask, size=64)`

## Constraints

- Raises RuntimeError when fewer than twenty input pixels are nonzero.
- Uses INTER_AREA resizing and a fixed threshold of 96; a 0/1 mask is not automatically rescaled to 0/255. This is image normalization, not numeral recognition.
