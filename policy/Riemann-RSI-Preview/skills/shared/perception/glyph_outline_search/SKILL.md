---
name: glyph_outline_search
description: "Proposes numeral locations by matching projected public outline edges against current LAB-image edges, then builds independent local foreground masks using seeded GrabCut."
---

# Glyph outline search

## Function

Proposes numeral locations by matching projected public outline edges against current LAB-image edges, then builds independent local foreground masks using seeded GrabCut.

## Inputs

- Numeral tools/template bank, current RGB, calibrated camera and own mounting frame.
- Optional placed flag and expected_xy localization bound; regions also receives the edge proposals.

## Outputs

- Ranked current-image edge proposals with digit alternatives, yaw, position and edge-distance evidence.
- Deduplicated local connected regions with masks, image hue and proposal provenance.

## Callable interface

Implementation: [implementation.py](implementation.py).

- `propose(tools, image, camera, frame, state, placed=False, expected_xy=None)`
- `regions(tools, image, camera, frame, proposals, placed=False)`

## Constraints

- Searches declared tabletop coordinates and yaw ranges, then requires current foreground/background contrast. An expected location restricts locality without supplying the foreground mask identity.
- GrabCut seeds depend on local brightness, chroma and distance; the proposed identity does not condition foreground segmentation.
- Proposals remain image-recognition hypotheses requiring downstream independent identity verification.
