---
name: garment_scene_view
description: "Opens both gripper command channels over sixteen held-arm samples, then reads garment landmarks and searches their contact-route alternatives to prepare the next approach stage."
---

# Garment scene view

## Function

Opens both gripper command channels over sixteen held-arm samples, then reads garment landmarks and searches their contact-route alternatives to prepare the next approach stage.

## Inputs

- Public robot API, bounded stage request and same-instruction garment progress from command memory.

## Outputs

- A GarmentSceneView program, or an unavailable program when the phase, attempt or part gates fail.
- On qualified reading and route planning, updates anchor, route, phase=approach and chunk=0; otherwise records uncertainty and another attempt.

## Callable interface

Implementation: [implementation.py](implementation.py).

- `GarmentSceneView: review_completion(self)`
- `run(api, request)`

## Constraints

- Requires view phase, fewer than three attempts and part index below three.
- Reuses the first flat anchor for the second sleeve and requests the sleeves-folded reading for the torso stage. Candidate contact-route ValueErrors skip that candidate.
- Gripper opening is a commanded ramp; the program does not measure physical finger contact or determine official success.
