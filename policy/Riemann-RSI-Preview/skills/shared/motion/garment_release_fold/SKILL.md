---
name: garment_release_fold
description: "Runs the original cuff or hem release clock for the selected own arms using sampled_gripper_clock with release=True. Cuff release opens to a stationary full-open interval; hem release preserves the distinct partial-open detachment clock before withdrawal."
---

# Garment release fold

## Function

Runs the original cuff or hem release clock for the selected own arms using sampled_gripper_clock with release=True. Cuff release opens to a stationary full-open interval; hem release preserves the distinct partial-open detachment clock before withdrawal.

## Inputs

- api PublicRobotAPI and request with a validated sixteen-command allowance
- recorded same-instruction state with part and route arms

## Outputs

- GarmentProgram with sixteen controls and following set to clear, or an unavailable result

## Callable interface

Implementation: [implementation.py](implementation.py).

- `run(api, request)`

## Constraints

- Requires allowed(current, 'release')
- Uses release=True and cuff=current['part'] != 2; no generic release clock is used
- Jaw opening values are normalized command/observation values, not physical finger width or contact
- Runs within the bounded sixteen-command chunk
