---
name: serial_transfer_contract
description: "Validates a bounded two-arm own-command serial transfer recipe containing left and right six-pose target sets and a public support-plane hypothesis. It returns a defensive deep copy and exposes shared stage, side, and minimum-action constants for downstream consumers."
---

# Serial transfer contract

## Function

Validates a bounded two-arm own-command serial transfer recipe containing left and right six-pose target sets and a public support-plane hypothesis. It returns a defensive deep copy and exposes shared stage, side, and minimum-action constants for downstream consumers.

## Inputs

- value: dict exactly {left, right, support_plane_m}
- support_plane_m: finite int or float public support hypothesis
- left and right: dicts exactly keyed by approach, grasp, lift, above_destination, place, retreat
- each target: finite 7-vector pose whose quaternion part has norm .99..1.01

## Outputs

- deep-copied validated recipe dict with left, right, and support_plane_m
- module constants STAGES, SIDE, TARGETS, and MINIMUM

## Callable interface

Implementation: [implementation.py](implementation.py).

- `validate_recipe(value)`

## Constraints

- raises ValueError for extra/missing keys, non-finite support, missing target sequence, non-7-vector targets, non-finite poses, or non-unit quaternions
- accepts no environment object, layout, asset dimensions, reward, or object state
- defensive copy prevents callers from mutating accepted memory
- stage order, physical-arm selection, and minimum action counts are command scheduling contracts, not proof of physical grasp or official success
