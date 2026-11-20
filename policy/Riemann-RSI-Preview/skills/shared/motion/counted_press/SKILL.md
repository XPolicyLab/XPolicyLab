---
name: counted_press
description: "Builds a phased press program: approach to a hover pose, then for each requested cycle a down move, dwell hold, and complete release back to hover. Returns PhasedProgram with a precondition based on instruction text containing press or button."
---

# Counted press

## Function

Builds a phased press program: approach to a hover pose, then for each requested cycle a down move, dwell hold, and complete release back to hover. Returns PhasedProgram with a precondition based on instruction text containing press or button.

## Inputs

- api adapter exposing targets(), instruction(), and PhasedProgram compatibility
- request with side, delta relative translation, depth downward distance, count cycles, and dwell hold duration

## Outputs

- PhasedProgram containing approach_visible_button, press_down_N, press_hold_N, and press_release_N phases
- precondition boolean true only when lowercased api.instruction() contains press or button

## Callable interface

Implementation: [implementation.py](implementation.py).

- `run(api, request)`

## Constraints

- Bound to the supplied api adapter; run reads api.targets() and api.instruction() and constructs PhasedProgram(api, request, phases, precondition=applicable).
- Assumes current public observations already qualify a hover and caller supplies declared relative command parameters.
- Generic primitive does not read number cards, choose confirmation roles, or count official activations; the precondition is not official success.
- Preserves separate approach, down, hold, and full-release phases; numerical procedure is unchanged from the extracted generic press tool.
