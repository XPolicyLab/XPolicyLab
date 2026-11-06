---
name: turn_based_board_contract
description: "Defines and validates the finite 5-turn place/home/wait own-command sequence, public board anchors, wait observations, and monotonic progress. advance appends one issued placement and checks that home/wait follow the same cell and source slot."
---

# Turn based board contract

## Function

Defines and validates the finite 5-turn place/home/wait own-command sequence, public board anchors, wait observations, and monotonic progress. advance appends one issued placement and checks that home/wait follow the same cell and source slot.

## Inputs

- wait observation: exact dict of visible_cells, expected_count, changed_pixels, board_region_pixels, stable_frames, action_count, budget_exhausted, history, official_success_claimed
- anchor: cells (9x3), source_homes (5x3), cell_spacing_m, source_top_plane_m
- progress: next_stage, own_commanded_cells, used_source_slots
- advance arguments: turn, stage, cell index, source slot index

## Outputs

- validate_wait_observation returns the validated wait dict; validate_anchor and validate_progress return deep copies
- advance returns the next validated progress; empty_progress returns zeroed progress
- minimum(stage, turn) returns action minima place 24 (turn 0) or 19, home 2, wait 12

## Callable interface

Implementation: [implementation.py](implementation.py).

- `validate_wait_observation(value)`
- `minimum(stage, turn)`
- `validate_anchor(value)`
- `validate_progress(value)`
- `advance(progress, turn, stage, cell, source)`
- `empty_progress()`

## Constraints

- SEQUENCE is 5 turns of place, home, wait, omitting only turn 4 wait; stage order and cursor must match.
- Wait observations are bounded public image/action data: nine bool-or-None visible cells, expected_count in {2,4,6,8}, board_region_pixels <=307200, stable_frames/action_count <=250, and at most 10 samples with increasing action_count.
- Anchors must be finite, max absolute coordinate <=3, cell_spacing_m in 0.02..0.25 m; progress rejects unknown or repeated cell/source decisions.
- Progress means issued command history, not measured contact or official success; hidden poses, layouts, real fingers, and score fields are rejected.
