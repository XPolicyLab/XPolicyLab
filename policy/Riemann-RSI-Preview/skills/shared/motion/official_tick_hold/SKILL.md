---
name: official_tick_hold
description: "Emits the current public command targets for an exact bounded number of official action ticks. Each emission is checked against the caller's own command counter to ensure it consumes exactly one tick."
---

# Official tick hold

## Function

Emits the current public command targets for an exact bounded number of official action ticks. Each emission is checked against the caller's own command counter to ensure it consumes exactly one tick.

## Inputs

- actions: integer count of official command ticks, inclusive 0-1600
- emit_one: zero-argument callable that emits one public command tick
- read_count: zero-argument callable returning the actor's own public command count

## Outputs

- returns the validated integer actions count after all requested ticks
- raises ValueError for a non-integer or out-of-range action count
- raises TypeError when emit_one or read_count is not callable
- raises RuntimeError when an emission does not increase read_count() by exactly one

## Callable interface

Implementation: [implementation.py](implementation.py).

- `run(actions, *, emit_one, read_count)`

## Constraints

- Each iteration reads the counter before emission, emits once, reads again and requires the new count to equal the old count plus one
- Zero actions performs no emission and returns 0
- Accepts no simulator environment, scorer, contact state or object metadata; callers must propagate public episode termination
- The helper itself does not issue commands outside the supplied one-tick emitter
