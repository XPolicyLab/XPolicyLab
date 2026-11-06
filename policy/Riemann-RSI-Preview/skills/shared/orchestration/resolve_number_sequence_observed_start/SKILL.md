---
name: resolve_number_sequence_observed_start
description: "Selects at most one next numeral-sorting operation from bounded public history and current adapter state, returning an operation dict or an empty non-applicable result. It sequences current views, placement stages, bounded recovery, relay, and final own-joint homes without claiming official success."
---

# Resolve number sequence observed start

## Function

Selects at most one next numeral-sorting operation from bounded public history and current adapter state, returning an operation dict or an empty non-applicable result. It sequences current views, placement stages, bounded recovery, relay, and final own-joint homes without claiming official success.

## Inputs

- api: public adapter exposing instruction(), image('cam_head'), targets(), and measurements used by imported helpers
- memory: bounded own history of skill events with request, execution_feedback, and outcome fields
- maximum_uncertain_attempts: integer uncertainty/zero-action retry cap, clamped to 0..2

## Outputs

- resolver dict with applicable, operations, relation {kind: public_number_sorting, cursor, stage}, and reason
- operations entries name one skill and wrap each parameter value in a one-element list; operations is empty when no next skill is qualified

## Callable interface

Implementation: [implementation.py](implementation.py).

- `operation(name, parameters, progress, stage, reason)`
- `matching_attempts(memory, skill, progress, item=None)`
- `actual_views(memory, progress)`
- `view_request(api, memory, progress, maximum_attempts)`
- `run(api, memory, maximum_uncertain_attempts=2)`

## Constraints

- maximum_uncertain_attempts is clamped to 0..2; initial actual views are capped at 5 when the cap is positive and zero-action attempts at twice that bound, otherwise at the supplied cap.
- A completed actual re-view for the current cursor clears earlier verification failures; another cursor or an unfinished view does not.
- Grasp recovery is one open-and-reset before choosing an orthogonal mature grasp axis; relay_number is not repeated or blind, and missing targets are not filled or routed by task ID.
- Own EE poses have shape (7,) and openings (1,); final home commands and visual placement are command evidence, not official success.
