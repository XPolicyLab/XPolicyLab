---
name: public_relation_probe
description: "Runs a bounded public relation probe: it validates the recorded phase/goal, performs clearing or wrist focus when needed, and schedules settled observation, approach, or gripper phases. Observation results must use the closed {'scene','no_more'} contract."
---

# Public relation probe

## Function

Runs a bounded public relation probe: it validates the recorded phase/goal, performs clearing or wrist focus when needed, and schedules settled observation, approach, or gripper phases. Observation results must use the closed {'scene','no_more'} contract.

## Inputs

- api: adapter exposing command_memory(), instruction(), motion/perception helpers, and actual_request/clear_view/unavailable/own_opening
- request: mapping with side 'auto', 'left', or 'right'
- semantic_goal: callable mapping api.instruction() to a qualified goal or None
- observe: callable(api, current) returning exactly {'scene': ..., 'no_more': ...}

## Outputs

- unavailable(api, request, current) when the goal/phase/attempt/observation bounds fail
- clear_view(api, request, current) during the clear phase
- RelationViewMotion approach program or RigidClock gripper program for view/approach execution
- completion reviewer mutates current scene/phase/home_leg/view_attempts or sets program.status='uncertain'

## Callable interface

Implementation: [implementation.py](implementation.py).

- `RelationViewMotion: __init__(self, *arguments, **keywords); observe(self, observation)`
- `run(api, request, semantic_goal, observe)`

## Constraints

- Requires phase in ('clear','view'), attempts < 3, job.empty_observations < 3, job.observation_count < 64, and current goal None or equal to semantic_goal.
- Raises ValueError if observe or the settled observation result keys are not exactly {'scene','no_more'}.
- Counts at most one view_attempt per RelationViewMotion instance after actions exist, capped at 4; zero-action precondition failures do not count.
- After settled observation, a scene clears empty_observations and enters pregrasp, no_more enters home, otherwise empty_observations increments up to 3 and status becomes uncertain; this is not official success.
