---
name: public_command_episode
description: "Runs an acknowledged public phase program through PublicCommandActor: it builds the public robot API, resolves or receives one command request, validates and executes its phase program with NativeJointExecutor, records skill outcomes, and holds the last finite action on typed public qualification stops. It does not use a learned model or claim official success."
---

# Public command episode

## Function

Runs an acknowledged public phase program through PublicCommandActor: it builds the public robot API, resolves or receives one command request, validates and executes its phase program with NativeJointExecutor, records skill outcomes, and holds the last finite action on typed public qualification stops. It does not use a learned model or claim official success.

## Inputs

- mode and declaration naming the controller, public API, contract, and motion modules
- initial public observation with state and optional instruction
- planner object exposing plan and robot_bases
- optional controller module; otherwise tasks.<mode>.controller is imported

## Outputs

- one normalized action dict per next_action call until hold or termination
- result dict with task, actions, boundaries, phase, original_workflow_commands_complete, public_boundary_hold_reason, events, model_calls, and official_success_claimed false
- skill_selected, skill_outcome, and original_public_boundary_stop events recorded internally

## Callable interface

Implementation: [implementation.py](implementation.py).

- `PublicCommandEvidence: append(self, observation, robots)`
- `PublicCommandActor: __init__(self, mode, observation, declaration, planner, *, controller=None); record(self, event); api(self, observation); select(self, observation); finish_boundary(self, observation); next_action(self, observation); close(self); result(self)`

## Constraints

- ExecutionError becomes a public boundary stop that holds the last finite command; other implementation errors propagate
- qualification requires a singleton skill and singleton list-valued parameters matching contract ACTION_BINDINGS and ACTION_PARAMETERS
- at most 64 zero-action boundaries are processed per frame before RuntimeError
- close cancels the active program if possible and joins its worker with a 2-second timeout; physical finger or contact data is unavailable
