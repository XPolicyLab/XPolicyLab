---
name: public_rule_policy_interface
description: "Implements the official reset, update_obs, and get_action policy surface for one blocking observation-driven workflow, using a worker thread and a single action-acknowledgement exchange. It validates projected public observations and normalized joint targets, returns one action per nonempty chunk, and records diagnostics without claiming success."
---

# Public rule policy interface

## Function

Implements the official reset, update_obs, and get_action policy surface for one blocking observation-driven workflow, using a worker thread and a single action-acknowledgement exchange. It validates projected public observations and normalized joint targets, returns one action per nonempty chunk, and records diagnostics without claiming success.

## Inputs

- runner callable taking (initial_observation, exchange) where exchange(action) blocks for the next observation
- public observation containing projected state and vision, with optional instruction string up to 5000 characters
- action dict with exactly four joint fields: either arm_joint_state widths 6 or ee_pose width 7 plus ee_joint_state width 1
- wait_timeout, join_timeout, and optional planning_poll_seconds

## Outputs

- get_action returns [target] for a nonempty chunk or [] for a bounded empty planning poll
- diagnostics dict with observation and action counts, empty_planning_chunks, workflow completion or error, worker_alive, and official_success_claimed false
- TimeoutError, RuntimeError, RulePolicyFailure, or PolicyEpisodeClosed on the corresponding failures

## Callable interface

Implementation: [implementation.py](implementation.py).

- `PolicyEpisodeClosed: exception or data type`
- `RulePolicyFailure: exception or data type`
- `normalized_action(action)`
- `BlockingRulePolicy: __init__(self, runner: Callable, *, wait_timeout=180.0, join_timeout=5.0, planning_poll_seconds=None); reset(self); close(self); update_obs(self, observation); get_action(self); diagnostics(self)`

## Constraints

- only one workflow command may be outstanding; a fresh official observation is required before each action
- an empty planning chunk delivers no command, replays no previous command, and must not acknowledge a pending action
- reset and close cancel the old exchange and join its worker; a worker that does not exit causes a cleanup error and closes the policy
- normalized_action requires finite shapes, unit quaternions within .01, and gripper commands in [0,1] as normalized commands rather than measured widths; PolicyEpisodeClosed derives from BaseException so broad legacy except Exception handlers cannot resume a cancelled episode
