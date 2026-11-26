# Read a run

Every episode leaves a complete record: what the model was sent and answered, every
program, every bounded execution and its event, the outcome. This tutorial reads one.

## The episode directory

| File | What it holds |
|---|---|
| `manifest.json` | the configuration snapshot (agent version, main model, runtime, prompt, task, budgets), the agent's lock and identity, where services ran, the embodiment profile, provenance (Python, git commit and dirty state, package hash) |
| `events.jsonl` | one JSON object per event, in order (`seq`, `time`, `event`, fields) |
| `summary.json` | status, success, reward, steps, model and verifier calls, cells, tokens, cost, time, decisions by reason, events handled and escalated, the strategy chosen |
| `kernel.log` | the kernel's own log: one JSON line per tool call (`locate`, `pick`, `release`, `transfer`, `push`, ...) with its result |
| `videos/` | one MP4 per camera when the worker records video |

## The summary

```json
{"status": "finished", "success": true, "reward": 1.0, "steps": 95,
 "model_calls": 4, "verifier_calls": 0, "cells": 4, "decisions": {"start": 1, "program_end": 2, "error": 1},
 "events_handled": 0, "events_escalated": 0, "cost_usd": 0.0029, ...,
 "strategy": {"strategy": "B", "kind": "pick up the blue race toy car by 10 cm", "policy": null, "step": 0}}
```

Read `strategy` and `decisions` first. `strategy` is what the model chose with
`start(strategy, kind)`: A or C handed the task to the learned policy (`policy`; with
`policy_ran`), B kept it for the programs; `after` records a B that handed over later. A
handed-over task usually takes two calls, a look and the handover (`start`, one
`program_end`), and ends with the episode (`episode_ended`), or with up to 4 more calls
after the policy stops. A task the programs keep takes one call per contact with an
object, plus the look and the finish. `events_handled` counts events the program dealt
with itself (handlers); `events_escalated` counts those that went to the model. Many
`error` or `escalation` calls are a signal: the program is not absorbing its own
failures. The fix is usually a skill, a check or a handler, not the prompt.

## The events

```bash
python scripts/episode_status.py runs/dev               # the newest episode, one line per turn
python scripts/episode_status.py runs/dev --code        # with the programs
```

| Event | Fields |
|---|---|
| `reset` | the first observation: instruction, robot state, images |
| `model_request` | `call`, `reason`, the new messages, a summary of the whole context |
| `model_reply` | `text`, `usage` (tokens, cost, finish reason), `seconds` |
| `cell` | the program (`code`), its `result`: status, stdout, error, escalation, the executions' events, robot state |
| `verify` | the claim and the verifier's answer |
| `finish` | the summary text (`source: "program"` when a program called `finish()`, `"driver"` when the calls after the policy's run were used up) |
| `policy_to_end` | the handed-over task run on to the end of the episode once the calls after the policy's run were used up |
| `wrap_up` | empty grippers open, the arms return to the start pose |
| `outcome` | success, reward, process score, steps |
| `protocol_error`, `malformed_dropped`, `kernel_died`, `error`, `cleanup_error` | what went wrong |

The [trace reference](../reference/traces.md) lists every event and field.

## The viewer

```bash
python scripts/view_runs.py runs
python scripts/serve_runs.py runs 8000      # http://127.0.0.1:8000/
```

`view_runs.py` writes static pages next to the traces and changes none of them:

- `runs/index.html`: every run, development and evaluation layouts in separate tables,
  with suite scores per dimension;
- `<run>/index.html`: the episodes of one run;
- `<episode>/view.html`: the summary, the camera videos in step with the turns, and
  every event: each request with its images, each program (numbered, highlighted),
  each execution with its event, the outcome.

`serve_runs.py` serves them on 127.0.0.1 with HTTP range requests, which the videos need
to seek; reach it from another machine through an SSH tunnel. `scripts/run_and_view.sh`
runs `harness program` and rebuilds the pages when it ends.

## Comparing runs

Two runs are comparable when their manifests carry the same identity and their
episodes the same tasks, seeds and layouts. `harness report` on both directories gives
the per-task tables side by side; a failed or errored episode stays in the denominator.
