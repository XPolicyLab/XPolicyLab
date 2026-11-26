# Traces

Each episode directory holds `manifest.json`, `events.jsonl`, `summary.json`,
`kernel.log` and, when the worker records video, `videos/`. A suite directory adds
`suite.json` (experiment, agent, identity, agent version, deploy, tasks, seeds, layouts), `suite.log`,
`workers/`, and, per task, `unstable.json` / `displaced.json` when evaluation layouts
were replaced.

## manifest.json

| Key | What it holds |
|---|---|
| `schema_version`, `kind` | 1, `program` |
| `config` | `environment`, `agent_version`, `main` (the main model role), `runtime`, `verifier`, `system_prompt`, `task`, `seeds`, `max_model_calls`, `max_wall_seconds`, `task_description`, `max_cost_usd`, `agent_id` (the identity), `agent_lock`, `placement` |
| `config_sha256` | the configuration's hash |
| `seed` | |
| `embodiment_profile`, `embodiment_sha256` | the worker's profile after reset (without task-specific keys) |
| `provenance` | Python, git commit and dirty state, the package's files and hash |

## events.jsonl

One JSON object per line with `seq` (from 0), `time` (epoch seconds), `event` and its
fields.

| Event | Fields |
|---|---|
| `reset` | `observation`: `episode_id`, `step`, `instruction`, `state`, `images` (data URLs) |
| `library_loaded` | `version`, `status`, `error` |
| `model_request` | `call` (from 0), `reason`, `total_messages`, `sent_messages`, `new_messages`, `context` (one row per message: role, text start, characters, images, omitted) |
| `context_shortened` | `history_turns`, `sent_messages` (after a model ran out of GPU memory) |
| `model_reply` | `call` (from 1), `text`, `usage` (`input_tokens`, `output_tokens`, `cost`, `finish_reason`), `response_id`, `seconds`, `reasoning` when the provider returns it |
| `protocol_error` | `message`, `truncated` |
| `malformed_dropped` | `messages` |
| `verify` | `claim`, `answer` |
| `cell` | `code`; `action` (direct mode); `result`: `status`, `error`, `escalation`, `episode_ended`, `finish`, `stdout`, `seconds`, `events` (one per execution: skill, event, ticks, handled), `state`, `usage`, `report` (the skills' turn report: `text`, the cameras it outlined), `strategy`; `attachments` (count) |
| `policy_to_end` | `status`, `error`: the handed-over task run on to the end once the calls after the policy's run were used up |
| `finish` | `text`; `source`: `"program"` when a program called `finish()`, `"driver"` when the driver ended the episode after `policy_to_end` |
| `wrap_up` | `status`, `error`, `events` |
| `kernel_died` | `message` |
| `outcome` | `outcome`: `success`, `reward`, `process_score`, `terminated`, `truncated`, `steps` |
| `episode_video` | `files` (camera → path) |
| `interrupted`, `error`, `cleanup_error` | `message` |

## summary.json

| Key | |
|---|---|
| `run`, `task`, `seed` | |
| `status` | `finished`, `episode_ended`, `budget_exhausted`, `wall_time_exhausted`, `cost_exhausted`, `interrupted`, `error` |
| `success`, `reward`, `process_score`, `steps` | from the outcome |
| `model_calls`, `verifier_calls`, `cells` | |
| `input_tokens`, `output_tokens`, `cost_usd` | model and verifier, plus the vision calls programs made |
| `agent_seconds`, `kernel_seconds`, `wall_seconds` | |
| `decisions` | model calls by reason |
| `events_handled`, `events_escalated` | |
| `vision_calls`, `vision_cost_usd` | |
| `strategy` | the choice of `start(strategy, kind)`: `strategy` (A, B, C), `kind`, `policy`, `step`; `policy_ran` once the handed-over task ran; `after` when B handed over later |
| `error` | when the status is `error` |

## Execution events

`cell.result.events` lists one entry per bounded execution:

```json
{"skill": "move_eef", "event": {"kind": "COMPLETED", "source": "move_eef", "detail": {}, "tick": 22}, "ticks": 22}
```

Event kinds: `COMPLETED`, `PREDICATE_SATISFIED` (nominal); `TIMEOUT`, `BLOCKED`,
`GUARD_TRIGGERED` (or a guard's own event name), `REGION_EXIT`, `SHIELD_REJECTED`,
`REACTIVE_FAULT`, `SKILL_ERROR`, `EPISODE_ENDED`. An escalation carries the whole
trace: `skill`, `parameters`, `start_tick`, `end_tick`, `event`, `samples` (joints
every 5 ticks).
