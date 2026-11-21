# Pudu_Agent

**Contributor:** Pudu Robotics | **Submission type:** Agent | **Scaffold origin:** RGB-only L4 Inspect EEF adapter (`policy/GPT_6_Astra_Direct_EEF`); this entry keeps that task-agnostic tool-use boundary and adds a calibrated-geometry `locate` skill, in-episode planning, and a selectable planner model.

Evaluation-only, RGB-only tool-use agent. Each turn the model sees the camera images, the current world-frame grasp-point state, and the task's official Detail Description, and emits one tool call. It acts through a small, task-agnostic skill set: `move_eef` (name the world-frame grasp-point dimensions to change), `locate` (turn a pixel into a world point using the cameras' published intrinsics/extrinsics — single-pixel ray-plane or two-pixel triangulation), `update_plan` (an in-episode plan the model keeps current), and `return_home`. RoboDojo's local Cartesian planner turns the EEF poses into time-aligned joint paths. There is no VLA checkpoint and no training entry; the planner model is selectable (`L4_INSPECT_PLANNER`). Shared Inspect runtime lives in `inspect/`.

**Compliance (Agent track, rules of 2026-11-14).** Nothing here is keyed on the task name: the same skills, prompt, and model run every task. The only task text is the official Detail Description; no per-task trajectory, program, tolerance, or checkpoint is pre-written, and RoboDojo rollout outcomes are not used to tune anything. Camera geometry and a general visual-localization skill are used as the rules permit.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

The policy server loads no weights. It borrows a Python that can import XPolicyLab (`uv` resolves `policy_uv_env_path` in `deploy.yml`, default `../Pi_05/openpi`). The eval client is where the LLM runs, and that interpreter needs `openai==3.8.0` and `pillow==12.3.0`:

```bash
bash policy/Pudu_Agent/install.sh \
  /path/to/RoboDojo-eval/.venv/bin/python
```

Set the planner's key before evaluating. For the GPT planners that is `ARK_API_KEY` (`ARK_API_KEY_BACKUP` optional). For the Claude planners (`fable`/`opus55`) point `L4_INSPECT_API_KEY_ENV` at the variable holding the endpoint token (see **Planner** below).

## Data Processing

Unsupported. This is an eval-only adapter: `process_data.sh` is omitted, and it does not convert datasets.

## Training

Unsupported. `train.sh` is omitted. There is no checkpoint directory and no training release to schedule. Actions are produced at eval time by the provider API.

## Evaluation

> **Submitted configuration.** The leaderboard entry runs `L4_INSPECT_PLANNER=fable` (`claude-fable-5-2`, reasoning effort `high`) against the Claude endpoint described in **Planner → Claude planner** below; the endpoint URL and token are supplied to evaluators in the evaluation application. The `astra`/`gpt55` GPT planners remain available for anyone reproducing the scaffold with their own key.

Layout-range entry, same arguments as the original EEF adapter. There is no separate policy GPU: the simulator uses `env_gpu`, and the only inference is the API call. Default task is `arrange_largest_number`. Default planner hard timeout is 90 seconds.

```bash
export ARK_API_KEY=...
export ROBODOJO_ROOT=/path/to/RoboDojo-eval

bash policy/Pudu_Agent/run_fixed_layout.sh \
  0 0 general_pickup uv
```

Positional arguments are `<layouts> <env_gpu> <task> <eval_env>`. `layouts` is one index (`16`), an inclusive range (`20-39`), or a mix (`1-3,5,12-14`).

Standard harness entry:

```bash
bash policy/Pudu_Agent/eval.sh \
  RoboDojo general_pickup no-checkpoint arx_x5 joint 0 0 0 uv uv
```

`policy_env` (arg 9) is `uv` or a uv project directory that contains `.venv/bin/python`. `eval_env` (arg 10) is `uv` (the `ROBODOJO_ROOT/.venv`), a virtualenv directory, a Python binary, or a conda env that already has the pinned `openai` and `pillow` versions.

`action_type` must stay `joint`: EEF is the model-facing action, and RoboDojo receives the planner's joint path. `env_cfg_type` is `arx_x5`. `ckpt_name` is ignored; `no-checkpoint` is the conventional placeholder. Batch evaluation is not supported.

Offline wiring check (no simulator). The debug client has no Cartesian planner, so motions are not planned. The loop still builds the action spec from the RoboDojo URDF (`ROBODOJO_ROOT/Assets/Robots/x5/X5A.urdf`) and can stop without calling the provider when `L4_INSPECT_MAX_LLM_CALLS=0`:

```bash
export EVAL_ENV_TYPE=debug
export ROBODOJO_ROOT=/path/to/RoboDojo-eval
export ARK_API_KEY=...
bash policy/Pudu_Agent/eval.sh \
  RoboDojo stack_bowls no-checkpoint arx_x5 joint 0 0 0 uv <eval_env>
```

Re-run with `DEBUG_OBS_ENCODED=1` to exercise encoded camera colors. The client decodes those with `decode_image_bit`.

## Configuration

Extra `deploy.yml` keys beyond the shared set:

| Key | Role |
| --- | --- |
| `policy_uv_env_path` | uv project used when `policy_env` is `uv` (default `../Pi_05/openpi`) |
| `result_dir` | default result directory name |
| `obs_transform_pipeline` | observation transform tag (`xspark-v1.0`) |

Planner overrides (environment):

| Variable | Default | Role |
| --- | --- | --- |
| `L4_INSPECT_PLANNER` | `astra` | `astra` → `gpt-6-astra` (responses); `gpt55` → `gpt-5.5-2026-04-24` (responses); `fable` → `claude-fable-5-2` (chat); `opus55` → `claude-opus-5-5` (chat) |
| `L4_INSPECT_BASE_URL` | provider default | chat/completions endpoint; set to the Claude endpoint for the Claude planners |
| `L4_INSPECT_API_KEY_ENV` | `ARK_API_KEY,ARK_API_KEY_BACKUP` | comma-separated names of env vars holding the endpoint token(s) |
| `L4_INSPECT_MODEL` | planner default | override the model id |
| `L4_INSPECT_REASONING_EFFORT` | `medium` | `low`..`max` |
| `L4_INSPECT_MAX_LLM_CALLS` | `100` | per-episode call budget |
| `L4_INSPECT_DEPTH` | `off` | must stay `off` or unset |

### Claude planner (endpoint access path for evaluators)

The `fable`/`opus55` planners call an OpenAI-compatible `chat/completions` endpoint that fronts Claude; it
is not a public Anthropic API — it is reached over the internet at a URL and bearer token the submitter
provides in the evaluation application. To run it:

```bash
export L4_INSPECT_PLANNER=fable
export L4_INSPECT_BASE_URL=<endpoint base url>      # given in the application
export RCA_KEY=<endpoint token>                      # given in the application
export L4_INSPECT_API_KEY_ENV=RCA_KEY
export L4_INSPECT_REASONING_EFFORT=high
```

The endpoint takes standard `chat/completions` (image parts as `image_url` data URIs, function `tools`),
so no client change is needed. Evaluators without that URL/token can reproduce the scaffold against any
GPT planner (`astra`/`gpt55`) with their own `ARK_API_KEY`.

## Notes

- The model estimates Cartesian targets from RGB and proprioception. It does not receive metric depth.
- Data processing and training are unsupported.
- A real episode needs `ROBODOJO_ROOT`, a RoboDojo client with CuRobo, and `ARK_API_KEY`.
