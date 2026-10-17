# RoboDojo_Agent_L3_Inspect

Alternative RGB-only L3 reference harness with a joint-space model-facing
action. The model calls `move_joints` with absolute named targets; local
non-learned code validates, clamps and interpolates them into RoboDojo-native
joint actions.

## Disclosure

- **Level:** L3 LLM-as-Policy
- **Status:** Alternative reference; no published score
- **Models:** named planner profiles for GPT-6 Astra, GPT-5.5 and Kimi K3;
  provider endpoint and credentials are runtime configuration
- **Prompt:** generated in [`policy.py`](policy.py), with optional task recipes
- **Tools:** `move_joints`, `give_up`; optional probe-only `done`
- **Motion stack:** joint-bound validation and velocity-limited interpolation
- **Memory:** configurable image horizon or complete image history
- **Default budget:** 100 LLM calls
- **Reference difference:** joint targets rather than Inspect EEF Cartesian
  grasp-point targets

## Observation boundary

The harness reads three RGB cameras, the official instruction and measured
robot state. It rejects metric depth. It does not read object pose, layout
metadata or reward internals. RoboDojo remains the only scorer.

## Provider configuration

Select a profile with `L3_INSPECT_PLANNER=astra|gpt55|kimi`. The host is
required: there is no default, because a missing URL used to fall through
to `https://api.openai.com/v1` and time out on the first call.

```bash
export L3_INSPECT_BASE_URL=https://your-provider.example/v1
export L3_INSPECT_API_KEY_ENV=OPENAI_API_KEY
export OPENAI_API_KEY=...
```

For `kimi` the typical host is `https://api.moonshot.cn/v1`.

`L3_INSPECT_MODEL`, `L3_INSPECT_API_STYLE`,
`L3_INSPECT_API_VERSION`, `L3_INSPECT_REASONING_EFFORT` and timeout settings
may override the selected profile. Secrets must remain in environment
variables.

## Install and evaluate

```bash
bash policy/RoboDojo_Agent_L3_Inspect/install.sh

bash policy/RoboDojo_Agent_L3_Inspect/eval.sh \
  RoboDojo general_pickup notes-recipes arx_x5 joint 0 0 1 uv base
```

For one deterministic layout:

```bash
ROBODOJO_RUN_ID=l3-inspect-general-pickup-layout0 \
  bash policy/RoboDojo_Agent_L3_Inspect/run_fixed_layout.sh \
  0 1 general_pickup /path/to/RoboDojo-eval/.venv
```

This is evaluation-only: it has no learned checkpoint, data conversion or
training entry point.
