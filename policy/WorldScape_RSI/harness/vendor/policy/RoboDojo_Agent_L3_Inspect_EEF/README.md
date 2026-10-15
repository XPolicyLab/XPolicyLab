# RoboDojo_Agent_L3_Inspect_EEF

Main RGB-only L3 reference harness. The model calls `move_eef` with absolute
world-frame grasp-point targets; RoboDojo's non-learned motion planner converts
those poses into native joint actions.

## Disclosure

- **Level:** L3 LLM-as-Policy
- **Status:** Main reference; full RoboDojo results published
- **Models:** named planner profiles for GPT-6 Astra, GPT-5.5 and Kimi K3;
  provider endpoint and credentials are runtime configuration
- **Prompt:** shared Inspect conversation in
  [`../RoboDojo_Agent_L3_Inspect/policy.py`](../RoboDojo_Agent_L3_Inspect/policy.py),
  EEF semantics in [`docs.py`](docs.py), optional task recipes
- **Tools:** `move_eef`, `give_up`
- **Motion stack:** Cartesian grasp-point targets converted to flange poses,
  CuRobo planning, then native joint playback
- **Memory:** configurable image horizon or complete image history
- **Default budget:** 170 LLM calls
- **Reference difference:** this is the baseline reference

## Model-facing action

`move_eef` exposes 14 named dimensions, seven per arm:

```text
<arm>_x, <arm>_y, <arm>_z,
<arm>_pitch_deg, <arm>_roll_deg, <arm>_yaw_deg,
<arm>_gripper
```

Positions name the center of the gripping face, not the robot flange. Unnamed
dimensions keep their observed values. Angles are measured from a straight
down grasp; their exact convention and bounds are generated from
[`pose.py`](pose.py).

## Observation boundary

The model receives three RGB cameras, the official instruction and measured
grasp-point/proprioceptive state. It receives no depth, object pose, layout
metadata or reward internals. RoboDojo is the only scorer.

## Provider configuration

`L3_INSPECT_BASE_URL` is required (no default host).

```bash
export L3_INSPECT_PLANNER=astra
export L3_INSPECT_BASE_URL=https://your-provider.example/v1
export L3_INSPECT_API_KEY_ENV=OPENAI_API_KEY
export OPENAI_API_KEY=...
```

See the joint reference README for model/API overrides.

## Install and evaluate

```bash
bash policy/RoboDojo_Agent_L3_Inspect_EEF/install.sh

bash policy/RoboDojo_Agent_L3_Inspect_EEF/eval.sh \
  RoboDojo general_pickup notes-recipes arx_x5 joint 0 0 1 uv base
```

For one deterministic layout:

```bash
ROBODOJO_RUN_ID=l3-inspect-eef-general-pickup-layout0 \
  bash policy/RoboDojo_Agent_L3_Inspect_EEF/run_fixed_layout.sh \
  0 1 general_pickup /path/to/RoboDojo-eval/.venv
```

Published summaries for GPT-6 Astra and GPT-5.5 are under
[`results/l3_inspect_eef_official_2100`](../../results/l3_inspect_eef_official_2100/).

This is evaluation-only: it has no learned checkpoint, data conversion or
training entry point.
