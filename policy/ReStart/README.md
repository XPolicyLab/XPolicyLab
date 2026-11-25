# ReStart

**Contributor:** Inpiora | **Paper:** — | **arXiv:** — | **Original code:** [ai-for-data/ReStart](https://github.com/ai-for-data/ReStart)

ReStart adapts frozen UniWAM-base at inference time using official clean demonstrations and TabICLv2. This adapter supports joint control on Aloha-AgileX in RoboTwin 2.0 and uses UniWAM's rollout interface.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

For the validated `lift_pot` run, start with the README in the review kit. It
pins the sources and prepares the checkpoint, simulator assets and context.
The reference run used Linux and four 80 GB NVIDIA GPUs on one host.

For an existing workspace, follow the [UniWAM setup](../UniWAM/README.md#installation)
and the [official RoboTwin installation](https://robotwin-platform.github.io/doc/usage/robotwin-install.html).
The policy and simulator use separate Conda environments in the launcher,
and TabICLv2 uses its own Python environment. The adapter's installer delegates
to UniWAM. From the XPolicyLab root, with the Hugging Face CLI installed, run:

```bash
bash policy/ReStart/install.sh uniwam
bash policy/ReStart/download_checkpoints.sh /path/to/models
```

Set `UNIWAM_WAN_PATH` and `UNIWAM_VLM_PATH` to the local Wan and Qwen
assets. Set `UNIWAM_ROBOTWIN_ROOT` to the simulator checkout. The reference
uses RoboTwin revision `13c3c47ff4312dd62484bcd51be034af55c062d1`
and the [official Aloha-AgileX registry](https://github.com/RoboTwin-Platform/RoboTwin/tree/ea8b21121ebb3cd201ff5b3fe361944ac94eda3f/env_cfg)
in the sibling `env_cfg/` directory.

## Data Processing

Use the review kit's prepared `lift_pot` context. For another task, follow
the [context preparation guide](https://github.com/ai-for-data/ReStart/blob/c4e3e8c25e5468df0d5b7c64d1c63787168f838c/docs/reproduction.md#data-and-labels)
with the task's 50 official clean demonstrations and all native replanning
windows. Use the same clean context for both scene conditions.

Set `RESTART_CONTEXT_DIR` to the directory containing `<task_name>.json`.
The review kit generates an example manifest with the context, checkpoint,
and TabICLv2 environment. Paths in the manifest can be relative to that file.

## Training

This is an evaluation-only adapter. It builds context from demonstrations
and keeps both model checkpoints frozen.

## Evaluation

From the XPolicyLab root, set `RESTART_CHECKPOINT` to the downloaded
`UniWAM-base/` directory. Set `POLICY_ENV` and `ROBOTWIN_ENV` to the
prepared policy and simulator environments. The review kit sets the checkpoint
and context paths in `replay.env.sh`.

```bash
EVAL_ENV_TYPE=debug bash policy/ReStart/eval.sh \
  RoboTwin lift_pot "$RESTART_CHECKPOINT" aloha_agilex joint 42 \
  0,1,2,3 0 "$POLICY_ENV" "$ROBOTWIN_ENV"

EVAL_ENV_TYPE=debug DEBUG_OBS_ENCODED=1 bash policy/ReStart/eval.sh \
  RoboTwin lift_pot "$RESTART_CHECKPOINT" aloha_agilex joint 42 \
  0,1,2,3 0 "$POLICY_ENV" "$ROBOTWIN_ENV"

EVAL_ENV_TYPE=sim UNIWAM_ROBOTWIN_TASK_CONFIG=demo_clean \
  bash policy/ReStart/eval.sh \
  RoboTwin lift_pot "$RESTART_CHECKPOINT" aloha_agilex joint 42 \
  0,1,2,3 0 "$POLICY_ENV" "$ROBOTWIN_ENV"
```

Both debug runs end with `[MAIN] eval finished`. The simulator runs 100
episodes with unseen instructions. The validated `lift_pot / demo_clean`
run recorded **99/100** successes.
