# ReStart on RoboTwin 2.0

- **Contributor** 黄伟达 (ReStart)
- **Paper** [ReStart project and method description](https://github.com/ai-for-data/ReStart/tree/c4e3e8c25e5468df0d5b7c64d1c63787168f838c)
- **arXiv** —
- **Original code** [ai-for-data/ReStart](https://github.com/ai-for-data/ReStart)

ReStart inverts actions from official clean demonstrations into a context
bank. A frozen TabICLv2 model predicts the starting action noise for frozen
UniWAM-base. The inherited [UniWAM adapter](../UniWAM) owns images, measured
14-D joint and gripper feedback, action chunks and resets. The policy retains
its history center, four denoising steps and its original 0.02 noise scale.

## Shared conventions

See the [XPolicyLab README](../../README.md) for the shared policy interface
and the [RoboTwin documentation](https://robotwin-platform.github.io/doc/usage/xpolicylab.html)
for evaluation.

## Installation

Use the [UniWAM setup](../UniWAM/README.md#installation), the
[ReStart reproduction guide](https://github.com/ai-for-data/ReStart/blob/c4e3e8c25e5468df0d5b7c64d1c63787168f838c/docs/reproduction.md)
and the [RoboTwin installation](https://robotwin-platform.github.io/doc/usage/robotwin-install.html).
Use RoboTwin score-run revision `13c3c47ff4312dd62484bcd51be034af55c062d1`.
Place the
[official robot registry](https://github.com/RoboTwin-Platform/RoboTwin/tree/ea8b21121ebb3cd201ff5b3fe361944ac94eda3f/env_cfg)
beside the XPolicyLab checkout, and select `aloha_agilex`.

```sh
bash policy/ReStart/install.sh uniwam
bash policy/ReStart/download_checkpoints.sh /absolute/path/to/checkpoints
```

The downloader verifies the UniWAM-base weights, author RoboTwin inference
metadata and TabICLv2 weights. Set `UNIWAM_WAN_PATH`, `UNIWAM_VLM_PATH` and
`UNIWAM_ROBOTWIN_ROOT` for the local assets and simulator checkout.

## Data Processing

The reviewer package contains a ready-made `lift_pot` example and a setup
script that verifies the inputs and generates the task manifest. For a new
task use ReStart's existing [offline context builder](https://github.com/ai-for-data/ReStart/blob/c4e3e8c25e5468df0d5b7c64d1c63787168f838c/docs/reproduction.md#data-and-labels)
with the complete set of native replans from its 50 official clean expert
trajectories. The same clean input corpus is used for both evaluation
conditions. The existing reconstruction-MSE admission rule is applied to the
whole bank.

Set `RESTART_CONTEXT_DIR` to a directory of `<task_name>.json` manifests.
Each manifest supplies `embodpfn_support`, `embodpfn_inverse_reference`,
`embodpfn_policy_receipt`, `embodpfn_pfn_checkpoint`, `embodpfn_pfn_python`
and `embodpfn_pfn_device`. Use `embodpfn_pfn_devices` for the existing
multi-device execution. Path values may be relative to the manifest.
`RESTART_METHOD_ARGS` remains available for an explicit single-task manifest.
The server prepares its context before it accepts requests.

## Training

ReStart fits TabICLv2's context from the supplied demonstrations at evaluation
time. Both model checkpoints remain frozen.

## Evaluation

Use the upstream 10-argument `eval.sh` interface. The supported combination
is RoboTwin, `aloha_agilex`, joint actions and seed 42. The reference
allocation is one host with four 80 GB GPUs.

```sh
EVAL_ENV_TYPE=debug bash policy/ReStart/eval.sh \
  RoboTwin lift_pot "$CHECKPOINT" aloha_agilex joint 42 \
  0,1,2,3 0 "$POLICY_ENV" "$ROBOTWIN_ENV"

EVAL_ENV_TYPE=debug DEBUG_OBS_ENCODED=1 bash policy/ReStart/eval.sh \
  RoboTwin lift_pot "$CHECKPOINT" aloha_agilex joint 42 \
  0,1,2,3 0 "$POLICY_ENV" "$ROBOTWIN_ENV"

EVAL_ENV_TYPE=sim UNIWAM_ROBOTWIN_TASK_CONFIG=demo_clean bash policy/ReStart/eval.sh \
  RoboTwin lift_pot "$CHECKPOINT" aloha_agilex joint 42 \
  0,1,2,3 0 "$POLICY_ENV" "$ROBOTWIN_ENV"
```

The simulator uses 100 episodes, unseen instructions and video recording. To evaluate
randomized scenes use `demo_randomized`. The WebSocket ping interval and
timeout use UniWAM's 20/120 second settings.
