# SWAM — RoboTwin EEF evaluation

**Contributor:** SWAM team | **Paper:** [SWAM](https://arxiv.org/abs/2611.12870) | **Original code:** [shaaaaron/SWAM](https://github.com/shaaaaron/SWAM)

Eval-only adapter for the public **12k RoboTwin posttrain checkpoint**, trained on
50 clean demonstrations per task across 50 tasks. Supports native Aloha-AgileX,
absolute 16D EEF actions (`ee`), and one environment at a time.
Shared conventions: [XPolicyLab README](../../README.md).

## Installation

Prepare [SWAM's model environment](https://github.com/shaaaaron/SWAM/blob/7d5bc85ec086b9a7fc4bfe407b2c7e0ea1606dbc/docs/SETUP.md) and
[RoboTwin's simulator environment and assets](https://robotwin-platform.github.io/doc/usage/robotwin-install.html). Use separate
Python environments; compatible existing environments can be reused.
Use an XPolicyLab checkout **containing this adapter** at `$ROBOTWIN_ROOT/XPolicyLab`
(the PR branch or a merged revision, not the unmodified base commit).

Download the upstream model source if it is not already available:
```bash
git clone https://github.com/shaaaaron/SWAM.git
git -C SWAM checkout 7d5bc85ec086b9a7fc4bfe407b2c7e0ea1606dbc
export SWAM_ROOT="$(pwd)/SWAM/dreamzero"
export ROBOTWIN_ROOT=/path/to/RoboTwin
export POLICY_ENV=/path/to/swam-env
export ROBOTWIN_ENV=/path/to/robotwin-env
cd "$ROBOTWIN_ROOT/XPolicyLab/policy/SWAM"
PYTHON_BIN="$POLICY_ENV/bin/python" bash install.sh
PYTHON_BIN="$ROBOTWIN_ENV/bin/python" bash install.sh
```

`install.sh` installs adapter dependencies and checks the WebSocket server import;
it does not install the model/simulator stack or assets. It uses normal pip resolution,
which may change installed versions. Complete RoboTwin's asset/path setup and provide
its sibling `env_cfg/` directory before evaluation. OpenCV and FFmpeg must be available.

## Model Assets

Public checkpoint: [FreDERicia/SWAM posttrain](https://huggingface.co/FreDERicia/SWAM/tree/ac5032420608a426176ed8de5568f7c79ad322c8/posttrain).
The download script pins this revision, including `experiment_cfg/conf.yaml`.
Reuse existing files; otherwise download the checkpoint and base components:
```bash
export PATH="$POLICY_ENV/bin:$PATH"
bash download_checkpoint.sh /path/to/models/SWAM
hf download Wan-AI/Wan2.2-TI2V-5B \
  --local-dir /path/to/models/Wan2.2-TI2V-5B
hf download google/umt5-xxl --include '*token*' '*.json' 'spiece.model' \
  --local-dir /path/to/models/umt5-xxl
hf download Wan-AI/Wan2.1-I2V-14B-480P \
  models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
  --local-dir /path/to/models/Wan2.1-I2V-14B-480P
```

These commands require `huggingface_hub` and the `hf` CLI in the model environment.
Wan2.2 supplies the VAE and UMT5 text encoder; the Wan2.1 file is the CLIP image encoder.
Set the paths for evaluation:
```bash
export SWAM_MODEL_PATH=/path/to/models/SWAM/posttrain
export SWAM_WAN_PATH=/path/to/models/Wan2.2-TI2V-5B
export SWAM_TOKENIZER_PATH=/path/to/models/umt5-xxl
export SWAM_IMAGE_ENCODER_PATH=/path/to/models/Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth
```

The adapter overrides component paths in memory using the upstream loader.
It reads the checkpoint directly without rewriting files or creating symlink directories.
A complete existing local 12k checkpoint can be used with the same path settings.

## Data Processing

The XPolicyLab data-processing entry point is not included in this eval-only adapter.
No training data is required for evaluation.

## Training

Upstream SWAM training code is public. XPolicyLab-specific data and training entry
points are planned by **2026-12-30**; this eval-only arrangement must be disclosed
to maintainers when submitting the PR.

## Evaluation

Run from `policy/SWAM` after the setup above:
```bash
export ROBOTWIN_TASK_CONFIG=demo_clean
export ROBOTWIN_TEST_NUM=1
bash eval.sh RoboTwin adjust_bottle swam-posttrain aloha_agilex ee 0 0 0 "$POLICY_ENV" "$ROBOTWIN_ENV"
```

Use `demo_randomized` for randomized scenes. `ROBOTWIN_TEST_NUM` defaults to 100;
`1` is a smoke test. Each invocation evaluates one task. The three zeros are the
seed, policy GPU ID and simulator GPU ID; both processes may use the same GPU.
The client uses RoboTwin's official evaluation entry and success criteria;
results are saved under `$ROBOTWIN_ROOT/eval_result/`.

Optional interface checks without simulation:
```bash
EVAL_ENV_TYPE=debug bash eval.sh RoboTwin adjust_bottle swam-posttrain aloha_agilex ee 0 0 0 "$POLICY_ENV" "$ROBOTWIN_ENV"
EVAL_ENV_TYPE=debug DEBUG_OBS_ENCODED=1 bash eval.sh RoboTwin adjust_bottle swam-posttrain aloha_agilex ee 0 0 0 "$POLICY_ENV" "$ROBOTWIN_ENV"
```

Cold loading and startup warmup can take several minutes; the launcher waits up to
1200 seconds for readiness. Keep one evaluation client per server. Check logs and
result files for rollout errors; process exit code alone does not establish success.

## Configuration and validation

`deploy.yml` supports `swam_root`, `model_path`, `wan_path`, `tokenizer_path` and
`image_encoder_path`, corresponding to the `SWAM_*` environment variables above.
Explicit configuration values take precedence over environment variables.

- Fixed checkpoint settings: `action_dim=16`, `action_horizon=48`, `video_history=8`.
- Each arm uses `[x,y,z,qw,qx,qy,qz,gripper]`, left then right; absolute EEF targets.
- First inference uses one frame; continuation uses eight frames at stride six.
- `eval_batch=false`. Joint control, RoboDojo and real hardware are outside this adapter's scope.
- The simulator robot is selected by `embodiment: [aloha-agilex]` in the task config.
  `arx_x5` is also accepted as an action-profile alias, not as a tested simulator robot.
- Tested hardware: **one 48 GB RTX 4090**. Policy inference exceeded 24 GB;
  a standard 24 GB 4090 has not been validated.

Tested source versions (not mandatory leaderboard revisions):

| Source | Commit |
|---|---|
| SWAM | `7d5bc85ec086b9a7fc4bfe407b2c7e0ea1606dbc` |
| RoboTwin | `ea8b21121ebb3cd201ff5b3fe361944ac94eda3f` |
| XPolicyLab base, plus this adapter | `4a9e3afd73b86a48c5db076f5cf9dbfbef1a180e` |

The tested XPolicyLab base includes the EEF helpers required here; RoboTwin's
pinned submodule is older. Model environment: Python 3.11 / PyTorch 2.6.0.
Simulator: Python 3.10 / PyTorch 2.4.1 / SAPIEN 3.0.0b1 / mplib 0.2.1 /
cuRobo 0.7.8 (`d64c4b005459db10c5dd867d8b30a87d5bda9bdb`).
Existing environments passed integration smoke tests; fresh installation steps were
statically checked. On 2026-11-28, direct loading of the original local 12k checkpoint
passed GPU evaluation on `adjust_bottle`: clean 2/2, randomized 2/2.
The same server handled all four episodes, including startup warmup, continuation
and episode resets. No full 50-task leaderboard result is claimed.
