# Bagel-Robotics-1-Preview

**Contributor:** Bagel Labs | **Base model:** [VPP2](https://github.com/roboterax/video-prediction-policy-2) | **Training source:** release pending

Bagel-Robotics-1-Preview is an eval-only adapter for our proposed pruning method
for world-action models (WAMs), targeting their video expert. We have evaluated
the approach on VPP2 so far. This preview checkpoint has 6,003,498,177 fewer
video-expert parameters than its VPP2 base checkpoint; the Action2B expert is
unchanged. This directory vendors the inference runtime and compact checkpoint
loader so the existing `policy/VPP2` adapter remains untouched.

The evaluated controller predicts 32 EE16 actions, executes 16, uses 10 Euler
steps with shift 1, and blends overlapping actions with 90% weight on the new
chunk and 10% on the prior chunk. The full training source, simulator, and checkpoints
are distributed separately. Parameter reduction and commanded EE similarity
do not establish physical task success.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

Run inside a RoboDojo workspace with `env_cfg/` next to the XPolicyLab checkout.
Use a compatible CUDA/PyTorch wheel index if your system needs one:

```bash
cd XPolicyLab/policy/Bagel-Robotics-1-Preview
bash install.sh "$PWD/.venv"
# Optional: TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 bash install.sh "$PWD/.venv"
```

The script creates a policy virtual environment and installs the vendored
inference runtime and XPolicyLab. The simulator environment is installed
separately using RoboDojo's instructions.

## Data Processing

No dataset conversion is needed for this eval-only adapter. The recovery
trainer uses a separate RoboDojo EE16 LeRobot v3 export; `process_data.sh` is
intentionally absent. At inference, the adapter receives decoded RGB from
XPolicyLab's shared server.

## Training

Training and pruning are not performed in XPolicyLab. A separate research
repository contains their implementation; `train.sh` is intentionally
absent. The training source is targeted for public release by **November 12,
2026**; its repository URL will be added when available.

## Evaluation

The standard XPolicyLab ten-argument command is:

```bash
bash eval.sh RoboDojo <task_name> bagel_robotics_1_preview arx_x5 ee 1 \
  <policy_gpu_id> <sim_gpu_id> "$PWD/.venv" <eval_env_conda_env>
```

Example with a local checkpoint and separate policy/simulator GPUs:

```bash
export BAGEL_ROBOTICS_1_BUNDLE=/absolute/path/to/recovered_bundle
export BAGEL_ROBOTICS_1_WAN_ROOT=/absolute/path/to/Wan2.1-I2V-14B-480P
EVAL_ENV_TYPE=sim bash eval.sh RoboDojo insert_tubes bagel_robotics_1_preview \
  arx_x5 ee 1 0 1 "$PWD/.venv" RoboDojo
```

Use `EVAL_ENV_TYPE=debug` for a wiring check without Isaac Sim. A Python venv
path in the final argument runs the debug client directly; set
`DEBUG_EVAL_EPISODES=1` for a single-episode smoke check. For
split-machine deployment, use `setup_eval_policy_server.sh` and
`setup_eval_env_client.sh` with the standard XPolicyLab argument order.

The adapter maintains observation history separately per client and records
each executed action step. Batch evaluation is disabled. Seed 1 is the setting
used in the current Bagel-Robotics-1-Preview experiments; other seeds and the final public
RoboDojo result remain to be reported.

## Model Assets

The recovered compact bundle contains:

```text
checkpoints/bagel_robotics_1_preview/
  action.pt
  video.pt
  dataset_stats.json
  manifest.json
  pruning_architecture.json
  recovery_metadata.json
```

The bundle must report exactly 6,003,498,177 removed video parameters and
checkpoint step 100000. The Wan2.1 VAE, CLIP, UMT5 encoder, and tokenizer are
also needed under `checkpoints/Wan2.1-I2V-14B-480P/`. The recovered bundle is
at [bageldotcom/Bagel-Robotics-1-Preview](https://huggingface.co/bageldotcom/Bagel-Robotics-1-Preview).
The download script pins the verified model revision:

```bash
bash download_checkpoint.sh
.venv/bin/python launch_policy.py --dry-run
```

When using a custom policy environment, set `PYTHON_BIN` to its Python before
calling `download_checkpoint.sh`.

For a local bundle, skip the download and set `BAGEL_ROBOTICS_1_BUNDLE` and
`BAGEL_ROBOTICS_1_WAN_ROOT` to absolute paths. You can validate those files with:

```bash
.venv/bin/python download_checkpoint.py --verify-only \
  --bundle "$BAGEL_ROBOTICS_1_BUNDLE" --wan-root "$BAGEL_ROBOTICS_1_WAN_ROOT"
```

The download script uses the official VPP2 Hugging Face repository for shared
Wan encoder assets.

## Limitations

This is an inference-only submission. The compact checkpoint is uploaded.
