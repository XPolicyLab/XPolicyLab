# MachEmbodied-Dex1.0

**Contributor:** Li Auto | **Paper:** Pending | **arXiv:** Pending | **Original code:** https://github.com/Liuxuetao1219/MachEmbodied-Dex1.0

`MachEmbodied_Dex1_0` is the eval-only XPolicyLab adapter for the MachEmbodied-Dex1.0 RoboTwin Clean50-to-Random
checkpoint. It supports `bench_name=RoboTwin`, `env_cfg_type=arx_x5`, and
`action_type=joint`.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, and
`EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results:
[RoboTwin Leaderboard](https://robotwin-platform.github.io/leaderboard).

## Installation

```bash
conda activate <policy_env>
cd XPolicyLab/policy/MachEmbodied_Dex1_0
bash install.sh
```

Install RoboTwin separately in `<robotwin_env>` following its official instructions.

## Data Processing

Unsupported in this eval-only submission.

## Training

Unsupported in this eval-only submission (release ETA: TBD).

## Evaluation

```bash
CHECKPOINT_DIR=checkpoints/MachEmbodied-Dex1.0-RoboTwin-Clean2Random-Leaderboard

hf download liuxuetao/MachEmbodied-Dex1.0-RoboTwin-Clean2Random-Leaderboard \
  --local-dir "${CHECKPOINT_DIR}"

hf download Wan-AI/Wan2.2-TI2V-5B \
  config.json \
  Wan2.2_VAE.pth \
  models_t5_umt5-xxl-enc-bf16.pth \
  google/umt5-xxl/special_tokens_map.json \
  google/umt5-xxl/spiece.model \
  google/umt5-xxl/tokenizer.json \
  google/umt5-xxl/tokenizer_config.json \
  --local-dir "${CHECKPOINT_DIR}/wan"
```

The MachEmbodied-Dex repository provides `model.pt`, `tactile_ae.pt`, `model_config.json`, and
`manifest.json`; the second command downloads the required Wan2.2 assets.

```bash
cd XPolicyLab/policy/MachEmbodied_Dex1_0
bash eval.sh RoboTwin adjust_bottle \
  MachEmbodied-Dex1.0-RoboTwin-Clean2Random-Leaderboard \
  arx_x5 joint 42 0 0 <policy_env> <robotwin_env>
```

`<policy_env>` runs MachEmbodied-Dex1.0; `<robotwin_env>` runs RoboTwin/SAPIEN.

## Configuration

This BGR-trained checkpoint sets `input_color_order: bgr`, applying one RGB-to-BGR conversion
before inference. Images are scaled to `[0, 1]` without mean/std normalization.

Clean50 training included collected three-axis tactile force arrays. RoboTwin evaluation has no
tactile observation, so both tactile input frames are zero.

The model returns 16 joint targets at t+3, t+6, ..., t+48; XPolicyLab submits them sequentially.
