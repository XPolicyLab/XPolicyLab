# aha-wam RoboDojo Adapter

This policy adapts the locally trained aha-wam checkpoint for XPolicyLab RoboDojo evaluation.

Default artifacts:

- checkpoint: `/mnt/petrelfs/caijisong/XPolicyLab/checkpoint/step_002500.pt`
- dataset stats: `/mnt/petrelfs/caijisong/XPolicyLab/checkpoint/dataset_stats.json`
- AHAWAM project: `policy/AHA_WAM/AHAWAM`
- base model cache: `/mnt/petrelfs/caijisong/dualWAM/checkpoints`
- env cfg root: `/mnt/petrelfs/caijisong/env_cfg` (`AHA_WAM_ENV_CFG_ROOT`)

The model was trained with `configs/task/robodojo_local_history_updated_kv_prior_only_16.yaml`, `action_type=joint`, and 14-D qpos actions ordered as `[left_arm, left_ee, right_arm, right_ee]`.
During evaluation, the default replanning cadence is one video DiT forward followed by two action DiT forwards, then another video DiT forward. Override with `AHA_WAM_CHUNKS_PER_VIDEO_PREFILL`.

Policy server example:

```bash
cd /mnt/petrelfs/caijisong/XPolicyLab/policy/AHA_WAM
bash setup_eval_policy_server.sh RoboDojo stack_bowls local_aha_wam arx_x5 3500 joint 0 0 wam 12345 localhost
```

Full debug flow:

```bash
bash eval.sh RoboDojo stack_bowls local_aha_wam arx_x5 3500 joint 0 0 0 wam wam
```

Environment setup:

```bash
conda create -n ahawam python=3.10 -y
conda activate ahawam
pip install -U pip
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
cd /mnt/petrelfs/caijisong/XPolicyLab/policy/AHA_WAM
bash install.sh
export DIFFSYNTH_MODEL_BASE_PATH=/mnt/petrelfs/caijisong/dualWAM/checkpoints
```

On this cluster, an existing `wam` conda environment can be used instead of
creating `ahawam`.

Training wrapper:

```bash
bash train.sh RoboDojo cotrain arx_x5 3500 joint 0 0,1,2,3,4,5,6,7
```

`train.sh` launches the local `AHAWAM` project with `task=robodojo_local_history_updated_kv_prior_only_16` and `model=ahawam` only. Override dataset, output, resume, and seed with `AHA_WAM_TRAIN_DATASET_DIR`, `AHA_WAM_OUTPUT_ROOT`, `AHA_WAM_INIT_CHECKPOINT`, `AHA_WAM_RESUME`, and `AHA_WAM_TRAIN_SEED`. If the XPolicyLab seed is `0`, the wrapper uses training seed `1` because the upstream AHAWAM seeding helper requires a positive uint32 seed.

One-step training smoke test:

```bash
AHA_WAM_MAX_STEPS=1 \
AHA_WAM_NUM_EPOCHS=1 \
AHA_WAM_BATCH_SIZE=1 \
AHA_WAM_GRADIENT_ACCUMULATION_STEPS=1 \
AHA_WAM_NUM_WORKERS=0 \
AHA_WAM_WANDB_ENABLED=false \
AHA_WAM_OUTPUT_ROOT=/tmp/aha_wam_smoke \
bash train.sh RoboDojo cotrain arx_x5 3500 joint 0 0,1,2,3,4,5,6,7 8
```

The Wan2.2/AHA-WAM training graph is large. With ZeRO-1, use 7-8 80GB GPUs for
a reliable smoke or full run; single-GPU and 2-GPU smoke tests can OOM during the
Adam optimizer step.

For simulator evaluation, set `eval_env: sim` in `deploy.yml` and run from a workspace that provides `scripts/eval_policy.sh` and the RoboDojo simulator environment.
