# AHA-WAM Local Training Project

This directory vendors the minimal AHA-WAM training code used by the
`policy/AHA_WAM` XPolicyLab adapter. It is intentionally scoped to the
RoboDojo prior-only chunk-local setup and omits teacher, ODE, causal, and
baseline model variants.

## Layout

- `configs/train.yaml`: shared Hydra training defaults.
- `configs/data/robodojo.yaml`: RoboDojo LeRobot v2.1 video dataset paths,
  observation/action metadata, normalization stats, and text embedding cache.
- `configs/model/ahawam.yaml`: AHA-WAM model, ActionDiT, scheduler, and loss
  defaults.
- `configs/task/robodojo_local_history_updated_kv_prior_only_16.yaml`: the
  training task matching the prepared RoboDojo data and 14-D joint/qpos action
  format.
- `scripts/train.py` and `scripts/train_zero1.sh`: Hydra and Accelerate
  launchers.
- `src/ahawam`: package code for datasets, Wan2.2/AHA-WAM modules, and trainer.

## Training

### Environment

The training package follows the upstream AHA-WAM release environment:

```bash
conda create -n ahawam python=3.10 -y
conda activate ahawam
pip install -U pip
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
cd XPolicyLab/policy/AHA_WAM
bash install.sh
export DIFFSYNTH_MODEL_BASE_PATH=/path/to/dualWAM/checkpoints
```

On this cluster, the preconfigured `wam` environment can be used directly.

### Full Run

Run through the outer policy wrapper so XPolicyLab arguments, dataset overrides,
base-model cache paths, and output directories are handled consistently:

```bash
cd XPolicyLab/policy/AHA_WAM
export AHA_WAM_TRAIN_DATASET_DIR=/path/to/RoboDojo_lerobot_v21_video
export DIFFSYNTH_MODEL_BASE_PATH=/path/to/dualWAM/checkpoints
bash train.sh RoboDojo cotrain arx_x5 joint 0 0,1,2,3,4,5,6,7
```

The dataset must be provided with `AHA_WAM_TRAIN_DATASET_DIR`; it should contain
`meta/`, `dataset_stats.json`, and text embeddings under `text_embeds_cache`
unless the paths below override those defaults.

Useful overrides:

```bash
export AHA_WAM_TRAIN_DATASET_DIR=/path/to/RoboDojo_lerobot_v21_video
export AHA_WAM_OUTPUT_ROOT=/path/to/checkpoints
export AHA_WAM_INIT_CHECKPOINT=/path/to/step_xxxxxx.pt
export AHA_WAM_TRAIN_SEED=1
export DIFFSYNTH_MODEL_BASE_PATH=/path/to/dualWAM/checkpoints
```

The wrapper maps seed `0` to training seed `1`, matching the upstream AHAWAM
requirement that seeds are positive `uint32` values.

### Smoke Test

Use one optimizer step to validate package imports, Hydra config, dataset access,
and checkpoint/cache paths without launching a full run:

```bash
cd XPolicyLab/policy/AHA_WAM
AHA_WAM_MAX_STEPS=1 \
AHA_WAM_NUM_EPOCHS=1 \
AHA_WAM_BATCH_SIZE=1 \
AHA_WAM_GRADIENT_ACCUMULATION_STEPS=1 \
AHA_WAM_NUM_WORKERS=0 \
AHA_WAM_WANDB_ENABLED=false \
AHA_WAM_OUTPUT_ROOT=/tmp/aha_wam_smoke \
bash train.sh RoboDojo cotrain arx_x5 joint 0 0,1,2,3,4,5,6,7 8
```

With the retained ZeRO-1 launcher, use 7-8 80GB GPUs for this smoke test.
Single-GPU and 2-GPU runs can reach data/model initialization but OOM during the
Adam optimizer step.
