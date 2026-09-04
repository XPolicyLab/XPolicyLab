# KinRT for RoboDojo

**Contributor:** Tianhang Yang, Yanze Zheng, Junjie Wang, Wei-Bin Kou, Ruotong Li, Yujiu Yang | **Paper:** Route by Kinematics, Act by Observation | **arXiv:** https://arxiv.org/abs/2607.26807 | **Original code:** https://github.com/gleeacast/KinRT

This adapter applies KinRT to RoboDojo's dual-ARX-X5 environment. It supports `bench_name=RoboDojo`, `env_cfg_type=arx_x5`, and `action_type=joint`. The KinRT source remains in a separate checkout; this directory contains only the XPolicyLab integration.

Shared conventions are documented in the [XPolicyLab README](../../README.md). Official results are published on the [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Interface

The adapter preserves RoboDojo's native state and camera semantics:

| RoboDojo field | KinRT input |
| --- | --- |
| `cam_head` | `cam_high` |
| `cam_left_wrist` | `cam_left_wrist` |
| `cam_right_wrist` | `cam_right_wrist` |
| left arm, left gripper, right arm, right gripper | 14-D state/action vector |

Images remain RGB throughout the conversion, training, and evaluation paths. The policy server decodes runtime image buffers before `model.py` receives them.

KinRT creates four offline routing classes by clustering 50-step future action chunks and their temporal velocities. Training uses the resulting label at each global LeRobot frame index to supervise observation-conditioned top-1 expert routing. LoRA and full fine-tuning use the same router, labels, losses, and data; they differ only in which base parameters are trainable.

## Installation

Use sibling RoboDojo and KinRT checkouts:

```text
<workspace>/
├── RoboDojo/
│   └── XPolicyLab/policy/KinRT/
└── KinRT_RoboDojo/
    └── policy/pi05/
```

Then install the KinRT OpenPI environment:

```bash
cd <workspace>/RoboDojo/XPolicyLab/policy/KinRT
bash install.sh <workspace>/KinRT_RoboDojo/policy/pi05
```

The explicit path is optional when the sibling layout above is used. If the sibling checkout is absent, `install.sh` clones KinRT and checks out commit `108368422539b79d2be2c596750de1fab1cfd8fd`. It then installs KinRT, XPolicyLab, and the label-generation dependencies into the KinRT OpenPI uv environment.

## Data Processing

Download or place RoboDojo demonstrations under the standard source tree before conversion:

```text
RoboDojo/data/RoboDojo/<task>/arx_x5/data/episode_*.hdf5
```

Set a persistent LeRobot cache and convert one or more tasks:

```bash
export HF_LEROBOT_HOME=<workspace>/cache/huggingface/lerobot

# One task
bash process_data.sh RoboDojo stack_bowls arx_x5 joint

# Multiple tasks in one KinRT dataset
bash process_data.sh RoboDojo multitask arx_x5 joint \
  stack_bowls,insert_key,hang_mugs
```

The output repo id defaults to `RoboDojo-KinRT-arx_x5-joint`. Set `KINRT_ROBODOJO_REPO_ID` to change it. Existing output is protected; set `KINRT_OVERWRITE_DATASET=1` only when replacement is intentional.

For a pipeline smoke test, download the small official RoboDojo demo bundle and point the converter at its exact embodiment directory. The current bundle contains 20 `stack_bowls` episodes and is intended for integration checks, not benchmark training or reporting.

```bash
cd <workspace>/RoboDojo/XPolicyLab
bash scripts/RoboDojo/download_robodojo_data.sh demo

cd policy/KinRT
export KINRT_ROBODOJO_SOURCE_DIR=<workspace>/RoboDojo/data/demo/arx_x5
export KINRT_ROBODOJO_REPO_ID=RoboDojo-KinRT-demo-arx_x5-joint
bash process_data.sh RoboDojo demo arx_x5 joint
```

RoboDojo observations are collected at 25 Hz. The converter defaults to 50 FPS in LeRobot metadata because the official XPolicyLab Pi 0.5 adapter uses that convention. This preserves baseline comparability, although each 50-step action chunk spans two seconds of source motion. Set `KINRT_LEROBOT_METADATA_FPS=25` only for a separately named experiment and do not compare it directly with the 50-FPS baseline.

The action target is the next recorded joint state; the final frame repeats its own state. Generate the four KinRT router classes after conversion:

```bash
bash generate_router_labels.sh
```

The script writes labels and clustering metadata to:

```text
${HF_LEROBOT_HOME}/RoboDojo-KinRT-arx_x5-joint/meta/router_labels_k4/
```

Compute normalization statistics for the training parameterization:

```bash
bash compute_norm_stats.sh kinrt_lora_robodojo
# or
bash compute_norm_stats.sh kinrt_full_robodojo
```

## Training

LoRA is the default parameterization:

```bash
OPENPI_TRAIN_CONFIG_NAME=kinrt_lora_robodojo \
  bash train.sh RoboDojo multitask arx_x5 joint 0 0,1,2,3
```

Full fine-tuning changes only the trainable parameter set:

```bash
OPENPI_TRAIN_CONFIG_NAME=kinrt_full_robodojo \
  bash train.sh RoboDojo multitask arx_x5 joint 0 0,1,2,3
```

Both KinRT configurations use four experts, top-1 dense routing, a supervised routing coefficient of `0.05`, balanced-sampling alpha `0.5`, action horizon `50`, and batch size `32`. The current loader does not consume the retained `router_sampling_mix_beta` compatibility field, so it is not part of the effective sampling policy.

The configuration registry defaults to 60,000 steps for future full-budget runs. The first RoboDojo submission candidate is a controlled 10,000-step run for both KinRT and the matched no-MoE baseline:

```bash
export KINRT_ROBODOJO_REPO_ID=RoboDojo-KinRT-stack_bowls100-arx_x5-joint
export OPENPI_NUM_TRAIN_STEPS=10000
export OPENPI_BATCH_SIZE=32
export OPENPI_NUM_WORKERS=8
export OPENPI_SAVE_INTERVAL=1000
export OPENPI_WANDB_ENABLED=0

# KinRT: four experts, Top-1 routing.
OPENPI_TRAIN_CONFIG_NAME=kinrt_lora_robodojo \
  bash train.sh RoboDojo stack_bowls100_lora_10k arx_x5 joint 0 0,1,2,3

# Matched pi0.5 baseline: no MoE and no router labels.
OPENPI_TRAIN_CONFIG_NAME=pi05_lora_robodojo \
  bash train.sh RoboDojo stack_bowls100_pi05_lora_10k arx_x5 joint 0 0,1,2,3
```

Checkpoints use the XPolicyLab layout:

```text
checkpoints/RoboDojo-<ckpt_name>-arx_x5-joint-<seed>/<step>/
```

The exact KinRT candidate resolves to `checkpoints/RoboDojo-stack_bowls100_lora_10k-arx_x5-joint-0/10000/`. Its normalization key is `RoboDojo-KinRT-stack_bowls100-arx_x5-joint`.

Set `KINRT_RESUME=1` to resume an existing run. Without it, the training entry point requests an overwrite and OpenPI performs its normal safety checks.

### Preliminary pipeline check

The following small run verifies data loading, KinRT routing supervision, optimization, checkpoint saving, and checkpoint restoration. It is not a RoboDojo benchmark run.

```bash
export HF_LEROBOT_HOME=<workspace>/cache/huggingface/lerobot
export KINRT_ROBODOJO_REPO_ID=RoboDojo-KinRT-demo-arx_x5-joint
export OPENPI_TRAIN_CONFIG_NAME=kinrt_lora_robodojo
export OPENPI_BASE_CHECKPOINT=<path-to-pi05-base-params>
export OPENPI_NUM_TRAIN_STEPS=100
export OPENPI_BATCH_SIZE=8
export OPENPI_NUM_WORKERS=0
export OPENPI_SAVE_INTERVAL=100
export OPENPI_WANDB_ENABLED=0

bash train.sh RoboDojo preliminary_demo_lora arx_x5 joint 0 0,1,2,3
```

`OPENPI_NUM_WORKERS=0` is useful when datasets reside on a shared filesystem with high process-start latency. It changes only input loading and does not change the model or loss definition.

## Model Assets

Download the exact 10,000-step KinRT and matched pi0.5 artifacts from Hugging Face and verify every packaged file:

```bash
cd <workspace>/RoboDojo/XPolicyLab/policy/KinRT
bash download_checkpoint.sh
```

The default destination is `checkpoints/KinRT-RoboDojo-10k/`. Use `KINRT_CHECKPOINT_PATH=checkpoints/KinRT-RoboDojo-10k/kinrt_10k` with `kinrt_lora_robodojo`, or select `pi05_baseline_10k` with `pi05_lora_robodojo`. Set `KINRT_HF_REPO_ID` or `KINRT_HF_REVISION` only when reproducing a mirror or pinned release revision.

## Evaluation

Select the configuration that matches the checkpoint and run the standard XPolicyLab entry point:

```bash
cd <workspace>/RoboDojo/XPolicyLab/policy/KinRT

KINRT_TRAIN_CONFIG_NAME=kinrt_lora_robodojo \
KINRT_REPO_ID=RoboDojo-KinRT-stack_bowls100-arx_x5-joint \
KINRT_CHECKPOINT_NUM=10000 \
  bash eval.sh RoboDojo stack_bowls stack_bowls100_lora_10k arx_x5 joint 0 \
  0 0 <workspace>/KinRT_RoboDojo/policy/pi05 <robodojo_conda_env>
```

For an explicit checkpoint path, set `KINRT_CHECKPOINT_PATH`. `KINRT_CHECKPOINT_NUM` selects a saved step, and `KINRT_ACTION_CHUNK_SIZE` truncates each 50-step prediction before the next inference call. Keep the default value of `50` for direct comparison with the existing Pi 0.5 adapter; smaller values increase closed-loop replanning frequency and inference cost.

Run the simulator-free transport and schema check with the same checkpoint:

```bash
EVAL_ENV_TYPE=debug KINRT_TRAIN_CONFIG_NAME=kinrt_lora_robodojo \
KINRT_REPO_ID=RoboDojo-KinRT-stack_bowls100-arx_x5-joint \
KINRT_CHECKPOINT_NUM=10000 \
  bash eval.sh RoboDojo stack_bowls stack_bowls100_lora_10k arx_x5 joint 0 \
  0 0 <workspace>/KinRT_RoboDojo/policy/pi05 base
```

Repeat with `DEBUG_OBS_ENCODED=1` to exercise the server-side image decoding path.

Reload a saved checkpoint and infer one action chunk without starting the simulator. By default, the smoke test uses deterministic synthetic camera and state inputs, so it validates a packaged checkpoint without requiring the training dataset:

```bash
export PYTHONPATH=<workspace>/RoboDojo:<workspace>/KinRT_RoboDojo/policy/pi05/src
python XPolicyLab/policy/KinRT/offline_smoke.py \
  --checkpoint-root XPolicyLab/policy/KinRT/checkpoints/RoboDojo-stack_bowls100_lora_10k-arx_x5-joint-0/10000 \
  --checkpoint-step 10000 \
  --repo-id RoboDojo-KinRT-stack_bowls100-arx_x5-joint \
  --train-config-name kinrt_lora_robodojo
```

Pass `--dataset-root <LeRobot-directory>` to use an actual dataset sample instead. The check requires three RGB camera tensors, a 14-D state, a non-empty instruction, and a finite `50 x 14` action output. Passing this check confirms the offline model path only; it does not measure closed-loop task success.

## Configuration

| Variable | Purpose |
| --- | --- |
| `KINRT_OPENPI_ROOT` | KinRT `policy/pi05` path used by install, conversion, labels, stats, and training. |
| `KINRT_SOURCE_REPO` / `KINRT_SOURCE_REV` | Source repository and pinned revision used by `install.sh`. |
| `KINRT_ALLOW_UNPINNED_SOURCE` | Development-only opt-out from the source revision check. |
| `KINRT_PYTHON_BIN` | Optional Python executable used with `KINRT_OPENPI_ROOT`; intended for isolated smoke tests. |
| `KINRT_ROBODOJO_REPO_ID` | LeRobot repo id; defaults to `RoboDojo-KinRT-arx_x5-joint`. |
| `KINRT_ROBODOJO_SOURCE_DIR` | Exact embodiment directory for a small or externally prepared HDF5 bundle. |
| `KINRT_IMAGE_WRITER_PROCESSES` | Image-writer process count used during conversion; defaults to `0`. |
| `KINRT_IMAGE_WRITER_THREADS` | Image-writer thread count used during conversion; defaults to `4`. |
| `KINRT_ROBODOJO_ROUTER_LABELS_PATH` | Explicit `router_labels.npy` path. |
| `OPENPI_TRAIN_CONFIG_NAME` | Training configuration: `kinrt_lora_robodojo` or `kinrt_full_robodojo`. |
| `KINRT_TRAIN_CONFIG_NAME` | Evaluation configuration matching the checkpoint. |
| `OPENPI_BASE_CHECKPOINT` | Pi 0.5 base parameter path used when training starts. |
| `KINRT_CHECKPOINT_PATH` | Explicit checkpoint run or step directory used for evaluation. |
| `KINRT_CHECKPOINT_NUM` | Preferred numeric checkpoint step. |
| `KINRT_ACTION_CHUNK_SIZE` | Number of predicted actions executed per inference call. |
| `KINRT_PALIGEMMA_LORA_RANK` | Explicit PaliGemma LoRA rank for a checkpoint trained with a CLI override. |
| `KINRT_ACTION_EXPERT_LORA_RANK` | Explicit action-expert LoRA rank for a checkpoint trained with a CLI override. |
| `KINRT_EXTRA_PYTHONPATH` | Optional dependency path for isolated testing; normal installations do not need it. |
| `OPENPI_NUM_TRAIN_STEPS` | Optional training-step override for controlled smoke tests or ablations. |
| `OPENPI_BATCH_SIZE` | Optional global batch-size override. |
| `OPENPI_NUM_WORKERS` | Optional training DataLoader worker-count override. |
| `OPENPI_SAVE_INTERVAL` | Optional checkpoint save-interval override. |
| `OPENPI_WANDB_ENABLED` | Set to `0` to disable Weights & Biases for an offline run. |

## Limitations

- Joint control for the dual-ARX-X5 embodiment is the only validated schema.
- Batched simulation evaluation is enabled and processes multiple layouts concurrently; model inference within each batch remains sequential to bound accelerator memory use.
- A RoboDojo-trained KinRT checkpoint is required for benchmark results. A checkpoint from another embodiment is suitable only for interface testing.
- Simulator evaluation requires the complete RoboDojo Isaac Sim environment and assets.
- The official demo bundle is sufficient for pipeline validation but not for reporting benchmark performance.
- The 10,000-step checkpoint is an initial submission candidate, not the 60,000-step full-budget result.
