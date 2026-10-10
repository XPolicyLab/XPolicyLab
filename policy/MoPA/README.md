# MoPA

**Contributor:** ZHUShaolong | **Paper:** [MoPA: Coordinated Mobile Manipulation via Subsystem-Specific Perception Alignment](https://mopa-policy.github.io/) | **arXiv:** [2609.12081](https://arxiv.org/abs/2609.12081) | **Original code:** [project page](https://mopa-policy.github.io/)

This adapter provides data conversion, training and inference with Qwen3-VL,
separate arm/base query banks and a flow-matching action head. The model package
lives in [`mopa/`](mopa/README.md). The default configuration uses 8 arm queries,
8 base queries and 32-step action chunks. Mobile observations use a 75-value
state/action representation; generic arm/gripper layouts are also supported with
compatible action-slice configuration.

Shared argument conventions, checkpoint naming and deployment are documented in
the [XPolicyLab README](../../README.md).

## Installation

```bash
conda create -n mopa python=3.11 -y
conda activate mopa
cd XPolicyLab/policy/MoPA
bash install.sh
```

The script installs this local package and XPolicyLab. Download Qwen3-VL-4B-Instruct
assets separately, including weights, configuration, tokenizer, processor and chat
template. Set `MOPA_BASE_VLM` to their local directory. No weights or datasets are
included in this adapter.

## Data Processing

```bash
bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> joint [expert_data_num] [options]

bash process_data.sh "$BENCH_NAME" mobile_run m92uw joint \
  --source /path/to/hdf5_episodes
```

Only `joint` actions are supported. For generic robots, dimensions come from
XPolicyLab's shared robot helper. The default input is
`<workspace>/data/<bench_name>/<ckpt_name>/<env_cfg_type>/data`; override it with
`--source` or `SOURCE_DATA`. The default output is
`data/<bench_name>-<ckpt_name>-<env_cfg_type>-joint/`; `--output` overrides it.
Existing output directories are rejected.

For `m92uw`, `mobile` or `mobile_m92uw`, the policy-local
[mobile layout](mopa/data/mobile.py) packs 69 raw values into 75 model values using
position plus rotation-6D for poses. Its manipulation slice is `[0,56)` and its
mobility slice is `[56,75)`. These pose, torso and base fields extend the shared
arm/gripper schema. Conversion writes a manifest pointing to source HDF5 files,
with exact min/max statistics and sampled quantiles; keep the source files at
those paths. Generic conversion writes decoded RGB NPZ episodes and quantile
statistics. Neither output uses the LeRobot format.

All stored image buffers are decoded with XPolicyLab's `decode_image_bit`. The
shared decoder handles legacy and marked RGB buffers. Custom data producers must
use `encode_image_bit` when writing images. Default cameras are
`cam_head cam_left_wrist cam_right_wrist`; use `--cameras` and
`--image-size HEIGHT WIDTH` to override the order and 224 x 224 resize.
Generic conversion accepts `--instruction` as a fallback; mobile HDF5 episodes
must contain a nonempty scalar `instruction` dataset.

## Training

```bash
bash train.sh <bench_name> <ckpt_name> <env_cfg_type> joint <seed> <gpu_id> [options]

export MOPA_BASE_VLM=/path/to/Qwen3-VL-4B-Instruct
bash train.sh "$BENCH_NAME" mobile_run m92uw joint 0 0
```

Use a comma-separated GPU list, such as `0,1`, or `all` for distributed training
through `torchrun`. `--batch-size` is per process. The packaged
[configuration](mopa/configs/model.json) targets the mobile layout. Other layouts
need `--config /path/to/config.json` with compatible
`manipulation_action_range` and `mobility_action_range`.

`--dataset` and `--output` override the standard paths. Training saves
`config.json`, `model.pt` and `dataset_statistics.json` in
`checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-joint-<seed>/` and optionally
`steps_<step>/` subdirectories. Keep these three files together. Nonempty outputs
are rejected; optimizer/scheduler resume is not implemented. See
`python train.py --help` for optimizer, logging and checkpoint options.

## Evaluation

```bash
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> joint <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_env_or_path> <eval_env_conda_env>

EVAL_ENV_TYPE=sim bash eval.sh "$BENCH_NAME" "$TASK_NAME" mobile_run m92uw joint 0 \
  0 0 mopa "$EVAL_ENV"
```

The standard launcher uses the workspace environment client. Set
`EVAL_ENV_TYPE=debug` for its interface check and `DEBUG_OBS_ENCODED=1` to exercise
server image decoding where supported. The server supplies RGB arrays to the
model. Batched observations/actions are associated using `env_idx`.

The additional mobile debug launcher uses synthetic observations and checks the
WebSocket transport plus the optional 90-value action packet:

```bash
export MOPA_CHECKPOINT_PATH=/path/to/checkpoint
MOPA_DEBUG_ENCODED=true MOPA_DEBUG_BATCH=true bash eval_mobile_debug.sh \
  "$BENCH_NAME" mobile_debug mobile_run m92uw joint 0 0 0 mopa mopa
```

This is an interface check, not a task-success evaluation. The extended packet
adds three world-frame poses held at their observed values; they are not predicted
by the 75-value model. `MOPA_DEBUG_EPISODES`, `MOPA_DEBUG_STEPS` and
`MOPA_DEBUG_BATCH_SIZE` control its synthetic run. The policy server and mobile
debug client accept environment names, full environment paths or Python
interpreters; use `CONDA_ENVS_PATH` for environments in other locations.

## Configuration

| Key | Default | Meaning |
| --- | --- | --- |
| `checkpoint_path` | null | Checkpoint directory or `model.pt`; otherwise use the shared checkpoint resolver |
| `base_vlm` | null | Relocated directory containing the same Qwen assets used in training |
| `device` | cuda | `cuda` or `cpu` |
| `dtype` | bfloat16 | Backbone dtype; CPU uses float32 |
| `execute_steps` | null | Execute up to 16 actions by default, capped by the saved horizon; explicit values must be within that horizon |
| `action_key_style` | singular | Mobile action keys: `singular` or `plural` |
| `input_color_order` | rgb | Use `bgr` only to adapt RGB observations to a checkpoint trained on BGR images |
| `mobile_action_contract` | mopa75 | Native 11-field output or `extended90` with 14 fields |
| `request_timeout_s` | 120 | RPC timeout in seconds |

`MOPA_CHECKPOINT_PATH`, `MOPA_EXECUTE_STEPS`, `MOPA_ACTION_KEY_STYLE` and
`MOBILE_ACTION_CONTRACT` override the corresponding deployment keys in the server
launcher. Instructions are read from `instruction`, then `instructions`, then the
configured task name; for multiple variants, the first is used.

## Model Assets

The checkpoint format is `xpl-mopa-query-dmot-v2`. Configuration records the layout,
query counts, action slices and camera order, all of which must match inference.
New datasets and checkpoints use the layout tag `mobile_m92uw`. Older layout
labels are accepted when their saved metadata has the same explicit 75-D model,
69-D raw and joint-dimension signature. The policy-local ignore file excludes weights,
datasets, generated outputs, logs and caches from version control.
