# MoPA

**Contributor:** ZHUShaolong | **Paper:** [MoPA: Coordinated Mobile Manipulation via Subsystem-Specific Perception Alignment](https://mopa-policy.github.io/) | **arXiv:** [2609.12081](https://arxiv.org/abs/2609.12081) | **Original code:** Not yet released ([project page](https://mopa-policy.github.io/): "Code Coming soon")

[MoPA: Coordinated Mobile Manipulation via Subsystem-Specific Perception Alignment](https://mopa-policy.github.io/)
is adapted to RoboDojo with one bank of **8 manipulation queries**, the base branch disabled,
and **4-step joint-action predictions**. Inputs are RGB images, a language instruction,
and arm/gripper joint states.

The complete model project lives in [`mopa/`](mopa/README.md), including installation metadata,
model code, data processing, training, and inference. It can be installed and run independently.
This directory provides the XPolicyLab entry points:

```text
policy/MoPA/
├── mopa/                    # Standalone model project
│   ├── pyproject.toml
│   ├── README.md
│   ├── models/, data/, training/
│   └── runtime.py
├── model.py                 # ModelTemplate entry point and shared path/dimension resolution
├── process_data.py          # Data conversion entry point
├── train.py                 # Training entry point
├── deploy.py / deploy.yml
└── *.sh                     # Installation, data, training, and evaluation launchers
```

See the [model documentation](mopa/README.md) for the architecture, native data format,
and standalone training and inference.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

```bash
conda create -n mopa python=3.11 -y
conda activate mopa
cd XPolicyLab/policy/MoPA
bash install.sh
```

The script installs the local `mopa` package and XPolicyLab. Prepare a local
Qwen3-VL-4B-Instruct-Action directory containing weights, tokenizer, processor,
and chat template files.

## Data Processing

```bash
bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> joint [expert_data_num] [options]

bash process_data.sh RoboDojo stack_bowls arx_x5 joint \
  --source /path/to/data/RoboDojo/stack_bowls/arx_x5/data
```

Only `joint` actions are supported. The example uses the registered `arx_x5` configuration;
dimensions come from the shared robot configuration. The default input is
`<parent>/data/<bench_name>/<ckpt_name>/<env_cfg_type>/data`, overridden by `SOURCE_DATA`
or `--source`. The output is `data/<bench_name>-<ckpt_name>-<env_cfg_type>-joint/`,
overridden by `--output`. Existing directories are not overwritten.

Conversion reads the HDF5 `state/`, `action/`, and `vision/` groups and decodes images
to RGB through the shared `decode_image_bit` helper. The model's data pipeline then
produces NPZ episodes and statistics; it does not use the LeRobot format.
The default camera order is `cam_head cam_left_wrist cam_right_wrist`, with images
resized to 224 x 224. Set these with `--cameras` and `--image-size HEIGHT WIDTH`.
Use `--instruction` when the source has no instruction. When multiple paraphrases
are available, the first is used.

## Training

```bash
bash train.sh <bench_name> <ckpt_name> <env_cfg_type> joint <seed> <gpu_id> [options]

export MOPA_BASE_VLM=/path/to/Qwen3-VL-4B-Instruct-Action
bash train.sh RoboDojo stack_bowls arx_x5 joint 0 0
```

The entry point validates dataset dimensions with the shared dimension helper, then
uses the model's [default configuration](mopa/configs/model.json) and trainer.
Override paths with `--dataset` and `--output`; see `python train.py --help` for other options.
Checkpoints are saved to `checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-joint-<seed>/`:

```text
config.json
model.pt
dataset_statistics.json
```

Keep all three files together. The checkpoint format is `xpl-mopa-arm-v1`; loading
validates the robot configuration and joint layout. Weights with a base branch or
two query banks cannot be loaded directly; train the corresponding arm policy.
Training uses one process and does not restore optimizer or scheduler state.
Nonempty output directories are not overwritten.

## Evaluation

```bash
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> joint <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_conda_env> <eval_env_conda_env>

EVAL_ENV_TYPE=sim bash eval.sh RoboDojo stack_bowls stack_bowls arx_x5 joint 0 \
  0 0 mopa base
```

Evaluation requires the checkpoint above and `env_cfg/` in the parent workspace.
Set `EVAL_ENV_TYPE=debug` to use the interface debugging environment and
`DEBUG_OBS_ENCODED=1` to transmit encoded images. The server decodes images before
passing RGB arrays to the model. Batched observations and actions are aligned by `env_idx`.

## Configuration

Model options in `deploy.yml`:

| Key | Default | Description |
| --- | --- | --- |
| `checkpoint_path` | null | Checkpoint directory or `model.pt`; uses standard path resolution when omitted |
| `base_vlm` | null | New directory for the same Qwen assets |
| `device` | cuda | Inference device; CPU automatically uses float32 |
| `dtype` | bfloat16 | GPU backbone precision; the action head uses float32 |
| `execute_steps` | null | Actions to execute per chunk, from 1 to 4; defaults to all 4 |
| `request_timeout_s` | 120 | RPC timeout in seconds |

Language is read from `instruction`, with `instructions` as a fallback when `instruction`
is missing or empty. Both fields accept a string or a list of strings; when multiple
variants are provided, the first is used, matching data conversion. `ee` actions are not supported.
