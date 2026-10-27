# Simate_beta

**Contributor:** SIPAILab | **Paper:** Not provided | **arXiv:** Not provided | **Original code:** [SIPAILab/simate-beta](https://github.com/SIPAILab/simate-beta)

AE memory policy supporting native SiMate training and self-contained inference
for `RoboDojo / arx_x5 / joint`. Training uses upstream revision
`8ae595cda0eea1595afdb240749c7b595872f5c0`.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

Run from this directory on Linux with Conda and a compatible NVIDIA driver.
Install the RoboDojo simulator environment separately.

Evaluation:

```bash
bash install.sh
conda activate simate-beta-eval
```

The default environment is `simate-beta-eval` with Python 3.11. The installer
reuses supported CUDA PyTorch 2.6–2.8, or installs 2.7.0/CUDA 12.8 when absent.
An optional argument to `install.sh` selects another environment name or path;
pass it to `eval.sh` as well. Set `CONDA_EXE` if Conda is not on PATH.

Training:

```bash
conda create -n simate-beta-train python=3.11 pip -y
conda activate simate-beta-train
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
bash install_training.sh
```

`install_training.sh` installs the pinned upstream source in `training_source/`
and its dependencies, including the `hf` CLI. `SIMATE_TRAIN_ROOT` selects an
existing clean checkout at that revision; `PYTHON` selects the interpreter for
training and data scripts.

## Data Processing

Use the prepared [SiMate dataset](https://huggingface.co/datasets/SIPAILab/simate-beta-training-data)
and [initialization bundle](https://huggingface.co/SIPAILab/simate-beta-init).
The dataset is native MP4/Parquet TAR with shard indexes, manifest and training
annotations. Download the complete package; this entry point does not convert
raw XPolicyLab HDF5 or LeRobot datasets.

Download and prepare the data in the training environment:

```bash
hf download SIPAILab/simate-beta-training-data --repo-type dataset \
  --revision ebcf1c15641c5b46eb1d7efef312e8ab567a3f4c --local-dir ./data/simate-beta-training-data
bash download_initialization.sh

bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [--verify]
bash process_data.sh RoboDojo mem20-ae arx_x5 joint --verify
```

`SIMATE_DATA_ROOT` overrides the dataset location. `process_data.sh` links the
prepared package into `data/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>/`;
`--verify` checks packaged SHA-256 sums. The recipe uses the complete package,
without an `expert_data_num` argument.

## Training

The upstream `configs/simate-beta_mem20_ae_80k.yaml` defines the default recipe:
80k updates, 4 × 8 GPUs, batch 4/GPU (global batch 128), BF16 HSDP,
AdamW 1e-5 → 1e-6, 500 warmup updates and EMA 0.99. SigLIP and the AE encoder
are frozen. Source weights are 5:0.7:1 for `robodojo_all`, `make_kong` and
`play_tic_tac_toe`.

```bash
bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id> \
  [--dry-run] [-- dotlist overrides...]

# Run on each node with NODE_RANK=0, 1, 2 or 3 and the same master address.
export NODE_RANK=0 MASTER_ADDR=10.0.0.1 MASTER_PORT=29500
export SIMATE_INIT_ROOT="$PWD/pretrained/simate-beta-init"
bash train.sh RoboDojo mem20-ae arx_x5 joint 0 0,1,2,3,4,5,6,7
```

All nodes must use the same source, data, initialization and shared checkpoint
root. `gpu_id` selects local GPUs. `--dry-run` checks the launch command;
dotlist overrides modify the recipe, while run name, seed and output location
come from the standard arguments. Use a new `ckpt_name` for each run.

`SIMATE_CHECKPOINT_ROOT` overrides `checkpoints/`; `CACHE_ROOT` selects temporary
training caches. Dataset paths and phase fingerprints come from
`training_assets.json`; normalization statistics come from the initialization bundle.

Each run writes its resolved configuration, `metrics.jsonl`, logs, distributed
checkpoints and FP32 EMA exports under
`checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>/`.
On completion, `model.safetensors` links to the final export and `tokenizer.model`
is copied alongside it. Pass the training label to eval, or the absolute run
directory for a custom checkpoint root. When copying a checkpoint, dereference
the model symlink.

## Evaluation

Download the evaluation weights and run:

```bash
bash download_checkpoint.sh

bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_env_or_uv_path> <eval_env_conda_env>

EVAL_ENV_TYPE=debug bash eval.sh RoboDojo stack_bowls simate-beta-robodojo-eval arx_x5 joint 0 0 1 simate-beta-eval RoboDojo
EVAL_ENV_TYPE=sim bash eval.sh RoboDojo stack_bowls simate-beta-robodojo-eval arx_x5 joint 0 0 1 simate-beta-eval RoboDojo
```

To evaluate your own completed training run, replace `simate-beta-robodojo-eval`
with its training label (for example, `mem20-ae`) or absolute run directory.

Set `DEBUG_OBS_ENCODED=1` with debug mode to check encoded camera observations.
The shared server decodes RGB images; the adapter preserves their channel order.

The deployment loop updates history every action step, including within a chunk.
History is isolated per evaluation/environment and cleared between episodes.
Batched inference is sequential and preserves active-environment order. Arm
actions are relative to the current joint state; grippers use absolute openness.

## Model Assets

Use the [initialization bundle](https://huggingface.co/SIPAILab/simate-beta-init)
for training. It contains the FP32 policy, history encoder, tokenizer,
frozen normalization statistics and license files.

```bash
bash download_initialization.sh [destination]
```

The default destination is `pretrained/simate-beta-init`. The downloader pins
revision `c924b68d04828221ed1f11e2abc7fe8e7174c24d`, verifies all packaged
SHA-256 sums and reuses matching local files. For a custom destination, set
`SIMATE_INIT_ROOT` to that directory before training.

Use the [evaluation checkpoint](https://huggingface.co/SIPAILab/sipai-robodojo-eval-ae)
for inference. `download_checkpoint.sh [destination]` pins
revision `0738aab54e0c873db76d4d2aec7a467ca1b04120`, verifies SHA-256 and writes
`model.safetensors` and `tokenizer.model` to
`checkpoints/simate-beta-robodojo-eval`.

For custom evaluation weights, provide a complete FP32 AE policy and its
matching tokenizer. Encoder-only and pooling checkpoints are unsupported.

## Configuration

Model-specific defaults in `deploy.yml`:

```yaml
device: cuda:0
exec_chunk_size: 10
architecture: simate_beta_memory_ae_joint_v1
dtype: float32
history_frames: 25
history_interval: 20
history_grid_size: 4
history_compressor_hidden_dim: 512
max_token_len: 200
num_inference_steps: 10
```

At action step t, history uses `t-N*K, ..., t-K`; unavailable frames are masked.
The defaults encode 25 primary-camera history frames, spaced 20 steps apart,
into 25 × 16 tokens using a 4 × 4 AE grid and a 512-channel encoder. Current
camera views remain separate. `max_token_len` limits prompt tokens;
`num_inference_steps` controls flow integration. `exec_chunk_size` must be 1–50.
