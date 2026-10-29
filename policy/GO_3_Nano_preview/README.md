# GO-3 Nano Preview

**Contributor:** AgiBot Research | **Paper (method reference):** [GE-Act 2.0](https://ge-act-v2.github.io/assets/paper/GE-Act-2.0.pdf) | **arXiv:** [2609.05588](https://arxiv.org/abs/2609.05588) | **Code:** [GO-3 Nano Preview adapter](./)

`GO_3_Nano_preview` provides inference and supervised fine-tuning (SFT) for RoboTwin joint control. Three RGB camera views, robot joint/gripper state, and a task instruction are used to produce absolute joint-action chunks. The Python implementation and SFT code are bundled in `go3_nano_preview/`.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md).

The examples below run from the XPolicyLab root directory. Scripts locate their configuration relative to their own location, so they also work when called by absolute path from another directory. Evaluation requires a complete XPolicyLab/RoboTwin environment.

## Installation

Use Linux, Python 3.12, an NVIDIA GPU, and a driver compatible with the installed PyTorch version. Create the Python environment once:

```bash
python3.12 -m venv "$HOME/.venvs/GO_3_Nano_preview"
```

Fill in `POLICY_ENV` in [configs/runtime.conf](configs/runtime.conf), then install the dependencies into that environment:

```bash
# Inference only
bash policy/GO_3_Nano_preview/install.sh infer

# Training, including inference dependencies and CPU video decoding
bash policy/GO_3_Nano_preview/install.sh train
```

The dependency lists are [requirements-infer.txt](go3_nano_preview/requirements-infer.txt) and [requirements-train.txt](go3_nano_preview/requirements-train.txt). `PYTHON_BIN` can explicitly select the installation interpreter; otherwise `install.sh` uses `POLICY_ENV/bin/python`, falling back to `python3` only when `POLICY_ENV` is empty. Installation does not download model weights.

## Data Processing

Training consumes processed RoboTwin data with this layout for each episode:

```text
<SFT_DATA_ROOT>/<clip>/
├── head.mp4
├── left_hand.mp4
├── right_hand.mp4
└── actions.pt
```

`actions.pt` contains absolute joint/gripper actions with shape `[num_frames,14]`, ordered as the 6 left-arm joints, left gripper, 6 right-arm joints, and right gripper. Each JSONL line describes one episode using `clip`, `task_caption`, `video_frames`, `action_frames`, and `step_caption`; `clip` is relative to `SFT_DATA_ROOT`. Step captions contain `start_frame`, `end_frame`, and `action_text`, with an exclusive end frame. Preserve other caption and sampling fields in the original data.

[tools/process_data.sh](tools/process_data.sh) validates existing JSONL, video, and action files, then writes `train.jsonl` and a validation report to `SFT_PREPARED_DIR`. The videos and actions remain under `SFT_DATA_ROOT`. It does not convert raw RoboTwin demonstrations or official LeRobot exports into this custom format.

Set `SFT_JSONL_PATH`, `SFT_DATA_ROOT`, and `SFT_PREPARED_DIR` in `configs/runtime.conf`, then run:

```bash
bash policy/GO_3_Nano_preview/tools/process_data.sh
```

The destination must be a new directory outside the policy checkout. The positional form is also supported, including an optional expected episode count:

```bash
bash policy/GO_3_Nano_preview/tools/process_data.sh \
  <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expected_episode_count]
```

## Training

`train.sh` is the single entry point for single-GPU, multi-GPU, and platform-managed multi-node training. Fill in the SFT paths under [Configuration](#configuration) first.

```bash
# Use all visible GPUs; the default recipe is configs/robotwin_sft_ka_step.py
bash policy/GO_3_Nano_preview/train.sh

# Use one GPU
CUDA_VISIBLE_DEVICES=0 bash policy/GO_3_Nano_preview/train.sh

# Use two GPUs
CUDA_VISIBLE_DEVICES=0,1 bash policy/GO_3_Nano_preview/train.sh

# Inspect the configuration, topology, and command without loading a model
CUDA_VISIBLE_DEVICES=0,1 bash policy/GO_3_Nano_preview/train.sh --dry-run
```

The launcher detects visible GPUs automatically. `NPROC_PER_NODE`, or `SFT_NPROC` when the former is unset, can explicitly select the number of GPU processes per node. GPU visibility must match that count. Use `SFT_CONFIG` in the configuration file, a positional Python configuration path, or `--config` to select a recipe. A command-line configuration takes precedence over `SFT_CONFIG`.

```bash
bash policy/GO_3_Nano_preview/train.sh \
  --config policy/GO_3_Nano_preview/configs/robotwin_sft_ka_step.py \
  --run-name cotrain_robotwin_sft_ka_robotype --seed 1016
```

The original XPolicyLab positional interface remains available. Its GPU argument accepts a single ID or a comma-separated list. This form preserves the `configs/robotwin_sft.py` default unless `SFT_CONFIG` or `--config` overrides it, and uses 10,000,000 steps unless `SFT_STEPS` or `--steps` overrides that value:

```bash
bash policy/GO_3_Nano_preview/train.sh \
  <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_ids> [options]

# Example: two-GPU training through the standard interface
bash policy/GO_3_Nano_preview/train.sh RoboTwin robotwin-sft arx_x5 joint 1016 0,1
```

For multi-node training, the task platform must launch the same command once per node. All nodes need access to the code, dependencies, data, and shared output directory.

| Variable | Meaning at the training launcher |
| --- | --- |
| `WORLD_SIZE` | Number of nodes, e.g. `8` for an 8-node job |
| `RANK` | Current node index, supplied by the platform |
| `MASTER_ADDR` | Training network address of the master node |
| `MASTER_PORT` | Rendezvous port; defaults to `12345` |
| `NPROC_PER_NODE` | GPU processes per node; otherwise use `SFT_NPROC` or detect visible GPUs |

For 8 nodes with 8 GPUs each, set the launcher's `WORLD_SIZE` to **8**. After torchrun starts the workers, their `WORLD_SIZE` is **64**. The launcher does not start other nodes over SSH. Single-node jobs use torchrun's standalone rendezvous.

For a short training check:

```bash
CUDA_VISIBLE_DEVICES=0 bash policy/GO_3_Nano_preview/train.sh \
  --run-name robotwin_smoke --seed 1016 \
  --steps 1 --batch-size 1 --workers 0 --skip-validation --skip-save
```

These options affect only the current run. Add `--gradient-checkpointing` to reduce activation memory or `--synchronize-timing` to synchronize CUDA for timing.

### Outputs and Resuming Training

Outputs normally go to `$RUN_ROOT/checkpoints/<run_name>`. Set `SFT_OUTPUT_DIR` or pass `--output` to choose another directory. The positional interface derives the run name from its benchmark/checkpoint/profile/action/seed arguments; the simple interface uses `SFT_RUN_NAME`, unless `--run-name` overrides it.

| File or Directory | Contents |
| --- | --- |
| `run_settings.json` | Effective batch size, process count, steps, and run overrides |
| `train_rank<N>.jsonl` | Per-rank loss, gradients, learning rate, GPU memory, and timing |
| `validation.jsonl` | Fixed validation results when enabled |
| `step_N/` | Video and action model weights for inference |
| `training_state/` | DeepSpeed model, optimizer, and scheduler states |

Existing outputs are not overwritten by default. `--overwrite` (also accepted as `--overwrite True`) archives the existing output directory and starts a new run; it does not resume training. To resume, use a new output name and the same GPU process count as when the checkpoint was saved:

```bash
bash policy/GO_3_Nano_preview/train.sh \
  --run-name robotwin_resumed --seed 1016 \
  --resume /path/to/previous_run/training_state --steps 10000
```

The final step count must exceed the saved step count. Resuming restores the model, optimizer, and scheduler, but restarts the random streams for data sampling and augmentation. Initialization weights alone do not restore the optimizer or warmup state. After training, set `CKPT_PATH` to the generated `step_N/` directory for evaluation.

## Evaluation

Fill in the inference assets, environments, robot profile, and task in [configs/runtime.conf](configs/runtime.conf) once. Then run the evaluator directly, including from a new terminal:

```bash
bash policy/GO_3_Nano_preview/eval.sh
```

The script loads the configuration, starts the official policy server, waits for it, runs the environment client, and stops the server on completion or failure. `EVAL_ENV_TYPE=sim` is the default. To inspect the resolved commands or run the synthetic Debug client:

```bash
bash policy/GO_3_Nano_preview/eval.sh --dry-run
EVAL_ENV_TYPE=debug bash policy/GO_3_Nano_preview/eval.sh
DEBUG_OBS_ENCODED=1 EVAL_ENV_TYPE=debug bash policy/GO_3_Nano_preview/eval.sh
```

`--dry-run` checks required arguments and prints commands; it does not validate weights or start a server. Debug evaluation still loads the real policy checkpoint and is not a task success-rate measurement.

The standard ten-argument XPolicyLab interface remains supported. These arguments override their corresponding defaults in `runtime.conf`; model asset paths continue to come from that file:

```bash
bash policy/GO_3_Nano_preview/eval.sh \
  <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_env> <eval_env>
```

The adapter uses `model.py`, `deploy.py`, and [deploy.yml](deploy.yml). For separate machines, use `setup_eval_policy_server.sh` and `setup_eval_env_client.sh` as described in the [Deployment Flow](../../README.md#-deployment-flow); each script automatically loads its local `configs/runtime.conf`.

## Model Assets

The inference checkpoint directory must contain:

```text
step_N/
├── config.json
├── diffusion_pytorch_model.bin
└── action_model/
    ├── config.json
    └── diffusion_pytorch_model.bin
```

Inference also needs the complete Qwen 3.5 2B model/processor directory, VAE configuration and weights, and VAE latent statistics. Models are loaded offline, so prepare all assets in advance. The model configuration and weight architecture must match.

`VIDEO_PRETRAIN_PATH` and `ACTION_PRETRAIN_PATH` initialize the model before checkpoint loading. Their defaults point to the matching files under `CKPT_PATH`. SFT initialization uses the separate `SFT_VIDEO_PRETRAIN_PATH` and `SFT_ACTION_PRETRAIN_PATH` fields.

## Configuration

[configs/runtime.conf](configs/runtime.conf) is the single, annotated runtime configuration file. It uses Bash syntax so scripts can load it before starting the chosen Python environment. Edit the default after `:-` in the relevant quoted assignment:

```bash
# Examples of entries to edit inside configs/runtime.conf
export POLICY_ENV="${POLICY_ENV:-/opt/venvs/go3}"
export CKPT_PATH="${CKPT_PATH:-/path/to/step_N}"
export VLM_MODEL_PATH="${VLM_MODEL_PATH:-/path/to/qwen3.5-2b}"
export VAE_PATH="${VAE_PATH:-/path/to/vae}"
export EVAL_ENV="${EVAL_ENV:-/opt/venvs/robotwin}"
export EVAL_ENV_CFG_TYPE="${EVAL_ENV_CFG_TYPE:-arx_x5}"
```

Save the file and run the desired script. No separate path-setup command is required. Environment variables supplied for a particular run override the saved defaults. Use absolute paths and keep machine-specific values out of public commits.

| Setting | Purpose |
| --- | --- |
| `POLICY_ENV` | Policy environment directory containing `bin/python` |
| `CKPT_PATH` | Checkpoint directory used by no-argument evaluation |
| `VIDEO_PRETRAIN_PATH`, `ACTION_PRETRAIN_PATH` | Inference initialization files; default to files under `CKPT_PATH` |
| `VLM_MODEL_PATH`, `VAE_PATH` | Complete VLM and VAE model directories |
| `STATS_TRACKER_PATH` | VAE latent statistics; defaults to `$VAE_PATH/latent_stats.pth` |
| `XPL_ROOT` | XPolicyLab root, normally inferred from the policy location |
| `EVAL_ENV` | Simulation/Debug Python environment directory |
| `ROBOTWIN_ACTIVATE_SCRIPT` | Optional simulator activation script |
| `EVAL_ENV_CFG_TYPE` | Profile filename stem supplied by the simulator |
| `EVAL_TASK`, `EVAL_SEED`, `POLICY_GPU_ID`, `EVAL_GPU_ID` | Evaluation task, seed, and GPU defaults |
| `SFT_VIDEO_PRETRAIN_PATH`, `SFT_ACTION_PRETRAIN_PATH` | Training initialization weights |
| `SFT_JSONL_PATH`, `SFT_DATA_ROOT` | Training metadata and episode data root |
| `SFT_VAL_JSONL_PATH`, `SFT_VAL_DATA_ROOT` | Validation metadata and episode data root |
| `TASK_RECAP_PATH`, `STEP_RECAP_PATH` | Task/step text rewrite mapping files |
| `SFT_PREPARED_DIR` | New output directory for data validation/staging |
| `RUN_ROOT`, `SFT_OUTPUT_DIR` | Output root and optional exact training output directory |
| `SFT_CONFIG`, `SFT_RUN_NAME`, `SFT_SEED`, `SFT_STEPS`, `SFT_NPROC` | Optional training recipe, naming, and launch overrides |

Training and validation metadata must match their respective data roots. Both validation path variables must be filled for training even when using `--skip-validation`, because the training configuration reads them at import time. Training-only fields are unnecessary for checkpoint evaluation.

Training, validation, and inference read bundled state/action statistics from [assets/stats/robotwin.json](assets/stats/robotwin.json). The separate `STATS_TRACKER_PATH` contains VAE latent statistics.

Model behavior is configured in [deploy.yml](deploy.yml): `policy_name: GO_3_Nano_preview`, `protocol: ws`, `eval_batch: false`, three camera keys, `image_size: [256, 384]`, `action_horizon: 52`, and `num_ae_steps: 5` are the adapter defaults.

## Notes

- The shared setup at the end of `runtime.conf` selects the policy interpreter and sets Python import paths, offline model loading, and bundled statistics. Entry scripts load it automatically.
- `go3_nano_preview/serve_GO-3.py` remains part of the inference implementation: `model.py` imports its model-building functions. Official evaluation is launched through `eval.sh` or the two `setup_eval_*.sh` scripts.
- [configs/deploy.yml](configs/deploy.yml) belongs to the low-level standalone Python service; the official XPolicyLab adapter uses the policy-root `deploy.yml`.
