# OLA_SEM

**Contributor:** OLA-SEM | **Paper:** To be released | **arXiv:** Pending | **Original code:** https://github.com/chris1220313648/OLA-Sem

This adapter integrates the OLA-SEM language-action world model for RoboTwin 2.0. It vendors the required upstream inference and training source under `ola_sem/`; checkpoints and pretrained Wan/Qwen assets remain external. The supported contract is `env_cfg_type=aloha_agilex`, `action_type=joint`, one environment, a 14-D absolute-qpos action, and a 16-step chunk.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

The installer is intended for a new user without a pre-existing OLA-SEM
environment. It creates an `ola_sem` conda environment with Python 3.10 and
installs the complete inference/training stack, including the validated
PyTorch 2.7.1/cu128 and FlashAttention 2.8.3 combination:

```bash
cd XPolicyLab/policy/OLA_SEM
bash install.sh ola_sem
conda activate ola_sem
```

Run the installer on a machine with a CUDA toolkit and compiler available,
because FlashAttention may be built from source. To use a compatible prebuilt
wheel, set `OLA_SEM_FLASH_ATTN_WHEEL=/path/to/flash_attn.whl`. CUDA/PyTorch
variants can be changed with `OLA_SEM_TORCH_INDEX_URL`,
`OLA_SEM_TORCH_VERSION`, and `OLA_SEM_TORCHVISION_VERSION`; see
`bash install.sh --help` for all overrides. Passing an existing conda name or
prefix is supported and is idempotent. Set `OLA_SEM_SKIP_TORCH_INSTALL=1` only
when that environment already has a compatible PyTorch build.

The policy environment is separate from the RoboTwin simulator environment.
Install RoboTwin by following its
[official installation guide](https://robotwin-platform.github.io/doc/usage/robotwin-install.html)
and install `msgpack-numpy>=0.4.8` there so the WebSocket client can exchange
NumPy arrays. The final two arguments to `eval.sh` are respectively the policy
environment (`ola_sem` below) and that RoboTwin environment.

## Data Processing

No training data is included. The OLA-SEM RoboTwin download and conversion
utilities are vendored under `ola_sem/data/robotwin2/robotwin_data_convert/`, so
a separate OLA-SEM checkout is not required. The commands below follow the
[upstream data guide](https://github.com/chris1220313648/OLA-Sem#robotwin-data)
and its detailed
[conversion guide](https://github.com/chris1220313648/OLA-Sem/blob/main/data/robotwin2/robotwin_data_convert/README.md).

Enter the adapter and activate the `ola_sem` environment created above:

```bash
cd /path/to/XPolicyLab/policy/OLA_SEM
conda activate ola_sem
CONVERTER_DIR=ola_sem/data/robotwin2/robotwin_data_convert
```

Download all clean and randomized demonstrations. Supplying `--tasks` followed
by task names downloads only those tasks:

```bash
python "$CONVERTER_DIR/download_robotwin_dataset.py" \
  --output_dir /path/to/robotwin_raw_dataset
```

Copy `$CONVERTER_DIR/config.yml`, then set at least `source_root`, `target_root`,
`wan_repo_path`, and `cuda_devices` in the copy. Keep
`enable_t5_embeddings: true`, because training expects `umt5_wan/`. Run the
conversion with the edited configuration:

```bash
python "$CONVERTER_DIR/robotwin_converter.py" \
  --config /path/to/robotwin_convert.yml

# Use the directory that directly contains the task folders. Depending on the
# downloader version, this may be robotwin_raw_dataset/dataset.
python "$CONVERTER_DIR/robotwin_generate_epos_from_raw.py" \
  --raw-root /path/to/raw-task-root \
  --target-root /path/to/robotwin_dataset

python "$CONVERTER_DIR/robotwin_generate_language_action.py" \
  --target-root /path/to/robotwin_dataset \
  --input-dir-name epos \
  --input-mode absolute_xyzrpy \
  --window-size 16
```

The final dataset must contain both `clean/` and `randomized/`, with task
directories containing `videos`, `qpos`, `umt5_wan`, and `language_action`.
Then register it with this adapter:

```bash
cd /path/to/XPolicyLab/policy/OLA_SEM
bash process_data.sh RoboTwin history_flow_clean aloha_agilex joint \
  /path/to/robotwin_dataset
```

`process_data.sh` does not download or duplicate the dataset. It validates the
converted layout and creates an ignored symlink using the standard XPolicyLab
run name. The source path may instead be supplied through
`OLA_SEM_DATASET_ROOT`.

## Training

Standard training requires these three pretrained assets:

- [`Kosmos524/d0_v`](https://www.modelscope.cn/models/Kosmos524/d0_v/files):
  initialization weights supervised on mixed robot datasets.
- [`Wan-AI/Wan2.2-TI2V-5B`](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B):
  Wan video backbone, VAE, and UMT5 components.
- [`Qwen/Qwen3-VL-2B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct):
  vision-language backbone.

One way to download them is:

```bash
python -m pip install modelscope huggingface_hub

modelscope download --model Kosmos524/d0_v \
  --local_dir /path/to/pretrained_models/d0_v

hf download Wan-AI/Wan2.2-TI2V-5B \
  --local-dir /path/to/pretrained_models/Wan2.2-TI2V-5B

hf download Qwen/Qwen3-VL-2B-Instruct \
  --local-dir /path/to/pretrained_models/Qwen3-VL-2B-Instruct
```

The expected layout is:

```text
pretrained_models/
├── d0_v/
├── Qwen3-VL-2B-Instruct/
└── Wan2.2-TI2V-5B/
    └── Wan2.2_VAE.pth
```

`d0_v` is the normal training initialization.
After downloading the assets, provide the model and data locations explicitly:

```bash
export OLA_SEM_DATASET_ROOT=/path/to/robotwin_dataset
export OLA_SEM_WAN_PATH=/path/to/Wan2.2-TI2V-5B
export OLA_SEM_VLM_PATH=/path/to/Qwen3-VL-2B-Instruct
export OLA_SEM_FINETUNE_CHECKPOINT=/path/to/pretrained_models/d0_v

bash train.sh RoboTwin history_flow_clean aloha_agilex joint 42 \
  0,1,2,3,4,5,6,7 8
```

Without explicit model-path overrides, `train.sh` looks below
`ola_sem/pretrained_models/`; `OLA_SEM_PRETRAINED_ROOT` changes that common
root. Set `OLA_SEM_FINETUNE_CHECKPOINT` to an OLA-SEM `pytorch_model/` directory
only when intentionally continuing from an existing fine-tuned checkpoint.

Outputs use `checkpoints/RoboTwin-history_flow_clean-aloha_agilex-joint-42/`.
Other supported overrides are `OLA_SEM_OUTPUT_ROOT`, `OLA_SEM_MAX_STEPS`,
`OLA_SEM_MAX_EPISODES`, `OLA_SEM_BATCH_SIZE`, `OLA_SEM_NUM_WORKERS`, and
`OLA_SEM_REPORT_TO`.

## Evaluation

The released
[`Kosmos524/ola_sem`](https://www.modelscope.cn/models/Kosmos524/ola_sem/files)
checkpoint can be used directly for evaluation. Its root must contain
`config.json` beside `pytorch_model/mp_rank_00_model_states.pt`. This adapter
validates `flow_source.mode=history`, `video_mode=gaussian`,
`action_noise_std=0.02`, and `history_length=16` and evaluates with four
denoising steps. Both workflows below run directly on a machine that already
has a GPU.

Set the shared model and simulator paths first:

```bash
export OLA_SEM_WAN_PATH=/path/to/Wan2.2-TI2V-5B
export OLA_SEM_VLM_PATH=/path/to/Qwen3-VL-2B-Instruct
export OLA_SEM_ROBOTWIN_ROOT=/path/to/RoboTwin
```

### Single-task evaluation

Choose `demo_clean` or `demo_randomized` and the number of valid episodes,
then call the standard ten-argument `eval.sh` interface:

```bash
export EVAL_ENV_TYPE=sim
export OLA_SEM_ROBOTWIN_TASK_CONFIG=demo_clean
export OLA_SEM_TEST_NUM=100
export OLA_SEM_EVAL_VIDEO_LOG=false

bash eval.sh RoboTwin hanging_mug \
  /path/to/checkpoints/ola_sem \
  aloha_agilex joint 42 0 0 \
  ola_sem \
  /path/to/robotwin-conda-env
```

Here `ola_sem` is the policy conda environment and the final argument is the
RoboTwin conda environment. Change `hanging_mug` to any RoboTwin 2.0 task name. `EVAL_ENV_TYPE=debug` instead uses XPolicyLab's synthetic
debug client and does not measure RoboTwin task success.

## Configuration

The standard run-specific keys in `deploy.yml` are supplied by `eval.sh`. The
OLA-SEM-specific and transport keys are:

| Key | Default | Meaning / override |
| --- | --- | --- |
| `checkpoint_path` | `null` | Explicit OLA-SEM checkpoint root. When unset, the shared checkpoint resolver uses `ckpt_name`; an absolute path passed as `ckpt_name` is also accepted. |
| `wan_path` | `null` | Wan2.2-TI2V-5B root. Required at runtime and normally supplied by `OLA_SEM_WAN_PATH`. |
| `vlm_path` | `null` | Qwen3-VL-2B-Instruct root. Required at runtime and normally supplied by `OLA_SEM_VLM_PATH`. |
| `device` | `cuda` | Device used for model inference. |
| `inference_mode` | `history_flow` | Inference algorithm. This adapter accepts only `history_flow`. |
| `num_inference_timesteps` | `4` | Number of denoising steps used for each predicted action chunk. |
| `history_action_noise_std` | `0.02` | Noise standard deviation for executed-action history; it must match the checkpoint metadata. |
| `future_video_denoise_fraction` | `1.0` | Fraction of the denoising schedule applied to future-video latents; valid range is `[0, 1]`. |
| `request_timeout_s` | `1200` | Compatibility value matching the top-level RoboTwin client's request timeout. The current wrapper sets the same value directly. |
| `max_connect_seconds` | `1200` | Compatibility value matching the top-level client's connection retry window. The current wrapper sets the same value directly. |
| `ws_ping_interval_s` | `20` | Policy-server WebSocket ping interval in seconds. |
| `ws_ping_timeout_s` | `120` | Policy-server WebSocket ping timeout in seconds. |

## Notes

- Batched inference with more than one environment is unsupported and fails explicitly.
- Images remain RGB throughout. The runtime adapter receives decoded arrays; the vendored training loader's BGR-to-RGB conversion occurs only immediately after `cv2.VideoCapture.read()`.
