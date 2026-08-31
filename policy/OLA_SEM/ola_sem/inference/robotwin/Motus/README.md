# OLA-SEM Policy Evaluation on RoboTwin

This guide explains how to deploy and evaluate OLA-SEM on RoboTwin 2.0. The
policy directory keeps the internal name `Motus` for compatibility with
RoboTwin and existing checkpoints.

## Environment

Install RoboTwin 2.0 by following its official installation guide. A separate
RoboTwin environment is recommended because simulator and training dependencies
may differ.

After activating the RoboTwin environment, install the policy dependencies:

```bash
cd /path/to/OLA-Sem/inference/robotwin/Motus
pip install -r requirements.txt
```

## Deployment

Copy the complete policy directory into RoboTwin:

```bash
cp -a /path/to/OLA-Sem/inference/robotwin/Motus \
  /path/to/RoboTwin/policy/
cd /path/to/RoboTwin/policy/Motus
cp paths_config.example.yml paths_config.yml
```

The deployed layout should include:

```text
RoboTwin/
├── policy/Motus/
│   ├── deploy_policy.py
│   ├── deploy_policy.yml
│   ├── eval.sh
│   ├── auto_eval.sh
│   ├── paths_config.yml
│   ├── tasks_all.txt
│   ├── models/
│   ├── utils/
│   └── bak/wan/
└── script/eval_policy.py
```

## Download the released checkpoint

The examples in this guide use the released
[`Kosmos524/ola_sem`](https://www.modelscope.cn/models/Kosmos524/ola_sem/files)
checkpoint. Download it with either the ModelScope CLI:

```bash
pip install modelscope
modelscope download --model Kosmos524/ola_sem \
  --local_dir /path/to/checkpoints/ola_sem
```

or the Python SDK:

```python
from modelscope import snapshot_download

snapshot_download(
    "Kosmos524/ola_sem",
    local_dir="/path/to/checkpoints/ola_sem",
)
```

The downloaded checkpoint should have this layout:

```text
/path/to/checkpoints/ola_sem/
├── config.json
├── configuration.json
└── pytorch_model/
    └── mp_rank_00_model_states.pt
```

Throughout this guide, `/path/to/checkpoints/ola_sem` is an example placeholder.
It may be placed anywhere, but `checkpoint_path` must point to its
`pytorch_model/` subdirectory rather than to the repository root or the `.pt`
file itself.

## Configuration

Edit `paths_config.yml` before evaluation:

```yaml
robotwin_root: "/path/to/RoboTwin"
conda_env: "/path/to/conda/envs/RoboTwin"
checkpoint_path: "/path/to/checkpoints/ola_sem/pytorch_model"
wan_path: "/path/to/Wan2.2-TI2V-5B"
vlm_path: "/path/to/Qwen3-VL-2B-Instruct"

gpu_ids: []
task_config: "demo_clean"
seed: 42
tasks_file: "tasks_all.txt"
instruction_type: "unseen"
test_num: 100

inference_mode: "history_flow"
num_inference_timesteps: 4
history_action_noise_std: 0.02
future_video_denoise_fraction: 1.0
```

Required paths:

- `robotwin_root`: the RoboTwin repository root.
- `checkpoint_path`: the exported `pytorch_model/` directory containing
  `mp_rank_00_model_states.pt`.
- `wan_path`: Wan2.2 directory containing the VAE, T5 weights, and tokenizer.
- `vlm_path`: Qwen3-VL directory used to load its configuration and processor.
- `conda_env`: RoboTwin environment path or name. Leave it empty if the correct
  environment is already active.

The published `Kosmos524/ola_sem` checkpoint records the following metadata in
its adjacent `config.json`:

```json
"flow_source": {
  "mode": "history",
  "video_mode": "gaussian",
  "action_noise_std": 0.02,
  "history_length": 16
}
```

The matching evaluation settings are therefore `inference_mode:
"history_flow"`, `num_inference_timesteps: 4`, and
`history_action_noise_std: 0.02`. Keep `config.json` beside the downloaded
`pytorch_model/` directory; the policy reads it at startup and validates that
`history_length` matches the checkpoint action chunk size.

`gpu_ids: []` enables `nvidia-smi` auto-detection. Set an explicit list such as
`[0, 1, 2, 3]` when needed. Batch evaluation assigns one task to each GPU at a
time.

### Clean2Clean and Clean2Rand

Use the same released checkpoint for both RoboTwin evaluation settings. Select
the environment split with `task_config`:

```yaml
# Clean2Clean
task_config: "demo_clean"
```

```yaml
# Clean2Rand
task_config: "demo_randomized"
```

Keep `tasks_file: "tasks_all.txt"` for a complete 50-task evaluation. A custom
task file may be used to resume or evaluate only a subset of tasks.

## Inference modes

### Legacy

```yaml
inference_mode: "legacy"
```

Legacy mode uses Gaussian flow sources and supports checkpoints created before
history-flow training was introduced. It is not the matching mode for the
released `Kosmos524/ola_sem` checkpoint shown above.

### History flow

```yaml
inference_mode: "history_flow"
```

History-flow mode conditions the next prediction on qpos values actually
executed by the simulator. Use this mode for the released
`Kosmos524/ola_sem` checkpoint. The checkpoint layout must be:

```text
checkpoint_step_N/
├── config.json
└── pytorch_model/
    └── mp_rank_00_model_states.pt
```

The adjacent `config.json` must contain `flow_source.mode: "history"` and a
`history_length` equal to the action chunk size. Its `video_mode` may be
`gaussian` or `history`. `history_action_noise_std` overrides the checkpoint
noise value and must be non-negative.

`num_inference_timesteps` must be positive.
`future_video_denoise_fraction` must be between `0.0` and `1.0`; `1.0` denoises
the future-video branch throughout the full inference schedule.

## Running evaluation

Evaluate one task:

```bash
cd /path/to/RoboTwin/policy/Motus
bash eval.sh hanging_mug
```

Evaluate all tasks listed in `tasks_all.txt` across the configured GPUs:

```bash
bash auto_eval.sh
```

To use another configuration file:

```bash
CONFIG_FILE=/absolute/path/to/paths_config.yml bash eval.sh hanging_mug
CONFIG_FILE=/absolute/path/to/paths_config.yml bash auto_eval.sh
```

Common values can also be overridden without editing YAML:

```bash
GPU_ID=1 TEST_NUM=10 NUM_INFERENCE_TIMESTEPS=4 bash eval.sh hanging_mug
```

Other supported overrides include `ROBOTWIN_ROOT`, `CONDA_ENV`,
`CHECKPOINT_PATH`, `WAN_PATH`, `VLM_PATH`, `TASK_CONFIG`, `SEED`,
`INSTRUCTION_TYPE`, `INFERENCE_MODE`, `HISTORY_ACTION_NOISE_STD`,
`FUTURE_VIDEO_DENOISE_FRACTION`, `TASKS_FILE`, and `GPU_IDS`.

## Logs and troubleshooting

Each run writes logs to `logs_YYYYMMDD_HHMMSS/` under the policy directory
unless `LOG_DIR` is set.
