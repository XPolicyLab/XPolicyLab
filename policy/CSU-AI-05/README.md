# CSU-AI-05 RoboDojo Policy Adapter

**Contributor:** CSU-AI-05 Team

CSU-AI-05 serves its evaluated checkpoint through the XPolicyLab websocket interface for RoboDojo `arx_x5` joint-action evaluation. The exact model runtime used for the submitted evaluation is bundled under `G05/`; the adapter supports single- and batched-environment inference with optional sparse visual history.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

Use Python 3.10 and a clean policy environment. `install.sh` installs XPolicyLab and the bundled model runtime:

```bash
cd XPolicyLab/policy/CSU-AI-05
export G05_PYTHON=/path/to/python3.10
bash install.sh
```

For exact reproduction, use the bundled runtime and do not override `G05_ROOT`, which defaults to `policy/CSU-AI-05/G05`. The `G05_` prefix is retained only for compatibility with the bundled runtime and existing launch scripts.

## Data Processing

CSU-AI-05 consumes the official RoboDojo LeRobot v3.0 export directly; no policy-specific conversion is performed. The standard entrypoint validates that the prepared dataset is present:

```bash
cd XPolicyLab/policy/CSU-AI-05
export ROBODOJO_LEROBOT_V30_ROOT=/path/to/RoboDojo_lerobot_v30_video

# Template
bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num]

# Runnable CSU-AI-05 example
bash process_data.sh RoboDojo CSU-AI-05 arx_x5 joint
```

## Model Assets

The evaluated CSU-AI-05 bundle uses the [CSU-AI-05 checkpoint folder on Hugging Face](https://huggingface.co/datasets/ShaoRun/vla_ro/tree/main/checkpoints/CSU-AI-05). `download_checkpoint.sh` recognizes this dataset-tree URL, downloads only that folder through `huggingface_hub`, keeps `checkpoints/CSU_AI_05.pt` as the canonical weight name, and then applies the submitted checksum manifest. Archive URLs remain supported as an alternative.

```bash
cd XPolicyLab/policy/CSU-AI-05
bash download_checkpoint.sh
```

The Hugging Face URL is the downloader default. The default local destination is `policy/CSU-AI-05/checkpoints/RoboDojo-CSU-AI-05-arx_x5-joint-0`, which is the same path resolved by `setup_eval_policy_server.sh` and is ignored by Git. Folder downloads require `huggingface_hub` in the Python environment selected by `G05_PYTHON`. `download_checkpoint.sh` validates the complete 12-file bundle against the submitted checksum manifest before evaluation. If the dataset requires authentication, authenticate with Hugging Face in that environment before running the downloader.

The downloaded bundle must contain these files:

```text
policy/CSU-AI-05/checkpoints/RoboDojo-CSU-AI-05-arx_x5-joint-0/
├── checkpoints/CSU_AI_05.pt
├── action_tokenizer.pt
├── dataset_stats.json
├── .hydra/config.yaml
└── processor/
    ├── config.json
    ├── configuration.json
    ├── preprocessor_config.json
    ├── tokenizer.json
    ├── tokenizer_config.json
    ├── video_preprocessor_config.json
    ├── merges.txt
    └── vocab.json
```

`download_checkpoint.sh` verifies every required file against `checkpoint_sha256.txt`. Point the adapter to the checkpoint file or the verified bundle directory:

```bash
export G05_CKPT_PATH="$PWD/checkpoints/RoboDojo-CSU-AI-05-arx_x5-joint-0/checkpoints/CSU_AI_05.pt"
export ROBODOJO_G05_ACTION_SOURCE=fm
```

The nonstandard checkpoint layout is intentional. `model.py` accepts the explicit file above and also discovers `.pt` files under a bundle's `checkpoints/` directory. It resolves `action_tokenizer.pt`, `dataset_stats.json`, and `processor/` from the bundle automatically before loading the saved Hydra configuration.

## Training

`train.sh` forwards the standard XPolicyLab arguments and trailing Hydra overrides to the bundled training launcher:

```bash
cd XPolicyLab/policy/CSU-AI-05
export G05_PYTHON=/path/to/python3.10
export ROBODOJO_LEROBOT_V30_ROOT=/path/to/RoboDojo_lerobot_v30_video
export G05_LOGGER_MODE=offline

# Template; optional Hydra overrides follow gpu_id
bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> \
  <seed> <gpu_id> [hydra_override ...]

# Runnable CSU-AI-05 example
bash train.sh RoboDojo CSU-AI-05 arx_x5 joint 0 0,1,2,3,4,5,6,7 \
  model.max_steps=1000
```

## Evaluation

Run the required debug closed loop without Isaac Sim:

```bash
cd XPolicyLab/policy/CSU-AI-05
export EVAL_ENV_TYPE=debug
export G05_PYTHON=/path/to/python3.10
export G05_CKPT_PATH=/path/to/CSU-AI-05/checkpoints/CSU_AI_05.pt
export ROBODOJO_G05_ACTION_SOURCE=fm

# Template
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> \
  <action_type> <seed> <policy_gpu_id> <env_gpu_id> \
  <policy_env_or_python> <eval_env_conda_env>

# Runnable CSU-AI-05 example
bash eval.sh RoboDojo stack_bowls CSU-AI-05 arx_x5 joint 0 0 0 \
  "$G05_PYTHON" RoboDojo
```

For simulator evaluation, use the same entrypoint from a RoboDojo workspace and set the evaluator environment explicitly:

```bash
cd XPolicyLab/policy/CSU-AI-05
unset EVAL_ENV_TYPE
export G05_PYTHON=/path/to/python3.10
export G05_CKPT_PATH=/path/to/CSU-AI-05/checkpoints/CSU_AI_05.pt
export ROBODOJO_G05_ACTION_SOURCE=fm
bash eval.sh RoboDojo stack_bowls CSU-AI-05 arx_x5 joint 0 0 0 \
  "$G05_PYTHON" RoboDojo
```

## Configuration

### Reproduction defaults

- Supported values: `bench_name=RoboDojo`, `env_cfg_type=arx_x5`, `action_type=joint`.
- Keep `action_source=fm`, `inference_batch_size=8`, `obs_offsets=[0]`, and `vision_attn_backend=sdpa`, which are the stable `deploy.yml` defaults used by the submitted evaluation.
- Use the bundled runtime under `policy/CSU-AI-05/G05` and the checkpoint bundle verified by `download_checkpoint.sh`.
- `XPOLICYLAB_BENCH_ROOT` points to the RoboDojo/RoboTwin checkout when XPolicyLab is not nested directly under that benchmark repository.
- `XPOLICYLAB_CLIENT_PYTHON` supplies an environment `bin/python` for minimal non-interactive shells; it defaults to `G05_PYTHON` when needed.
- `XPOLICYLAB_CONDA_EXE` supplies `bin/conda` when a non-interactive evaluator shell does not include conda in `PATH`.

### Compatibility overrides

The following existing variable names are consumed by the bundled runtime and are retained so that the tested launch path remains compatible. Leave them unset unless intentionally testing a different configuration:

- `ROBODOJO_G05_ACTION_SOURCE` overrides the FM/AR route.
- `ROBODOJO_G05_INFERENCE_BATCH_SIZE` overrides model-side micro-batching.
- `ROBODOJO_G05_OBS_OFFSETS` accepts comma-separated frame offsets.
- `G05_VISION_ATTN_BACKEND` overrides the vision attention backend.
- `G05_ROOT` overrides the bundled runtime source path.
- `G05_ACTION_TOKENIZER_PATH`, `G05_MIXED_STATS_PATH`, and `G05_HF_PROCESSOR_PATH` override bundle sidecars only for compatibility testing.

## Notes

`deploy.py` intentionally differs from `demo_policy/deploy.py`: it keeps frame-level history on the environment side so temporal checkpoints receive training-aligned sparse observations. The bundled runtime and its license, NOTICE, third-party notices, and modification notices are part of the submitted source state. Replacing that runtime or overriding its reproduction defaults may change results. Model checkpoints and Python environments are not committed to this repository.
