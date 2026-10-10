# Liber_0_lite

**Contributor:** LiberAI | **Paper:** Pending | **arXiv:** Pending | **Original code:** [liber0/](liber0/)

RoboDojo evaluation policy for `arx_x5` with joint actions.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

Inference source is included. Use Python 3.10 and a CUDA 13-compatible NVIDIA
driver. Install into a dedicated environment; keep the simulator environment
separate. Component notices: [NOTICE.md](NOTICE.md).
`/path/to/env` must be an existing Python environment with `bin/python` and pip;
the installer installs packages but does not create that environment.

External dependencies:

| Dependency | Required for |
| --- | --- |
| Linux, NVIDIA GPU/driver, Python 3.10, PyTorch 2.11.0 (CUDA 13), torchvision 0.26.0 | Policy inference; installed Python package versions are pinned by `install.sh` and [requirements.txt](requirements.txt). |
| Checkpoint and base model assets | Downloaded separately into `checkpoint/` and `assets/`; requires network access. Set `LIBER0_MODEL_PATH` to the assets directory. |
| XPolicyLab and RoboDojo `env_cfg/` | Required for both debug and simulation. Place `XPolicyLab/` beside `env_cfg/`, including `arx_x5.yml` and `robot/_robot_info.json`. |
| RoboDojo simulator, scene/robot assets, Isaac Sim and its Conda environment | Simulation only; install separately using the benchmark's environment instructions. Not installed by this policy. |

The inference engine and backend source are bundled; no external runtime/source
checkout or FlashAttention installation is required.

```bash
cd XPolicyLab/policy/Liber_0_lite
bash install.sh /path/to/env
export PATH=/path/to/env/bin:$PATH
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
python download_checkpoint.py --destination ./checkpoint --assets-dir ./assets
export LIBER0_MODEL_PATH="$PWD/assets"
```

Checkpoint: [Liber0-Lite-Robodojo](https://huggingface.co/LiberAI/Liber0-Lite-Robodojo).

The downloader pins both repositories and verifies checkpoint checksums.
Downloads total approximately 53 GB. The downloader does not use proxy environment variables.

## Data Processing

Not required for evaluation.

## Training

This adapter is evaluation-only. Training code will be released with the paper.

## Evaluation

The checkpoint directory must contain `model.pt`, `config.yaml` and
`dataset_stats.json`.

```bash
EVAL_ENV_TYPE=debug bash eval.sh RoboDojo stack_bowls "$PWD/checkpoint" \
  arx_x5 joint 0 0 0 /path/to/env /path/to/env

EVAL_ENV_TYPE=sim bash eval.sh RoboDojo stack_bowls "$PWD/checkpoint" \
  arx_x5 joint 0 0 0 /path/to/env RoboDojo
```

Repeat debug mode with `DEBUG_OBS_ENCODED=1` for encoded observations. Debug mode
accepts an environment path as argument 10; simulation uses the simulator's
Conda environment name. The parent workspace must contain RoboDojo `env_cfg/`;
see the shared installation guide.

### Batch inference

Set `eval_batch: true` in `deploy.yml` to use the same commands with batched
observations. The default remains `false`. Image preprocessing is per item;
the vision encoder and denoising decoder run on the combined batch, including
different-length instructions. Batch size is determined by the active observations.

Each observation must carry a unique, stable `env_idx`. The policy preserves
each environment's first-frame cue and replan counter; outputs follow the requested
`env_idx_list` order. The seed remains `run_seed + per_environment_replan_count`.
`reset()` clears all state; use `reset_envs(env_idx_list)` before reusing selected
environment IDs for new episodes (WebSocket: `call(func_name="reset_envs", obs=ids)`).
Use one coordinating client per policy server.

Batch size 1 retains the single-item inference path. Larger batches can differ
numerically from separate BF16 calls; task success equivalence requires a separate
simulator evaluation. Batch inference uses more working memory; no automatic
serial fallback is applied on memory errors.
