# REAL_RSI

**Contributor:** PKU DAGroup | **Paper:** PyRUA-REAL | **arXiv:** [2611.06768](https://arxiv.org/abs/2611.06768) | **Original code:** https://github.com/DAGroup-PKU/PyRUA-Lean/tree/REAL

Evaluation-only controllers for 42 RoboDojo base tasks and 12 official `_random`
variants. 

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

Use this checkout inside an installed RoboDojo workspace with the official
`env_cfg/arx_x5.yml` in its parent directory. The reference environment is Linux,
Python 3.11 and an NVIDIA GPU compatible with CUDA 12.8. A separate simulator
environment is required.

```bash
conda create -n real-rsi python=3.11 -y
conda activate real-rsi
cd XPolicyLab/policy/REAL_RSI
bash install.sh
python download_models.py
```

`constraints.txt` pins the tested transitive dependency resolution, including the
XPolicyLab server dependencies. `install.sh` ends with `pip check`.

The downloader retrieves approximately 4.8 GB of public vision weights using pinned
Hugging Face revisions and verifies their hashes. This includes SAM3 (`facebook/sam3`,
3.45 GB), a gated repository: accept its licence on the model page and run
`hf auth login` before downloading. Set `REAL_RSI_MODEL_DIR` to choose
another model directory; the default is `model_assets/`. `HF_ENDPOINT` supports a
compatible mirror. Inference is offline. Install the simulator separately using
RoboDojo's instructions.

Each task's program is listed in `routes.json`; programs are in `pyrua_programs/.` The 12 _random variants reuse the program of their standard task.

## Data Processing

Not applicable. 

## Training

Eval-only. Full frozen inference source is included; there is no REAL_RSI training
pipeline or planned training release. Public pretrained model versions are specified
in `models.lock.json`. There is no `train.sh`.

## Evaluation

Use `env_cfg_type=arx_x5` and `ckpt_name=.` for every task. The `action_type=joint` for every task.

The standard interface is:

```bash
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_env_or_python> <eval_env_conda_env>

# example
bash eval.sh RoboDojo stack_blocks . arx_x5 joint 0 0 0 real-rsi RoboDojo
```

The `action_type` argument is also part of the result directory name
(`<seed>_ckpt_name=.,action_type=<action_type>`).

Use the required benchmark seed in the sixth argument. Set `EVAL_ENV_TYPE=sim` or
leave it unset for simulation; unset `EVAL_NUM` to retain native episode counts.
The ninth argument accepts a conda environment name, a virtualenv directory, or
an executable Python path. Only sequential evaluation is supported
(`eval_batch: false`).

## Configuration

- `camera_calib`: `camera_calib.json` contains camera intrinsics, the fixed head
  transform and wrist mounts. Wrist transforms use observed end-effector poses.
  Calibration supports the official `arx_x5` camera configuration.
- `policy_python`: optional controller interpreter; falls back to
  `REAL_RSI_POLICY_PYTHON`, then the server interpreter.
- `reset_timeout_s: 240`, `act_timeout_s: 120`: controller request deadlines.
- `hold_chunk: 50`: hold actions after normal policy termination.
- `strict_errors: true`: surface controller failures to the evaluator. Setting it
  false enables the original hold-on-error behavior for diagnosis.
