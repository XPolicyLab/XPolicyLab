# REAL_RSI

**Contributor:** PUK DAGroup | **Paper:** REAL_RSI | **arXiv:** Not provided | **Original code:** Frozen programs included here

Evaluation-only controllers for 42 RoboDojo base tasks and 12 official `_random`
variants. All 54 task names use the `REAL_RSI` adapter; `routes.json` maps each task
name to its program:

- `insert_key`: a frozen LILO controller, `scripts/insert_key/policy.py`.
- the other 53 task names (`engine: pyrua`): a fixed PyRUA-Lean program,
  `pyrua_programs/<task>.py`, from PyRUA-RSI
  ([PyRUA-Lean](https://github.com/DAGroup-PKU/PyRUA-Lean) `integrations/xpolicylab/PyRUA_RSI`
  at `20dbb21e`, Apache-2.0; see `pyrua_programs/LICENSE` and `NOTICE`). No language
  model runs at evaluation time; SAM3 is the only model these routes load.

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

The downloader retrieves approximately 10.8 GB of public vision weights using pinned
Hugging Face revisions and verifies their hashes. This includes SAM3 (`facebook/sam3`,
3.45 GB), a gated repository: accept its licence on the model page and run
`hf auth login` before downloading. Set `REAL_RSI_MODEL_DIR` to choose
another model directory; the default is `model_assets/`. `HF_ENDPOINT` supports a
compatible mirror. Inference is offline. Install the simulator separately using
RoboDojo's instructions.

## Data Processing

Not applicable. Frozen controllers, reference images and geometric assets are
included in the task subdirectories under `scripts/`. No dataset conversion or `process_data.sh` is needed.

## Training

Eval-only. Full frozen inference source is included; there is no REAL_RSI training
pipeline or planned training release. Public pretrained model versions are specified
in `models.lock.json`. There is no `train.sh`.

## Evaluation

Use `env_cfg_type=arx_x5`, `ckpt_name=.` and `action_type=ee` for every task.
RoboDojo infers each action's type from its keys: the PyRUA routes send native joint
targets, the frozen route sends end-effector poses, both under the same arguments.
The standard interface is:

```bash
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_env_or_python> <eval_env_conda_env>

# Example
bash eval.sh RoboDojo stack_blocks . arx_x5 ee 0 0 0 real-rsi RoboDojo

```

Use the required benchmark seed in the sixth argument. Set `EVAL_ENV_TYPE=sim` or
leave it unset for simulation; unset `EVAL_NUM` to retain native episode counts.
The ninth argument accepts a conda environment name, a virtualenv directory, or
an executable Python path. Only sequential evaluation is supported
(`eval_batch: false`).

For the official layout sets, use seed `1` for set1 and seed `2` for set2, leaving
`EVAL_NUM` unset. RoboDojo determines each task's native 25/50 valid-episode count
and handles unstable layouts. Do not treat an exit code alone as a complete
evaluation: verify the native `_result.json` episode count and check the log for
policy exceptions.

`routes.json` maps each official task name to its program and the `action_type`
argument it expects. Separate Standard/Random implementations are selected by their
official task names. For the frozen route each episode runs in a fresh child
process; the program constructor retains the original runtime's fixed internal seed
of zero. For PyRUA routes the policy server runs each episode in PyRUA-Lean's
runtime and sandbox: a local endpoint answers the runtime's first request with the
task's whole program as one `python` call and later requests with `finish`, and
`deploy.py` uses PyRUA-Lean's client loop for these task names (it asks again while
the program computes, ends the episode when the program finishes, and reports the
verdict). The evaluation seed controls benchmark layouts in both cases.

The shared checkpoint resolver uses this adapter directory as its root.
`ckpt_name=.` selects the bundled task folders under `scripts/`. An explicit checkpoint
root must contain `routes.json` and the same task-directory structure.

## Configuration

- `camera_calib`: `camera_calib.json` contains camera intrinsics, the fixed head
  transform and wrist mounts. Wrist transforms use observed end-effector poses.
  Calibration supports the official `arx_x5` camera configuration.
- `depth_provider`: the frozen route fixes `mono` or `none`. `mono` estimates depth from
  RGB using Depth Anything V2 Large. `mono_model_dir` overrides its model location;
  the route takes precedence over the deployment depth default.
- `policy_python`: optional controller interpreter; falls back to
  `REAL_RSI_POLICY_PYTHON`, then the server interpreter.
- `reset_timeout_s: 240`, `act_timeout_s: 120`: controller request deadlines.
- `hold_chunk: 50`: hold actions after normal policy termination.
- `strict_errors: true`: surface controller failures to the evaluator. Setting it
  false enables the original hold-on-error behavior for diagnosis.
- `pyrua`: PyRUA-Lean episode settings for the PyRUA routes, as in its PyRUA_RSI
  adapter. Each episode's program output and tool calls are kept under
  `pyrua.result_dir` (`PYRUALEAN_RESULT_DIR` overrides it). `SAM3_CHECKPOINT_PATH`
  overrides the SAM3 checkpoint (default `<model dir>/facebook__sam3/sam3.pt`).

`LILO_MODELS` is accepted as a fallback for `REAL_RSI_MODEL_DIR`. Robot meshes,
URDFs, reference images and numerical galleries are runtime assets. Vendored
robot assets retain their original license files; the PyRUA programs carry
PyRUA-Lean's Apache-2.0 `LICENSE` and `NOTICE` in `pyrua_programs/`.
