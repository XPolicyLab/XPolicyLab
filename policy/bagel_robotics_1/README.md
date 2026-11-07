# Bagel-Robotics-1

**Contributor:** Bagel Labs | **Paper:** Technical report in preparation | **arXiv:** Not yet available | **Original code:** [Bagel-Robotics-1 implementation](.)

Bagel-Robotics-1 combines mask-based structured pruning, recovery training and selection of important task segments to produce an inference-friendly compact WAM. VPP2 is the base model used to validate the approach; reused inference components are bundled in `runtime/` with [source notices](THIRD_PARTY_NOTICES.md). This adapter supports RoboDojo ARX X5 end-effector inference.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](https://github.com/XPolicyLab/XPolicyLab/blob/main/README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

Install RoboDojo and simulator assets using the [official guide](https://robodojo-benchmark.com/doc/usage/install-and-download/). Place XPolicyLab in that workspace, alongside `env_cfg/`.

Until merged, fetch [PR #234](https://github.com/XPolicyLab/XPolicyLab/pull/234) in a dedicated checkout. Replace the example path and keep the printed commit with evaluation records; use the application's exact commit when one is specified.

```bash
BAGEL_XPL_ROOT=/absolute/path/to/XPolicyLab
git -C "$BAGEL_XPL_ROOT" fetch https://github.com/XPolicyLab/XPolicyLab.git refs/pull/234/head
git -C "$BAGEL_XPL_ROOT" checkout --detach FETCH_HEAD
git -C "$BAGEL_XPL_ROOT" rev-parse HEAD

BAGEL_POLICY_DIR="$BAGEL_XPL_ROOT/policy/bagel_robotics_1"
BAGEL_POLICY_ENV="$BAGEL_POLICY_DIR/.venv"
cd "$BAGEL_POLICY_DIR"
PYTHON_BIN=python3.10 bash install.sh "$BAGEL_POLICY_ENV"
```

Keep these variables in the same shell. The installer sets up the standalone runtime, XPolicyLab and PyTorch 2.11 / torchvision 0.26 CUDA 13.0 wheels; `TORCH_INDEX_URL` overrides the wheel index. Tested setup: Python 3.10.21, PyTorch 2.11.0+cu130, RTX PRO 6000 Blackwell (96 GB), 128 GiB host RAM and eight CPUs.

The [checkpoint repository](https://huggingface.co/bageldotcom/Bagel-Robotics-1) is private. Obtain the dedicated repository-only read token through the agreed evaluator channel, then download and verify all assets:

```bash
set +x
read -r -s -p 'Hugging Face read token: ' HF_TOKEN
export HF_TOKEN
PYTHON_BIN="$BAGEL_POLICY_ENV/bin/python" bash download_checkpoint.sh
unset HF_TOKEN
```

Allow approximately **45.7 GB** for checkpoint and encoder assets, plus cache and installation space. Every file is hash-verified. Optional `--metadata-only` checks four small JSON files; the full download is required for inference.

## Data Processing

Inference receives decoded RGB observations from XPolicyLab. No conversion is required, so `process_data.sh` is omitted for this eval-only submission. Recovery training uses selected important task segments from the RoboDojo EE16 dataset rather than the complete dataset; the selection recipe will accompany the technical report.

## Training

This submission is **eval-only**; `train.sh` is omitted. The technical report and training, pruning and export code are planned for **November 20, 2026**. Maintainer agreement on the eval-only submission is requested.

## Evaluation

Supported: `bench_name=RoboDojo`, `env_cfg_type=arx_x5`, `action_type=ee`, `ckpt_name=cp1000`.

The launcher uses the standard ten arguments:

```text
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_env_or_venv_path> <simulator_conda_env>
```

After installation and download, run both debug transports. On a dedicated workstation choose an available GPU; on a cluster use the allocated GPU and preserve scheduler visibility.

```bash
cd "$BAGEL_POLICY_DIR"
BAGEL_GPU_ID=0
EVAL_ENV_TYPE=debug DEBUG_OBS_ENCODED=0 DEBUG_EVAL_EPISODES=1 bash eval.sh \
  RoboDojo stack_blocks cp1000 arx_x5 ee 1 \
  "$BAGEL_GPU_ID" "$BAGEL_GPU_ID" "$BAGEL_POLICY_ENV" "$BAGEL_POLICY_ENV"

EVAL_ENV_TYPE=debug DEBUG_OBS_ENCODED=1 DEBUG_EVAL_EPISODES=1 bash eval.sh \
  RoboDojo stack_blocks cp1000 arx_x5 ee 1 \
  "$BAGEL_GPU_ID" "$BAGEL_GPU_ID" "$BAGEL_POLICY_ENV" "$BAGEL_POLICY_ENV"
```

Both must exit successfully with `[TestEnv] Episode finished` and `[MAIN] eval finished`. A venv path in argument 10 is supported for debug; simulation uses a conda environment.

Example native simulator check, seed 1 and original layouts 0/1:

```bash
BAGEL_SIM_ROOT=/absolute/path/to/RoboDojo
BAGEL_SIM_ENV=your_simulator_conda_env
cd "$BAGEL_SIM_ROOT"
EVAL_ENV_TYPE=sim bash scripts/robodojo.sh eval \
  --policy-dir "$BAGEL_POLICY_DIR" \
  --task insert_tubes --ckpt cp1000 --env-cfg arx_x5 --action-type ee \
  --seed 1 --eval-num 2 --policy-env "$BAGEL_POLICY_ENV" \
  --eval-env "$BAGEL_SIM_ENV" \
  --policy-gpu "$BAGEL_GPU_ID" --env-gpu "$BAGEL_GPU_ID"
```

Recorded integration results: `insert_tubes` **2/2 successes, 100/100 partial**; `store_laptop_and_headphones` **1/2 successes, 60/100 partial**. See [VALIDATION.md](VALIDATION.md) for evidence and [REEVALUATION.md](REEVALUATION.md) for complete task, suite and three-seed commands. Preserve native result files and startup failures for review.

## Model Assets

Checkpoint: [bageldotcom/Bagel-Robotics-1](https://huggingface.co/bageldotcom/Bagel-Robotics-1), pinned revision `2651631c5d08819088475e8778676f5cf6d9afcf`.

The downloader places the six checkpoint files in `checkpoints/cp1000` and separately pinned public encoder/tokenizer assets in `checkpoints/Wan2.1-I2V-14B-480P`. [artifact-manifest.json](artifact-manifest.json) records their revisions and hashes.

To reuse existing assets, set the following paths and verify them:

```bash
export BAGEL_CP1000_BUNDLE=/absolute/path/to/update_001000
export BAGEL_CP1000_WAN_ROOT=/absolute/path/to/Wan2.1-I2V-14B-480P
"$BAGEL_POLICY_ENV/bin/python" "$BAGEL_POLICY_DIR/download_checkpoint.py" \
  --verify-only --bundle "$BAGEL_CP1000_BUNDLE" --wan-root "$BAGEL_CP1000_WAN_ROOT"
```

These variables override `bundle_path` and `wan_model_dir` in `deploy.yml`; otherwise checkpoints resolve through XPolicyLab's shared resolver using `ckpt_name`.

## Configuration

Use the same checkpoint and settings across tasks: BF16, action horizon 32, replan every 16 steps, ten Euler steps, sigma shift 1 and 90/10 temporal blending. `eval_batch: false` supports one environment per process. The custom deployment loop records every executed observation for history, resets between episodes and stops on termination.

The server launcher applies the recorded deterministic backend settings. Direct API users must set `PYTHONHASHSEED=1` and `CUBLAS_WORKSPACE_CONFIG=:4096:8` before starting Python. Optional `BAGEL_VALIDATION_TRACE` recording is documented in [VALIDATION.md](VALIDATION.md). The compact video expert has **12.468B parameters, 23.95% fewer** than the validation base; self-attention heads and Action2B are retained.

## Notes

Reported five-suite research results: **31.58% success / 38.55 mean partial out of 100**, equally weighted across suites. [EVALUATION_RESULTS.md](EVALUATION_RESULTS.md) contains suite/task tables, source details and the three replacement scenes plus one repaired Memory variant. Official three-seed evaluation is pending.
