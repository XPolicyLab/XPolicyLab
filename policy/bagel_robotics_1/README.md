# Bagel-Robotics-1

**Contributor:** Bagel Labs | **Paper:** Technical report in preparation | **arXiv:** Not yet available | **Original code:** [VPP2](https://github.com/roboterax/video-prediction-policy-2)

Evaluation-only adapter for Bagel-Robotics-1 checkpoint 1000, combining mask-based
structured pruning and recovery with a dataset subset focused on critically
important segments of each task. We evaluate the approach using VPP2 as the base
model; `runtime/` contains an adapted VPP2 inference runtime and compact loader,
with upstream attribution retained. The video expert has 12,468,409,664 parameters,
23.95% fewer than the recorded 16,394,878,784-parameter parent video expert,
while Action2B remains unchanged.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](https://github.com/XPolicyLab/XPolicyLab/blob/main/README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

Install this directory under `XPolicyLab/policy/bagel_robotics_1` in a RoboDojo
workspace. `env_cfg/` belongs next to XPolicyLab, not inside the policy. Install
RoboDojo and its simulator/assets separately using the
[official installation guide](https://robodojo-benchmark.com/doc/usage/install-and-download/).

Until this adapter is merged, fetch [PR #234](https://github.com/XPolicyLab/XPolicyLab/pull/234)
into a dedicated reviewer checkout of XPolicyLab in that workspace. Replace the
example path, and keep the printed commit SHA with the evaluation records:

```bash
git -C /absolute/path/to/XPolicyLab fetch https://github.com/XPolicyLab/XPolicyLab.git refs/pull/234/head
git -C /absolute/path/to/XPolicyLab checkout --detach FETCH_HEAD
git -C /absolute/path/to/XPolicyLab rev-parse HEAD
```

For an evaluation application that specifies a commit, check out that exact SHA
after fetching. The upstream base commit alone does not contain this adapter.

**1. Set paths and install the policy environment.** Replace the example checkout
path below. Run the following sections in the same shell so these variables remain
available. Use **Python 3.10**, the version used for the GPU qualification; set
`PYTHON_BIN` to its executable if it is not named `python3.10`.

```bash
BAGEL_XPL_ROOT=/absolute/path/to/XPolicyLab
BAGEL_POLICY_DIR="$BAGEL_XPL_ROOT/policy/bagel_robotics_1"
BAGEL_POLICY_ENV="$BAGEL_POLICY_DIR/.venv"
cd "$BAGEL_POLICY_DIR"
PYTHON_BIN=python3.10 bash install.sh "$BAGEL_POLICY_ENV"
```

The installer creates a venv, installs PyTorch 2.11 / torchvision 0.26 CUDA 13.0
wheels and installs the isolated runtime plus XPolicyLab. `TORCH_INDEX_URL` changes
the PyTorch wheel index. GPU qualification used Python 3.10.21 and PyTorch
2.11.0+cu130 on an RTX PRO 6000 Blackwell; installation on another stack does not
establish numerical equivalence. The recorded GPU has 96 GB VRAM; a lower
minimum has not been qualified. The isolated validation jobs reserved 128 GiB
host RAM and eight CPUs; these are tested allocations, not measured minima.

**2. Download and verify before evaluation.** The public inference release is
prepared at [bageldotcom/Bagel-Robotics-1](https://huggingface.co/bageldotcom/Bagel-Robotics-1).
The repository is currently private. Evaluators need a dedicated fine-grained
token granting read access only to this repository, supplied separately through
an agreed private channel. The dedicated credential and repository-only read scope are verified; assigned
evaluator handoff remains pending.
Set `HF_TOKEN` securely before downloading (for example, `read -r -s HF_TOKEN`
followed by `export HF_TOKEN`; use `set +x` first and `unset HF_TOKEN` afterward).
Do not place the token in scripts, the PR, logs, or the application text.
Public release is a separate leaderboard milestone; once public, no token is needed.

```bash
PYTHON_BIN="$BAGEL_POLICY_ENV/bin/python" bash download_checkpoint.sh
```

The download is approximately **45.7 GB** before cache/installation overhead.
To check pinned metadata before downloading tensors, run
`PYTHON_BIN="$BAGEL_POLICY_ENV/bin/python" bash download_checkpoint.sh --metadata-only`
first. This checks four small JSON files only; run the full downloader above
before evaluation. The downloader verifies every checkpoint and auxiliary-asset hash. For existing
files, follow [Model Assets](#model-assets) instead. Continue to
[Evaluation](#evaluation) only after the download or verification succeeds.

## Data Processing

No conversion is needed for inference. A key component of our approach is to use
a dataset subset containing critically important segments from each task instead
of the complete training dataset. The research recovery run used a curated,
task-balanced 30,000-window RoboDojo EE16 subset.

Training reproduction is outside this initial adapter. `process_data.sh` is
intentionally omitted under the proposed eval-only exception. Inputs are decoded
RGB arrays from XPolicyLab's shared server. The segment-selection procedure and
recovery recipe will be described in the technical report.

## Training

`train.sh` is intentionally omitted. Training, mask learning and export are not included in this adapter; the
technical report and training,
pruning, and export code are planned for release on **November 20, 2026**.
The proposed eval-only integration and release schedule remain subject to maintainer
coordination; this preparation does not claim that the exception is approved.

## Evaluation

Supported: `bench_name=RoboDojo`, `env_cfg_type=arx_x5`, `action_type=ee`.
Use the same checkpoint and settings for every task. Complete Installation first.

**3. Check both debug transports.** These load the real model but use synthetic
observations. Run from the policy directory with the environment defined above.
On a dedicated workstation, choose a GPU with sufficient free memory:

```bash
cd "$BAGEL_POLICY_DIR"
BAGEL_GPU_ID=0
```

On a scheduled cluster, run inside an allocation and set `BAGEL_GPU_ID` to its
assigned single GPU instead; preserve the scheduler's `CUDA_VISIBLE_DEVICES`.
For a one-GPU allocation that exports this variable:

```bash
BAGEL_GPU_ID="${CUDA_VISIBLE_DEVICES:?Run inside a one-GPU allocation}"
```

Run both commands sequentially:

```bash
EVAL_ENV_TYPE=debug DEBUG_OBS_ENCODED=0 DEBUG_EVAL_EPISODES=1 bash eval.sh \
  RoboDojo stack_blocks cp1000 arx_x5 ee 1 \
  "$BAGEL_GPU_ID" "$BAGEL_GPU_ID" "$BAGEL_POLICY_ENV" "$BAGEL_POLICY_ENV"

EVAL_ENV_TYPE=debug DEBUG_OBS_ENCODED=1 DEBUG_EVAL_EPISODES=1 bash eval.sh \
  RoboDojo stack_blocks cp1000 arx_x5 ee 1 \
  "$BAGEL_GPU_ID" "$BAGEL_GPU_ID" "$BAGEL_POLICY_ENV" "$BAGEL_POLICY_ENV"
```

Each must finish with `[TestEnv] Episode finished` and `[MAIN] eval finished`,
exit zero, and contain no traceback. Debug completion is an interface check,
not a task success result. A venv path in argument 10 is supported for debug
only; simulation uses the official conda client launcher.

**4. Run a small native simulator check.** Set the installed RoboDojo path and
simulator conda environment name. The following command matches the tested
`insert_tubes` smoke: seed 1, two episodes, original layouts 0 and 1. It uses
the same GPU selected above for the policy and simulator.

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

For the second tested task, repeat with `--task store_laptop_and_headphones`.
Preserve native `_result.json` files under RoboDojo's `eval_result/` and record
source revisions and any startup errors. See [VALIDATION.md](VALIDATION.md) for
measured outcomes and the shared-installation startup workaround used in our
qualification. Keep these two-episode checks separate from full benchmark runs.

For a full task, suite, or three-seed run, follow [Reevaluation](REEVALUATION.md).
It uses the official runner, native episode counts and original layouts.

The direct launcher retains XPolicyLab's ten-argument contract (run from the
policy directory):

```text
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_env_or_venv_path> <simulator_conda_env>
```

The custom `deploy.py` records every executed observation, calls reset at episode
boundaries and stops sending actions when the environment terminates. This is
necessary for the checkpoint's history conditioning. Batched execution is disabled;
the batch methods only support a single environment and reject larger batches.

## Model Assets

The Installation quickstart authenticates and downloads before evaluation.
The script downloads the six exact bundle files into `checkpoints/cp1000` and
the pinned public VAE/CLIP/UMT5/tokenizer assets into
`checkpoints/Wan2.1-I2V-14B-480P`. Bundle repository:
[bageldotcom/Bagel-Robotics-1](https://huggingface.co/bageldotcom/Bagel-Robotics-1),
revision `2651631c5d08819088475e8778676f5cf6d9afcf` (**public release prepared; visibility change pending**).
Access must be arranged for evaluators. No token is included.

For existing artifacts, set `BAGEL_CP1000_BUNDLE` and `BAGEL_CP1000_WAN_ROOT`.
`bundle_path` and `wan_model_dir` in `deploy.yml` are alternatives. The environment
variables take precedence; otherwise the shared XPolicyLab checkpoint resolver
uses `ckpt_name`. `artifact-manifest.json` pins every file and encoder revision.

```bash
export BAGEL_CP1000_BUNDLE=/absolute/path/to/update_001000
export BAGEL_CP1000_WAN_ROOT=/absolute/path/to/Wan2.1-I2V-14B-480P
"$BAGEL_POLICY_ENV/bin/python" "$BAGEL_POLICY_DIR/download_checkpoint.py" \
  --verify-only --bundle "$BAGEL_CP1000_BUNDLE" --wan-root "$BAGEL_CP1000_WAN_ROOT"
```

Full file hashes are checked before loading. `--metadata-only` downloads only
small checkpoint metadata for CPU architecture checks and does not qualify weights.

## Configuration

BF16, action horizon 32, execute/replan 16, ten Euler steps, sigma shift 1,
90/10 temporal blending, and `eval_batch: false` are fixed and validated.
`seed` is supplied by the evaluator; task names never select alternate models.
The server launcher also preserves the recorded deterministic backend controls
(`PYTHONHASHSEED=1`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, deterministic algorithms,
and the recorded TF32/matmul settings). Direct API users must set those startup
environment variables before launching Python. The checkpoint loader accepts only the compact channel architecture and loads
weights strictly. There is no teacher, mask-training module or dense fallback.

Optional validation: setting `BAGEL_VALIDATION_TRACE` to a new absolute output
directory enables lossless capture around the official policy server. It records
the first 209 observations per episode and fingerprints preprocessing, history
and actions. It changes no predictions or scoring and is disabled by default.
See [VALIDATION.md](VALIDATION.md) for capture/replay scope.

## Notes

The approach combines two key components:

- **Mask-based structured pruning and recovery:** produce physically smaller,
  nonuniform FFN and cross-attention layers.
- **Task-focused data selection:** use a subset containing critically important
  segments from each task for recovery, instead of the complete training dataset.

The approach is intended for compatible WAM backbones; VPP2 is the base model used
for this evaluation. The technical report will describe pruning, segment selection
and recovery in detail.

The recorded research results cover all five suites:

| Suite | Successful / reported outcomes | Success rate | Mean partial /100 |
|---|---:|---:|---:|
| Precision | 126/400 | 31.50% | 40.43 |
| Memory† | 173/300 | 57.67% | 58.00 |
| Long Horizon‡ | 137/400 | 34.25% | 53.00 |
| Generalization | 183/600 | 30.50% | 36.82 |
| Open§ | 16/400 | 4.00% | 4.48 |
| **Equal-weight five-suite average** | **635/2,100 total; not the averaging formula** | **31.58%** | **38.55** |

Memory includes one repaired variant, Long Horizon two labeled replacements, and Open one labeled replacement. Generalization combines standard and randomized episodes. Memory and nine Generalization tasks use user-supplied summaries; see [EVALUATION_RESULTS.md](EVALUATION_RESULTS.md) for complete counts, task results, source limitations and original-only figures. These are research-runner results, not official leaderboard scores or proof of packaged-adapter equivalence.

The learned masks have been exported into nonuniform per-layer FFN/cross-attention
reductions. Action2B and all video self-attention heads are retained.

Package validation so far: 15 CPU tests, real compact-video strict tensor loading, all checkpoint/encoder hashes, static checks, isolated wheel processor loading, and clean-checkout installation. Both real-checkpoint GPU debug transports passed in job 53609. Four native simulator episodes completed with valid results, and all eight action/history/reset/backend comparison checks passed with zero recorded differences. See [VALIDATION.md](VALIDATION.md) for the measured task scores and test scope.
The repository-only read credential is verified; evaluator assignment and private handoff remain pending. The technical report and training/pruning/export code are planned for **November 20, 2026**. Full-suite
coverage is a separate leaderboard-verification milestone. See [validation evidence](VALIDATION.md) and [historical task results](EVALUATION_RESULTS.md).
