# WorldLA

**Contributor:** WorldLA Team | **Paper:** WorldLA: Bridging World Action Models and Efficient VLAs via Interaction Distillation and Coherent Replanning | **arXiv:** Not available | **Original code:** [18643638196xhx/WorldLA](https://github.com/18643638196xhx/WorldLA)

WorldLA transfers cached Fast-WAM action-to-visual routing relations to a
pi0.5 policy and fuses time-aligned old-plan tails with new-plan heads.
This adapter supports the released RoboTwin Aloha-AgileX checkpoint with
`env_cfg_type=arx_x5`, `action_type=joint`. No upstream model code or weights
are vendored, and Fast-WAM does not run during cached-target training or deployment.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

Use Linux, Python 3.11 and a CUDA-12-compatible NVIDIA driver. Keep simulator
and policy environments separate. Run from this directory:

```bash
# bash install.sh <new-venv-directory> <runtime|train>
bash install.sh "$PWD/.venv" runtime
source .venv/bin/activate
```

`install.sh` refuses an existing environment. It installs the upstream pinned
runtime recipe and WorldLA commit `d84647cb4f412af1a9b852146073e42b4e3b76ea`,
plus the official XPolicyLab WebSocket dependencies. For training use a **new**
environment with `bash install.sh "$PWD/.venv-train" train` instead.
`WORLDLA_SOURCE_DIR` optionally selects a dedicated clean upstream checkout;
the installer checks out the pinned commit there. Do not point it at active work.
Fresh-environment installation has not been certified on every CUDA stack.

The policy environment selector in the standard eval/server commands accepts
a Conda name, absolute environment directory, or absolute Python executable.
Use the same selector convention for the simulator/debug-client environment.

## Data Processing

WorldLA uses **LeRobot v2.1**, dataset ID
`robotwin_official50_demo_clean_v21`. Its three image keys are
`observation.images.cam_high`, `observation.images.cam_left_wrist`, and
`observation.images.cam_right_wrist`; state/action are 14-D ordered
`left arm, left gripper, right arm, right gripper`.
An arbitrary official-converter export is not interchangeable with the released
route cache: task/episode/frame order, seen prompts and state/action fingerprints
must match. The released strict converter uses **only**
`XPolicyLab.utils.process_data.decode_image_bit` for official trajectory images.
It writes the specified LeRobot keys and verifies the cached anchor alignment.
No caller-side channel swaps are applied.

First obtain the bundle and follow the upstream
[training-material preparation](https://github.com/18643638196xhx/WorldLA#4-download-the-training-bundle)
for archive checksum verification and extraction into a **new** directory.
These 20,000 cached frame anchors are derived supervision, not extra demonstrations.
The original robot dataset must be downloaded separately: 50 `demo_clean`
demonstrations per task, not the randomized-500 archives.

```bash
bash download.sh training "$PWD/checkpoints/worldla-100k-decay"
export WORLDLA_REPRO_ROOT=/absolute/path/to/unpacked/stage_routes_20k
export WORLDLA_RAW_ROOT=/absolute/path/to/robotwin_raw
export WORLDLA_LEROBOT_ROOT=/absolute/path/to/lerobot

# bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type>
bash process_data.sh RoboTwin cotrain arx_x5 joint
```

`ckpt_name` is the experiment label, not a per-task filter. The strict recipe
processes the complete released task list; it refuses incompatible data rather
than regenerating or reassigning teacher targets. Data conversion/training
requires the upstream training dependencies. The source dataset Hub revision
was not historically recorded, so the published numeric/prompt alignment audit
is mandatory; it is not an image-pixel hash audit.

## Training

Training is implemented upstream and invoked by `train.sh`, not an eval-only
stub. Obtain the **pretrained** JAX pi0.5 initialization from
`gs://openpi-assets/checkpoints/pi05_base` and verify its fingerprints with the
released bundle. Fine-tuned WorldLA full weights are **not** that initialization.
The upstream [training guide](https://github.com/18643638196xhx/WorldLA#6-download-initialization-and-set-training-paths)
documents base download, tokenizer and dataset preparation.

```bash
export WORLDLA_INIT_PARAMS=/absolute/path/to/pi05_base/params
export WORLDLA_TOKENIZER=/absolute/path/to/paligemma_tokenizer.model
# Keep WORLDLA_REPRO_ROOT and WORLDLA_LEROBOT_ROOT from Data Processing.

# bash train.sh <bench> <ckpt> <env_cfg> <action> <seed> <gpu_ids> [all|router|action]
bash train.sh RoboTwin cotrain arx_x5 joint 20260901 0,1,2,3,4,5,6,7 all
```

Only use idle GPUs. The main release uses a 60k-update router followed by 100k
action updates, global batch 32, FSDP=8. The action stage freezes the router,
updates LoRA/message adapter/gates, and uses the released warmup/plateau/cosine
schedule. `WORLDLA_FSDP_DEVICES` can override the GPU count while keeping the
batch divisible; this is not a separately verified full-budget reproduction.
`train.sh` checks action width with XPolicyLab's shell helper. It stops on a
failed or incomplete router and does not implicitly overwrite training outputs.

Default training output is
`checkpoints/RoboTwin-cotrain-arx_x5-joint-20260901/training/`; a new
`WORLDLA_OUTPUT_ROOT` may override it. Completed action training is exported
to that run's `original/` subdirectory for the normal checkpoint resolver.
Parameter-only warm starts reset optimizer state. Full-budget retraining was
not rerun for this adapter PR; earlier upstream four-update training tests
validate the training path, not convergence.

## Evaluation

Download the pinned, public, ungated checkpoint. No Hugging Face login is needed:

```bash
# bash download.sh <original|merged|training> [release-directory]
bash download.sh original "$PWD/checkpoints/worldla-100k-decay"

# bash eval.sh <bench> <task> <ckpt> <env_cfg> <action> <seed> \
#   <policy_gpu> <env_gpu> <policy_env_or_python> <eval_env_or_python>
EVAL_ENV_TYPE=debug bash eval.sh RoboTwin press_stapler worldla-100k-decay \
  arx_x5 joint 0 0 0 "$PWD/.venv" /absolute/path/to/client-env

# Repeat to exercise the official transport's encoded RGB input path.
DEBUG_OBS_ENCODED=1 EVAL_ENV_TYPE=debug bash eval.sh \
  RoboTwin press_stapler worldla-100k-decay arx_x5 joint 0 0 0 \
  "$PWD/.venv" /absolute/path/to/client-env

# Requires the native RoboTwin simulator, planners and assets in the parent workspace.
WORLDLA_TASK_CONFIG=demo_clean WORLDLA_TEST_NUM=100 EVAL_ENV_TYPE=sim \
  bash eval.sh RoboTwin press_stapler worldla-100k-decay arx_x5 joint 0 0 0 \
  "$PWD/.venv" /absolute/path/to/robotwin-env
```

Set `WORLDLA_TASK_CONFIG=demo_randomized` for unseen instructions; do not
remove clutter, change success rules, skip hard seeds or call setup crashes
completed policy failures. This adapter calls the native
`scripts/eval_policy_xpolicylab.py`. `run_robotwin.py` binds SAPIEN 3 rendering
to visible `cuda:0`; it does not alter physics, assets, expert validation or
task termination. `WORLDLA_TEST_NUM` defaults to 100; smaller values are only
pipeline checks, not full benchmark scores. Result paths and accepted seeds
are printed by the native evaluator.

`serve.py` constructs the actual WorldLA policy **before** importing native
transport/image libraries; the generic setup script's import order crashed
on the tested mixed CUDA runtime. It then uses the unmodified official
`PolicyServer`, codecs and WebSocket protocol. The eval/server/client argument
order is unchanged. `deploy.py` executes complete K=10 chunks before replanning
and supports explicit environment-ID batches (serial policy inference, not
vectorized throughput). One server supports one coordinated evaluation stream;
use separate ports for independent tasks. RoboDojo/real-robot performance is
not claimed for this RoboTwin-specific checkpoint.

## Model Assets

Hugging Face: [xhx0327/WorldLA-100k-decay-full](https://huggingface.co/xhx0327/WorldLA-100k-decay-full),
pinned revision `7fd1111c6fa8419a64bb6f26383b5025ffbf436d`.

- `original/`: full base + explicit LoRA/router/message adapter/gates, tokenizer
  and normalization assets; preferred for reproducing the historical runtime.
- `merged/`: LoRA-folded dense weights with the router/adapter retained.
  Folding changes arithmetic, so equal task SR is not assumed.
- `training/stage_routes_20k/`: cached targets, manifests, priors, alignment
  fingerprints, configs and verification/conversion tools. Raw Fast-WAM
  instrumentation is not bundled; this is cached-target student retraining.

Checkpoints resolve through the shared `resolve_checkpoint_root` helper.
An explicit `model_dir` or path-valued `ckpt_name` points to a full export;
otherwise shared run-directory naming or `checkpoints/<ckpt_name>` applies.
A release/run directory containing `original/config.json` is accepted.
Set `WORLDLA_MODEL_DIR=/absolute/path/to/full/original` on the server/eval
command to reuse an existing release read-only. Optimizer-only or delta-only
training checkpoints cannot be deployed directly.

## Configuration

`deploy.yml` retains the full standard key set. Extra keys:

| Key | Default | Meaning |
| --- | --- | --- |
| `model_dir` | null | Explicit full export/release directory; standard resolver precedence |
| `inference_seed` | 20260901 | Policy RNG base; distinct from simulator seed group / run label `seed` |
| `old_plan_weight` | 0.75 | Old-tail arm weight; new-head weight is 0.25; 0 disables fusion |

The released horizon is 50, denoising iterations 10, execution interval 10;
gripper fusion is discrete as defined by upstream `WorldLAPolicy.plan()`.
Reset clears previous plans and RNG per episode/environment ID. Inputs are
decoded HWC RGB uint8, language and joint/gripper states. Runtime dimensions
are retrieved from the shared robot metadata helper, then validated against
the released ALOHA layout. Unsupported robots/EE actions fail explicitly.

## Notes

For CUDA/Curobo compilation failures, select the installed toolkit compatible
with the simulator's PyTorch for that process; do not change global CUDA or
another job's environment. This PR does not claim new leaderboard success
rates. See `TESTING.md` for exactly which static, debug and simulator checks
were run and their scope. Publication of official scores remains subject to
maintainer reproduction.
