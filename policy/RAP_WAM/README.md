# RAP-WAM

**Contributor:** Yuanpeng He | **Paper:** RAP-WAM: Learning World-Action Models through Predictive Rehearsal and Action Perturbations | **arXiv:** https://arxiv.org/abs/2610.16443 | **Original code:** https://github.com/heyuanpengpku/RAP-WAM

This is an **inference-only** RoboDojo adapter for the actual retained v58_light
language-conditioned checkpoint at step 742500. The historical model and Wan
runtime modules are bundled under `vendor/`; inference does not import a private
training checkout, instantiate a training dataset, or open a latent cache.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

Use Python 3.10–3.11 and a CUDA GPU. Keep the checkout and workspace on shared
storage when evaluating on a cluster:

```bash
export RAP_WAM_WORK_DIR=/absolute/shared/path/rap_wam_eval
# Put this XPolicyLab checkout inside the shared workspace.
cd "$RAP_WAM_WORK_DIR/checkouts/XPolicyLab"
export RAP_WAM_ENV="$RAP_WAM_WORK_DIR/envs/policy"
bash policy/RAP_WAM/install.sh
export RAP_WAM_PYTHON="$RAP_WAM_ENV/bin/python"
```

The installer creates a fresh environment and refuses to reuse a non-empty one.
If a network download interrupts that installation, `bash policy/RAP_WAM/install.sh --resume` can continue the same isolated environment after validating its location
and Python configuration; it does not accept a system or external environment.
Adapter entry points configure temporary files, package/model downloads, CUDA/Torch
caches and adapter runtime state under `RAP_WAM_WORK_DIR`; they reject output and cache
symlinks that escape that workspace. The editable checkout must also be inside
the workspace. The installer clears inherited pip destination overrides;
`PIP_REPORT`, when set to a file, must also stay inside the workspace. Package
indexes, proxies and certificate settings remain configurable. Existing model
assets may be read from another location. `HOME`
is unchanged. In these examples, `$RAP_WAM_WORK_DIR/checkouts` is the configured
RoboDojo workspace root. It contains `env_cfg/`, `scripts/eval_policy.sh`,
`src/eval_client/`, `task/` and the `XPolicyLab/` checkout. Prepare the benchmark
using the [RoboDojo setup instructions](https://github.com/robodojo-benchmark/RoboDojo).
The simulator uses a separate environment; `RoboDojo` in the evaluation command
is the name of that preinstalled Conda environment.
The benchmark owns additional simulator caches: configure Isaac/Kit output paths
and cuRobo cache paths in that environment as well. The adapter does not modify
the simulator installation or redirect `HOME`. The supplied integration evidence
records the isolated shared-storage simulator configuration used for validation.

## Data Processing

Not required for this inference-only adapter. The released bundle retains the
model's normalization parameters and original 35-task language-context table.
There is no trajectory conversion or latent-cache preparation step.

## Training

Training is outside this XPolicyLab adapter. Base-model training is already
available in the [RAP-WAM repository](https://github.com/heyuanpengpku/RAP-WAM/tree/cd32fba405b26f8c3e4ee29120ffc7432a126f47).
The public trainer has its own package, configuration and workflow; historical
checkpoint migration is a separate task. This adapter preserves the retained
checkpoint and does not claim that the public trainer directly reproduces its
parameters. This contribution requests maintainer acceptance as an eval-only
integration. Training code is already released upstream; no release date has
been announced for an XPolicyLab training adapter for this checkpoint.

## Evaluation

Supported: `bench_name=RoboDojo`, `env_cfg_type=arx_x5`, `action_type=joint`.
Download the RAP inference bundle (10.31 GB) and complete official Wan2.2-TI2V-5B
assets (34.20 GB) once before the first evaluation:

```bash
bash policy/RAP_WAM/download_weights.sh
bash policy/RAP_WAM/download_wan.sh
export WAN_CHECKPOINT_DIR="$RAP_WAM_WORK_DIR/assets/Wan2.2-TI2V-5B"
```

The downloader pins official revision
`921dbaf3f1674a56f47e83fb80a34bac8a8f203e` and verifies all 11 files against
`WAN_ASSETS.json` / `WAN_SHA256SUMS`, including the three transformer shards,
index/config, VAE, UMT5 and tokenizer. A partial directory becomes usable only
after verification. To verify existing assets without making another copy:

```bash
bash policy/RAP_WAM/download_wan.sh --source "$WAN_CHECKPOINT_DIR" --verify-only
```

```bash
export EVAL_ENV_TYPE=sim
export WAN_CHECKPOINT_DIR="$RAP_WAM_WORK_DIR/assets/Wan2.2-TI2V-5B"
bash policy/RAP_WAM/eval.sh RoboDojo stack_bowls \
  "$RAP_WAM_WORK_DIR/assets/RAP-WAM-s742500" arx_x5 joint 0 0 1 \
  "$RAP_WAM_ENV" RoboDojo
```

The checkpoint argument names the **bundle directory**, not a training `.ckpt`.
The policy server accepts either a conda environment name or an absolute venv
path. Standard single-environment inference is the default. Batched requests are
served serially to preserve the original B=1 computation; each environment has
independent observations and boundary-blending state.

For the official debug environment without a simulator:

```bash
export EVAL_ENV_TYPE=debug
export RAP_WAM_DEBUG_PYTHON="$RAP_WAM_PYTHON"
export RAP_WAM_DEBUG_INSTRUCTION='Stack the three bowls together.'
bash policy/RAP_WAM/eval.sh RoboDojo stack_bowls \
  "$RAP_WAM_WORK_DIR/assets/RAP-WAM-s742500" arx_x5 joint 0 0 0 \
  "$RAP_WAM_ENV" system
```

For encoded images, set `export DEBUG_OBS_ENCODED=1` and rerun the command.
For two environments, also set
`export RAP_WAM_DEBUG_BATCH=true RAP_WAM_DEBUG_BATCH_SIZE=2`.

## Model Assets

The inference bundle contains `model.pt`, `inference_config.json`,
`task_context.pt`, `task_strings.json`, `model_tensors.json` and `SHA256SUMS`.
Model tensors preserve the original checkpoint values and dtypes; optimizer and
scheduler state are omitted. The 30 normalizer tensors remain inside `model.pt`.

Optional: if you already have the trusted native retained checkpoint and its
context table, re-export them losslessly. The public bundle needs no export step:

```bash
"$RAP_WAM_PYTHON" -B -m XPolicyLab.policy.RAP_WAM.export_weights \
  --checkpoint /absolute/path/to/retained/s742500.ckpt \
  --task-context /absolute/path/to/task_ctx_35.pt \
  --output "$RAP_WAM_WORK_DIR/assets/RAP-WAM-s742500-exported"
```

The exporter rejects other model tensor fingerprints. A bundle can be copied and
verified with:

```bash
bash policy/RAP_WAM/download_weights.sh \
  --source "$RAP_WAM_WORK_DIR/assets/RAP-WAM-s742500" \
  --output "$RAP_WAM_WORK_DIR/assets/RAP-WAM-s742500-copy"
```

Export and copy destinations must be new directories inside `RAP_WAM_WORK_DIR`.

The public bundle is [hyppyh010403/RAP-WAM](https://huggingface.co/hyppyh010403/RAP-WAM),
pinned in `PUBLIC_RELEASE.json` to revision
`8221f826a6891af46d0584ec46d32168c6d01c6c`. The acquisition command used in Evaluation downloads exactly
the six bundle files, verifies them in a staging directory, then creates the
output directory. Run it once for a new destination:

```bash
bash policy/RAP_WAM/download_weights.sh
# Bundle: $RAP_WAM_WORK_DIR/assets/RAP-WAM-s742500
```

Use that bundle path as the checkpoint argument in the evaluation examples.
An explicit alternative requires both `--repo-id <HF_REPO>` and
`--revision <40_CHAR_COMMIT>`; `--output` must stay inside the shared work directory.
No checkpoint or model asset is committed to this adapter directory.

## Configuration

The historical evaluation profile is fixed across tasks:

| Setting | Value |
| --- | --- |
| Observation mosaic | Head 320×256 above two 160×128 wrist views; 384×320 RGB |
| Action | 14 absolute joint/gripper targets, left arm/gripper then right arm/gripper |
| Prediction horizon | 16 |
| `num_inference_steps` | 30 |
| `replan_steps` | 8 |
| `savgol_window` / `savgol_polyorder` | 9 / 2, full horizon, float64 calculation then float32 |
| `boundary_blend_steps` | 4, against the previous pre-blend smoothed continuation |
| `device` | `cuda:0` within the selected visible GPU |
| `tokenizer_path` | Optional local UMT5 tokenizer override |
| `wan_checkpoint_dir` | Optional asset-directory override; takes precedence over `WAN_CHECKPOINT_DIR` |
| `request_timeout_s` | 600-second debug RPC budget; the debug entry point sets it explicitly, so changing this YAML value does not change that client timeout |

Trained instructions use the original table, with case-insensitive trimmed lookup.
Unseen instructions use the historical Wan UMT5 encoder; they are not mapped to an
arbitrary trained task. UMT5 is loaded lazily and needs additional GPU memory.
The same checkpoint and inference profile apply across tasks. `reset()` clears
per-episode observations and blending state. It preserves the historical continuous
policy RNG; the benchmark seed does not reset action noise at initialization or on
each request. Runtime images are decoded RGB arrays supplied by XPolicyLab; the adapter
never decodes or swaps color channels.

## Notes

Validation on 2026-10-23 passed four actual GPU-forward comparisons between the
new and historical wrappers sharing the same strictly loaded policy, plus plain,
encoded and batch-size-2 encoded WebSocket debug checks. A complete RoboDojo
`stack_bowls` smoke episode (seed 0, layout 0) succeeded with native score 100.
This single episode checks integration; it does not establish aggregate benchmark
performance. A newly created Python 3.10 environment also passed `pip check`, all
48 adapter tests, and the complete `eval.sh` launch, wait, client and cleanup path
for plain, encoded and batch-size-2 encoded observations (20 steps per environment).
The public release is pinned above; the accompanying review report records asset
verification, installation provenance and the exact test scope.

On 2026-10-24, all 51 CPU adapter tests passed in the existing isolated environment,
including three additional installer checks for inherited pip options and report
paths. This follow-up changed installation safeguards only; model computations
and weights were unchanged. It did not repeat installation or GPU evaluation.
