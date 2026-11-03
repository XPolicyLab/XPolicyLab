# Era0-Preview

**Contributor:** Liming Zheng | **Paper:** To be announced | **arXiv:** To be announced | **Original code:** Inference runtime included in `runtime/`.

Era0-Preview uses a Qwen3.5-4B visual language backbone and a G0.5 prefix-KV action expert. This eval-only adapter vendors the inference runtime and supports RoboDojo `arx_x5`, Cartesian `ee` actions, and batched simulation. The import/directory name is `Era0_Preview`.

**Model repository:** [roboterax/Era0-Preview-5B-RoboDojo](https://huggingface.co/roboterax/Era0-Preview-5B-RoboDojo) · **Files:** [repository files](https://huggingface.co/roboterax/Era0-Preview-5B-RoboDojo/tree/main) · **Model card:** [README](https://huggingface.co/roboterax/Era0-Preview-5B-RoboDojo/blob/main/README.md).

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

Install RoboDojo's Isaac Sim environment and assets following the [official installation guide](https://robodojo-benchmark.com/doc/usage/installation/). Keep the policy and simulator in separate environments. Use a real XPolicyLab checkout under the SDK workspace; the shared robot-dimension helper resolves `env_cfg/` relative to the checkout and does not support linking an external checkout into the SDK. The policy environment needs Python 3.11, a compatible CUDA 12.9 driver, and FlashAttention-2/FLA; the known working versions are in `requirements.txt` and `install.sh`.

```bash
cd RoboDojo/XPolicyLab/policy/Era0_Preview
bash install.sh
```

`install.sh` creates `.venv` by default; use `POLICY_VENV=/path/to/venv` to change it. Launchers accept a conda environment name, venv directory, or Python executable. The required inference runtime is included. `runtime/` imports use the policy namespace so unrelated policies' `models` packages cannot shadow it. No SDK source files or shared configurations are edited.

## Input processing

The adapter accepts three RGB camera views and Cartesian observations. It includes the image transforms, observation/action conversion and normalization loader. Use the bundled normalization unchanged; never recompute it from evaluation observations.

## Evaluation

Evaluation code lives in the [XPolicyLab policy directory](https://github.com/Zheng-Liming/XPolicyLab/tree/feature/era0_preview/policy/Era0_Preview). Use this checkout directly; no code archive is needed.

Model weights, tokenizer/processor assets and the exact normalization file are provided in the model repository. Download them from this policy directory:

```bash
bash download_checkpoint.sh
```

The script defaults to `roboterax/Era0-Preview-5B-RoboDojo` at revision `main`, writing into this policy's `checkpoints/era0-preview/`. Use `POLICY_ENV=/path/to/venv` if the environment is not `.venv`; pass `--dry-run` to inspect without downloading. A custom destination or the previous explicit repository syntax also works:

```bash
bash download_checkpoint.sh /path/to/checkpoints/era0-preview
bash download_checkpoint.sh roboterax/Era0-Preview-5B-RoboDojo \
  checkpoints/era0-preview main
```

The published checkpoint uses a flat layout:

```text
checkpoints/era0-preview/
  norm_stats.pkl
  assets_manifest.json
  config.json
  hy_xvla_adapter.json
  model.safetensors
  tokenizer.json
  tokenizer_config.json
  processor_config.json
  chat_template.jinja
  deploy.yml
  SHA256SUMS
```

Evaluation requires the exact `norm_stats.pkl` alongside the model. Its expected SHA256 is `f0d1a21fc63e778ef5d75799073a719470b507f96f3822a905b4195ed755b285`; the downloader checks this value. [checkpoint_release.json](checkpoint_release.json) records the repository, revision and expected weight/normalization fingerprints. The weight is approximately 11.9 GB. Evaluation code is obtained from XPolicyLab, independently of model assets.

Use the logical checkpoint name `era0-preview` on standard SDK entries. The shared resolver finds `checkpoints/era0-preview/`. For an external model directory, set `ERA0_CHECKPOINT_PATH=/path/to/model` on the policy server while retaining a path-safe checkpoint name; this keeps the SDK result directory readable by its official summarizer. The three-seed driver accepts an explicit `--checkpoint` directory and forwards it separately from `--checkpoint-name` (default `era0-preview`). The legacy bundle layout with a `model/` subdirectory is also supported. The same checkpoint and normalization are used for all three evaluation seeds. Use this adapter/runtime to load the custom VLA architecture; the public checkpoint does not provide generic `AutoModel(..., trust_remote_code=True)` loading. If the release includes `SHA256SUMS`, verify it after downloading:

```bash
(cd checkpoints/era0-preview && sha256sum -c SHA256SUMS)
```

Standard ten-argument XPolicyLab entry (one task / seed):

```bash
ERA0_EXECUTION_PROFILE=official ROBODOJO_EVAL_LEGACY_RENDER=0 bash eval.sh RoboDojo stack_bowls \
  era0-preview arx_x5 ee 0 0 0 .venv RoboDojo
```

The split-machine `setup_eval_policy_server.sh` / `setup_eval_env_client.sh` use the standard XPolicyLab arguments; see the shared Deployment Flow. `EVAL_ENV_TYPE=debug` uses the official debug client; set `DEBUG_OBS_ENCODED=1` for its encoded RGB transport check. Actual model loading still requires the selected weights and a free GPU.

Complete evaluation, preview only:

```bash
POLICY_ENV=.venv bash eval_three_seeds.sh \
  --checkpoint "$(pwd)/checkpoints/era0-preview" --sim-env RoboDojo \
  --gpu-ids 0,1,2,3,4,5,6,7 --execution-profile official --render-mode sdk \
  --output /path/to/fresh/eval
```

The same adapter is recognized by RoboDojo's official CLI (run from the RoboDojo root); repeat with seeds 1 and 2:

```bash
ERA0_EXECUTION_PROFILE=official ROBODOJO_EVAL_LEGACY_RENDER=0 bash scripts/robodojo.sh benchmark \
  --policy-dir XPolicyLab/policy/Era0_Preview \
  --ckpt era0-preview \
  --policy-env /path/to/policy-venv --eval-env RoboDojo --eval-num native --seed 0 \
  --gpu-ids 0,1,2,3
```

Append `--run` to start simulation. Tasks are read from the installed SDK default all-dimensions runnable inventory (`task_inventory.py --only-runnable --dimension all`), including its random variants. No task list is bundled or pinned. Seeds are fixed to 0/1/2. Episode counts come from the installed SDK's `_task.yml` using `EVAL_NUM=native` (current official protocol: 2,100 per seed, 6,300 total). The selected SDK counts/revision, checkpoint metadata, normalization hash, rendering mode and execution contract are recorded before launch. With a separately located SDK, pass `--robodojo-root /path/to/RoboDojo` (or export `ROBODOJO_ROOT` for standard entries). Each available GPU receives tasks from a shared queue. `--output` must be a fresh directory for launch logs and metadata. It does not relocate simulator results or videos; the SDK writes those to its normal `eval_result/` tree.

Use `--tasks-file /path/to/subset.txt` only for diagnostics; the plan explicitly labels subsets. Missing trials fail completeness checks even for `make_toast_random`; that task is not automatically retried. Other SDK in-process failure recovery is retained. There is no task-specific action tuning, no gripper postprocess, no layout/seed-based action branch, and no evaluation-data subsampling.

Result JSON and videos are written by the installed SDK, with exactly the same layout as other policies:

```text
RoboDojo/eval_result/RoboDojo/<task>/Era0_Preview/arx_x5/
  <seed>_ckpt_name=<checkpoint-name>,action_type=ee/<run_id>/
    _result.json
    episode_0000000_cam_head_success.mp4
    episode_0000000_cam_left_wrist_success.mp4
    episode_0000000_cam_right_wrist_success.mp4
```

Video outcome suffixes and camera files are controlled by the SDK. Both profiles preserve its stream, final video and resume-manifest paths; the adapter adds no combined video, text overlay or alternate video directory.

Use the SDK's official summary command from the RoboDojo root:

```bash
bash scripts/robodojo.sh summarize
```

This invokes `scripts/internal/summarize_result.py` and writes `eval_result/RoboDojo/_summary.md`. **Overview / Avg Score** in that file is the primary metric. The three-seed driver calls this installed script after evaluation; it contains no copied score aggregation. The official script also checks the three camera filenames against result records.

The driver stores only `evaluation_contract.json`, execution completeness in `evaluation_summary.json`, and `logs/<task>_seed<seed>.log` under `--output`. These are launch metadata, not a second score report. Use a separate SDK workspace or logical checkpoint name when isolating evaluations: the SDK summarizer follows its own latest-run selection across the native result tree.

### Local execution, camera timing and official submissions

The default `--execution-profile local` (`ERA0_EXECUTION_PROFILE=local` on standard entries) bundles the local SDK's IK protection: solver failure, invalid targets or any joint jump above 0.5 rad holds that arm at its live measured position. Other arms/grippers continue, and the step still counts. Only IK protection and its diagnostics differ from the selected SDK; random-task concurrency, result/resume paths and video handling stay SDK-native. Neither profile edits installed SDK files. See [SDK_COMPATIBILITY.md](SDK_COMPATIBILITY.md) for the full local change audit and tests.

The default is **policy-side image delay with SDK rendering** (`HY_OBS_LAG=1`, `ROBODOJO_EVAL_LEGACY_RENDER=0`; driver: `--obs-lag 1 --render-mode sdk`). To opt into the earlier renderer compatibility instead, explicitly disable policy delay and select legacy rendering (`--obs-lag 0 --render-mode legacy`, or `HY_OBS_LAG=0 ROBODOJO_EVAL_LEGACY_RENDER=1` on the standard entry). It uses a single `env.render()` and skips zero-delay Kit flags, without changing physics, reset-drive fixes, rewards, layouts or action semantics. It is not a full historical SDK rollback. Both modes keep their configuration in the result artifacts.

**Use explicit `--execution-profile official --render-mode sdk` for the audited official execution/render combination**, or `ERA0_EXECUTION_PROFILE=official ROBODOJO_EVAL_LEGACY_RENDER=0` on standard entries. This uses the installed, hash-validated SDK execution code and renderer; policy-side image lag remains enabled. No result/video path plumbing or renderer override is installed. Setting only the rendering switch retains local IK protection. Compatibility-mode results must be disclosed; acceptance is up to benchmark maintainers. Unknown SDK versions fail before simulator startup rather than silently replacing newer client code. The default dry run never starts a simulation. `install.sh` only installs the policy environment.

## Configuration

`deploy.yml` fixes the original evaluated policy contract: bf16 inference; 224-pixel letterbox preprocessing; head history 8, left/right wrist 1 each; 40-step history interval; full-history VLM with bidirectional image blocks/current-only wrist visibility; current-image-only prefix-KV action expert; UMI coordinates; relative EE rotation-matrix actions; saved z-score normalization; domain 16; 25 predicted slots, skip anchor slot 0 and execute slots 1–10; ten denoising steps. `history_cadence=chunk` retains only control-boundary images, as in the original evaluator. RGB arrays come from the shared XPolicyLab server decoder and are never swapped or decoded by the adapter.

Time RoPE is loaded **from checkpoint `config.json`** and retained unchanged across evaluation seeds. Disabled RoPE on a compatible checkpoint remains disabled. Language prompt placement and all observation/action transforms are included in the adapter. `source_manifest.json` records copied source and package hashes; SDK execution snapshots and their separate audit manifest are documented in [SDK_COMPATIBILITY.md](SDK_COMPATIBILITY.md).

## Submission status and verification

Model weights and inference assets are available at the model repository link above. Model/normalization fingerprints are retained in `checkpoint_release.json`. This directory contains evaluation code only. The plain/encoded debug transport checks and full three-seed evaluation remain pending; the single-task smoke test is documented in [SUBMISSION.md](SUBMISSION.md).

Reference standards: [XPolicyLab CONTRIBUTING](../../CONTRIBUTING.md), [official quick evaluation and complete protocol](https://robodojo-benchmark.com/doc/usage/quick-evaluation/), [official policy integration](https://robodojo-benchmark.com/doc/usage/xpolicylab/).

This package contains evaluation and inference code only. Its runtime includes the
model architecture, checkpoint loading, observation transforms and action sampling.

### Policy-side image observation delay

By default, the **policy server** uses `HY_OBS_LAG=1`: images come from the
previous observation update. Set `HY_OBS_LAG=0` (driver: `--obs-lag 0`) to disable
image delay while keeping SDK rendering. All three cameras
share the same delay; state, language and the relative-action reference pose
remain current. Each environment owns copied image snapshots, initial updates
repeat its first image, and reset clears the queues. Delay advances on every
`update_obs` / `update_obs_batch`, including within a ten-action chunk; it does
not count chunks or change history spacing, masks, time RoPE or action execution.

For a standard single-task launch from this policy directory:

```bash
HY_OBS_LAG=1 ROBODOJO_EVAL_LEGACY_RENDER=0 ERA0_EXECUTION_PROFILE=official \
  bash eval.sh RoboDojo stack_bowls era0-preview arx_x5 ee 0 0 0 /path/to/policy-env RoboDojo
```

For an all-task, native-count, three-seed preview:

```bash
bash eval_three_seeds.sh --checkpoint /path/to/checkpoint --output /path/to/new-results \
  --obs-lag 1 --render-mode sdk --execution-profile official
```

Append `--run` to start the evaluation. The driver forwards the delay to policy
servers and records `policy_image_lag_updates` in the evaluation contract.
The wrappers default to SDK rendering, including when lag is disabled; explicitly requesting
legacy rendering with positive lag is rejected to avoid stacking both delays. A separately hosted
policy server defaults to `HY_OBS_LAG=1` as well; any explicit override must match
the client configuration. Setting it only on the simulator cannot
change the server's inputs. This is a fixed observation delay, not an exact
reproduction of asynchronous rendering. It does not require SDK source changes.
