# WALL-WM-2.0 (VAP adapter, evaluation-only)

**Model / leaderboard name:** **WALL-WM-2.0**.

**Contributor:** X-Square-Robot  
**Paper:** Not supplied  
**arXiv:** Not supplied  
**Original code:** Vendored model implementation under `upstream/`.  
**Checkpoint:** https://huggingface.co/x-square-robot/wall-wm-2.0-robodojo/tree/main

This adapter evaluates the specified WALL-WM-2.0 checkpoint on RoboDojo-Sim.

- Robot configuration: `arx_x5`
- Action space: `ee`
- Transport: XPolicyLab's unmodified WebSocket protocol
- Training support: evaluation-only

Model construction and inference are executed on the same owner thread to satisfy the model's CUDA Graph requirements.

For shared conventions, see the [root README](../../README.md). Official evaluation results are published on the [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

`install.sh` creates an isolated, evaluation-only Python 3.11 environment. It downloads public PyPI and PyTorch wheels and verifies every wheel against the hashes in `environment/requirements.lock`. It does not read `host.env`, use another Conda environment, install Isaac Sim, or download model weights.

~~~bash
# CPython 3.11 on Linux x86_64 is required; no Conda installation is required.
cd XPolicyLab/policy/vap
bash install.sh                         # creates .venv; refuses to overwrite it
bash install.sh check                   # imports and pip dependency checks
CUDA_VISIBLE_DEVICES=0 bash install.sh check --gpu
# Or use an explicit, new target directory:
bash install.sh install /path/to/vap-policy-env --python /path/to/python3.11
bash install.sh check /path/to/vap-policy-env --gpu
~~~

See [INSTALLATION.md](INSTALLATION.md) for public sources, CUDA requirements, resource inputs and verification limits. The simulator environment follows the separate official RoboDojo installation instructions; it is not bundled with this policy. Private reference-environment cloning is not part of this submission.

Prepare the RoboDojo checkout, simulator Conda environment and simulation Assets using
the [official installation guide](https://robodojo-benchmark.com/doc/usage/install-and-download/).
Place this adapter at `<RoboDojo>/XPolicyLab/policy/vap`. Supply the simulator environment
path at launch, and keep Assets at `<RoboDojo>/Assets` (a symlink to an external Assets
directory is supported). Run the official `utils/update_embodiment_config_path.py` after
placing or moving the assets. See INSTALLATION.md section 3.1 for the path setup commands.

Set `VAP_QWEN_MODEL` to the Qwen3.5-9B directory before evaluation. This complete VAPv2 serving checkpoint restores every persistent DiT/VAE tensor with `strict=True`, so serving does not read Wan weights. Training and legacy VAP keep their separate Wan-initialized path. `download_qwen.py` fetches the pinned official Qwen revision and verifies each file's expected size and SHA256. `download_checkpoint.py` downloads the current VAP checkpoint from the repository's `main` branch and requires the three named payload files to be present and nonempty.

The official combined entrypoint runs without parent-repository patches. The pinned parent does not forward VAP's timeout fields when constructing its client, so the environment-side adapter reads its own `deploy.yml` and configures the existing official WebSocket client's public `PolicyEvalClientConfig` before its first policy RPC. The defaults are 600 seconds per request and an 1800-second budget for any later reconnection loop; the existing connection was already established by the parent. This accommodates first-inference CUDA Graph compilation without changing requests or retrying a timed-out stateful action. The [diagnostic patch](compat/README.md) is not part of the XPolicyLab-only reproduction contract.


## Data Processing

Evaluation-only: process_data.sh is intentionally absent. No training data conversion is performed. Inputs are official RGB observations; transport decoding uses the shared helpers. The environment-side history collector and recorder use decode_image_bit / encode_image_bit. Normalization uses the packaged training statistics and this checkpoint's own verified normalizer.


## Evaluation

Use the standard XPolicyLab ten-argument `eval.sh` entrypoint. It initializes a
unique run directory before starting the policy server and simulator, and cleans
up its own processes when evaluation ends.

~~~bash
export ROBODOJO_ROOT=/path/to/RoboDojo
export ROBODOJO_SIM_ENV=/path/to/conda/envs/RoboDojo
export VAP_QWEN_MODEL=/path/to/Qwen3.5-9B
export VAP_OUTPUT_ROOT=/path/to/external/vap-results
export VAP_CACHE_ROOT=/path/to/external/vap-cache
cd "$ROBODOJO_ROOT"
EVAL_NUM=1 bash XPolicyLab/policy/vap/eval.sh \
  RoboDojo stack_bowls /path/to/CHECKPOINT \
  arx_x5 ee 0 0 0 \
  /path/to/vap-policy-env "$ROBODOJO_SIM_ENV"
~~~

Both environment arguments above are full directory paths. The simulator uses the
official Conda environment; `conda` must be on PATH. `ROBODOJO_ROOT` is a shell variable
for selecting the checkout, not a runtime override for an adapter located elsewhere.
Assets are resolved through that checkout's `Assets/`, not a separate eval CLI flag.

`VAP_RUN_DIR` does not need to be set: the launcher generates a run ID when none
is supplied and creates `$VAP_OUTPUT_ROOT/$ROBODOJO_RUN_ID`. With the unmodified
official `scripts/robodojo.sh eval`, the server and client independently derive the
same ID from the canonical bench/task/checkpoint/robot/action/seed/eval-mode/eval-num
request. Relative checkpoint paths are canonicalized against `policy/vap/`, and
bare names follow the standard resolver's concatenated run-directory precedence,
then its `checkpoints/<name>` fallback, even though the official server and
client start in different directories. Their start times, PIDs and ports are not used. The server atomically
creates the directory and records a shared server request; the client validates
both identity and that request before reusing it. The shared request excludes
`task` and `eval_num`, so an explicitly identified batch server can serve its
assigned tasks without knowing each client's episode count. It still binds
checkpoint, robot, action, seed and evaluation mode. The default combined run ID
still includes both `task` and `eval_num`; repeating that request refuses to
overwrite the existing directory. Direct `eval.sh` retains its unique timestamp-based
owner ID, and explicit `ROBODOJO_RUN_ID` / `VAP_RUN_DIR` values take precedence.

For split server/client, provide the same explicit `ROBODOJO_RUN_ID` and
`VAP_RUN_DIR` on both sides, and start the server first. Invoke this adapter's
`setup_eval_env_client.sh` through XPolicyLab's standard client launcher; set
`ROBODOJO_SIM_ENV` to the simulator environment. This explicit split contract does
not require the official parent to select a policy-specific client setup.
Smoke's explicit per-task IDs are preserved.

GPU indices are physical node indices. This example uses GPU 0 for both policy and
simulator; select available GPUs before launching. Existing workloads are never
terminated by the launcher. `EVAL_NUM=1` runs a smoke test; use `EVAL_NUM=native`
for official per-task counts. Smoke results are engineering validation, not
leaderboard results.

Prefix with EVAL_ENV_TYPE=debug for the unchanged official TestEnv, invoked by debug_client.py with the 600-second request timeout from deploy.yml (the stock debug CLI does not expose this tuning). It retains the official ten episodes and 20-step limit. Repeat with DEBUG_OBS_ENCODED=1. These artificial observations cannot establish task success. The separate test_adapter_contract.py exercises 321 actions per episode and resets over unencoded/encoded transport using an explicitly fake backend; it is not real-checkpoint evidence.

The standard ten-argument entrypoint runs directly on the evaluation host. The launcher cleans up its own processes and records exit_code.txt. No frontend export or training-cluster controller is required.

Native scores and three-camera videos remain under RoboDojo/eval_result/RoboDojo/TASK/vap/arx_x5/.../RUN_ID/. Per-run logs, recorded observations and runtime configuration remain in `$VAP_RUN_DIR`; the command above places them outside the submission at `/path/to/external/vap-results/RUN_ID/`.

## Model Assets

Pass the checkpoint as an explicit directory, or use the standard resolver with a `checkpoints/` directory in this adapter. The example above uses an explicit checkpoint directory. Required files: `model.safetensors`, `config.yml`, `normalizer_action.pth`. Do not put checkpoint files or model assets in the policy environment.

The public repository is [`x-square-robot/wall-wm-2.0-robodojo`](https://huggingface.co/x-square-robot/wall-wm-2.0-robodojo). [`environment/CHECKPOINT_SOURCES.json`](environment/CHECKPOINT_SOURCES.json) fixes that repository, `revision: "main"` and the three required filenames. 

Preparation verifies training recipe, normalizer statistics and Qwen identity before writing a per-run config view; it never modifies the local source checkpoint directory. `VAP_QWEN_MODEL` names Qwen3.5-9B. Online Qwen uses the checkpoint's HEAD truncation + EOS recipe. This pure VAP adapter makes no GPT/judge calls.

### Relative checkpoint paths

`checkpoint_path` in deploy.yml, or a path-valued `ckpt_name` such as
`./checkpoints/baseline`, resolves relative to `policy/vap/` through the shared
XPolicyLab checkpoint resolver. An absolute checkpoint path remains supported.

Inside a VAPv2 checkpoint's `config.yml`, filesystem paths instead resolve relative
to the directory containing that config, independently of the launch directory.
This covers `checkpoint.resume_from`, `checkpoint.save_path`, the training and
validation dataset config paths, both action-normalizer statistics paths,
`model.text_feature_cache_path`, and `model.qwen35_model_path`. Model repository
IDs and Wan model specifications are not rewritten. Absolute paths remain valid.
The adapter resolves these paths **before** writing its per-run config view;
the vendored typed config loader also supports direct relative-path loading.

For a portable checkpoint, place the canonical statistics file under
`assets/normalizer_keys_robodojo_ee_v3_full_master_relative_global.json` beside
`config.yml` and reference that same relative filename in both normalizer fields.
The standard launcher explicitly selects the adapter's packaged statistics;
their hash and the checkpoint's own normalizer must still match. Direct preparation
without `VAP_NORMALIZER_STATS` reads the path in the checkpoint config. An explicitly
supplied relative `VAP_NORMALIZER_STATS` also resolves against the checkpoint directory.

Historical training references can be recorded under `training_workspace/`
(the former training `/workspace/` tree). These references are provenance only:
the evaluation package does not include the original pretraining checkpoint,
training/validation data configs or text cache. Serving disables training resume
and the offline text cache, and never loads training datasets. This does not make
the package suitable for resuming training without those assets. Qwen remains a
separately supplied inference asset through `VAP_QWEN_MODEL`.

Changing a local config's paths changes its SHA256. Checkpoint download provenance records the bytes of the original download, not a later locally relocated config.

Download the exact official `Qwen/Qwen3.5-9B` revision recorded in `environment/QWEN_SOURCES.json` (Apache-2.0; the license and model terms still apply). The destination must not exist. Every runtime file is bound by its expected byte size and SHA256; mirrors are transport only and do not change the official repository/revision identity:

~~~bash
python -I download_qwen.py --output /path/to/new-Qwen3.5-9B
# If the direct Hub is unreachable, an explicitly selected public mirror can
# transport the same bytes. This does not re-upload or replace the model.
python -I download_qwen.py --output /path/to/new-Qwen3.5-9B \
  --base-url https://www.modelscope.cn/models --transport-revision master
python -I download_qwen.py --verify-only --output /path/to/Qwen3.5-9B
~~~

The download is staged in a sibling temporary directory and becomes visible at the destination only after every hash and the checkpoint Qwen identity pass. Its provenance is written as `<destination-name>.provenance.json` beside the model directory, not inside it, because Qwen runtime identity hashes all JSON payload files. `--verify-only` checks both the 13-file payload and this sibling provenance. Do not use an incomplete `*.partial-*` directory.

The `download_checkpoint.py` script downloads the three named payload files from `main`, rejects an empty file and checks the exact three-file inventory. The destination must not exist; files are staged in a sibling temporary directory and become visible at the destination only after the transfer and inventory checks complete. There are no predetermined size or SHA256 expectations. Each file is fetched in bounded 64 MiB ranges, with six concurrent workers by default (`--workers 1` through `--workers 16`), bounded transient retries and ETag-bound resume checks:

~~~bash
python -I download_checkpoint.py --output checkpoints/wall-wm-2.0-robodojo --workers 6
python -I download_checkpoint.py --verify-only --output checkpoints/wall-wm-2.0-robodojo
# An explicitly selected public mirror is transport only; it still reads main:
python -I download_checkpoint.py --base-url https://hf-mirror.com \
  --output checkpoints/wall-wm-2.0-robodojo --workers 6
~~~

An interrupted invocation retains its incomplete `<destination-name>.partial-*` directory and prints `CHECKPOINT_INCOMPLETE <path>`. To continue it, pass that exact path with the same destination and mirror; do not pass the partial directory as the runtime checkpoint:

~~~bash
python -I download_checkpoint.py --base-url https://hf-mirror.com \
  --output checkpoints/wall-wm-2.0-robodojo \
  --resume checkpoints/wall-wm-2.0-robodojo.partial-EXAMPLE
~~~

Completed chunks are verified again before assembly. The downloader reads the current `main` file metadata: LFS files must match their public `lfs.oid` (the whole-file SHA256). An Xet `xetHash` or bridge ETag is a different object identifier, not a plain-file SHA256; ETags bind ranges and reject remote changes instead. A regular Git file such as `config.yml` has no LFS `oid`, so its verified ranges, ETag and actual downloaded SHA256 provide its transfer and provenance checks. If a HEAD response omits `Content-Length` (for example, a chunked response), the required file's repository-API size is used; a supplied HEAD length must still match it. For a weak ETag, the downloader does not use `If-Range`. A mirror that ignores Range is supported only for one complete single-file response: its status, matching ETag, exact byte count and absence of trailing bytes are checked, and an interrupted partial file is never appended to a new whole response. An already assembled file is also retained, bound to its remote ETag and checked by its actual SHA256, including the current LFS `oid` where one exists, before reuse. Resume refuses a different destination name, source mirror, file size or remote ETag; it never combines retained bytes with a changed remote payload. After a successful download, the partial directory is consumed. A failed or incomplete directory is never published as a usable checkpoint.

Download provenance is written beside the checkpoint as `<destination-name>.provenance.json`. It records the actual size and SHA256 of each downloaded file for traceability, not as a preconfigured release allowlist. `--verify-only` recomputes these values and requires the local payload and its provenance to agree.

Run the downloader again with a **new output directory** to get the then-current `main`; existing directories are never overwritten or updated automatically. 

## Configuration

Stable defaults are in deploy.yml. config_env.py maps them to backend environment variables; caller overrides are retained and recorded. The validated numerical baseline is:

| Setting | Default / meaning |
| --- | --- |
| evaluation_profile | a32v16_320x256_eev3_lfu_nostate_globalnorm_memory |
| inference_mode | ar; 32 actions + 16 frames, images 256x320 |
| action_inference_steps, action_inference_shift | 16, 1 |
| ar_prefix_action_source | measured_state |
| use_kv_cache | true |
| state_conditioning, se3_smoothing, gripper_hysteresis | false |
| base_seed | 0; combined with task/eval-seed/layout model-noise identity |
| v2_action_acceleration | graph_rope |
| v2_action_precision, v2_action_noise_dtype | fp32_head_accum, fp32 |
| image_resize_mode | pil_bicubic |
| ops_backend | pytorch; no private xcompute package required |
| checkpoint_path | null; optional explicit directory |
| text_encoder_device | null; optional visible CUDA device for unchanged online Qwen |
| request_timeout_s, max_connect_seconds | 600, 1800 seconds |
| ws_ping_interval_s, ws_ping_timeout_s | 20, 20 seconds |

Checkpoint history offsets are [32,64,96,128,256], plus the first ego frame. The deploy loop gathers executed RGB/state histories without changing official physics, layouts, rewards, seeds or success rules. One active client per server; batching is unsupported.

`POLICY_ENV` defaults to this adapter's `.venv`; the standard tenth script argument supplies the separately installed simulator environment. `VAP_RUN_DIR`, `VAP_OUTPUT_ROOT`, `VAP_CACHE_ROOT` and `VAP_POLICY_PORT` override output/cache/port paths. `VAP_QWEN_MODEL` is a mandatory model input; Wan is not read by this complete-checkpoint serving path. Set `VAP_REQUIRE_NO_WAN=1` to unset Wan variables and install a fail-closed file-access audit before importing the server. Its `wan_free_server.json` remains RUNNING during a healthy long-lived server and records FAIL if any prohibited open is attempted. The launcher never reads a deployment-level `host.env`. `VAP_RUN_DIR` reserves VAP-owned runtime records. Native results are separately written under RoboDojo `eval_result/` using the run ID. A valid smoke must have a fresh run-bound result directory, `_result.json` with `eval_time >= 1`, and the expected videos; launcher exit 0 alone is insufficient because the unmodified parent can mask a client failure. VAP configures its transport timeouts in the environment-side adapter; the pinned parent need not forward them. The parent does not guarantee Kit exception exit status. Do not treat a compatibility patch as part of an XPolicyLab-only reproduction.

The deployment uses the release's explicit PyTorch operator backend; import-time xcompute rebuilding is disabled. GPU driver compatibility is a host prerequisite; installation and inference on a different GPU model require separate validation. See INSTALLATION.md for the separate public policy and simulator environments.

On a shared node, an optional VAP_TEXT_ENCODER_GPU physical index routes online Qwen to a second GPU. The server exposes policy GPU first and text GPU second; only the resulting BF16 hidden tensor is copied back. Tokenization, checkpoint, attention backend and dtype are unchanged. This does not assert bitwise identity across different GPU hardware. The model is still single-thread owned, and the standard policy GPU positional argument remains a single physical index. Example: prefix the evaluation command above with `VAP_TEXT_ENCODER_GPU=1` to place Qwen on physical GPU 1 while the policy remains on GPU 4. Without this variable the original single-GPU placement remains the default. Check memory on all chosen GPUs first.

## Standard server accelerations

The standard `setup_eval_policy_server.sh` / `eval.sh` entrypoint supports
optional server optimizations from this adapter, disabled by default. No experiment-directory imports are
required. Initialization runs on the model owner thread before the server listens.
Existing client transport, EEF actions, history semantics and IK are unchanged.

| deploy.yml key (default: false) | Environment override | Effect |
| --- | --- | --- |
| text_encoder_warmup | VAP_TEXT_WARMUP | Load/encode Qwen before accepting requests; preserve CPU/CUDA RNG states. |
| pixel_normalization_cache | VAP_PIXEL_NORMALIZATION_CACHE | Exact uint8-to-BF16 normalization lookup. |
| action_time_cache | VAP_ACTION_TIME_CACHE | Cache action timestep embeddings with parameter/version/shape keys. |
| prefill_rope_fusion | VAP_PREFILL_EXACT_ROPE | Reuse the existing exact BF16 Triton RoPE kernel during prefill. |
| prefill_layout_cache | VAP_PREFILL_LAYOUT_CACHE | Cache integer pack/unpack permutations; never cache semantic KV across windows. |

Set any key to true (or its environment override to 1) to enable it independently.
Environment overrides take precedence over deploy.yml. The configuration fallback
defaults are also false. Existing KV cache, CUDA Graph and decode RoPE settings
remain enabled by the baseline deployment configuration.
For example, prefix the existing evaluation command with
`VAP_PREFILL_EXACT_ROPE=1 VAP_PREFILL_LAYOUT_CACHE=1` to enable the two prefill
optimizations. M16 attention tuning is not installed by this adapter.

`server_acceleration.json` records resolved switches and initialization.
`text_warmup.json`, `exact_acceleration.json`, and `history5_prefill_exact.json`
record enabled components. When pixel/time or prefill optimizations are enabled,
the first complete A0 and A1 calls compare optimized
outputs to their reference using the same RNG state; a mismatch fails the request
and prevents subsequent actions. These checks add startup latency. A0-only debug
runs do not validate A1: use a real history-bearing request or longer rollout too.

The implementation comes from the deployed history5 numerical baseline, now
packaged as `server_acceleration.py`, `exact_acceleration.py` and
`prefill_acceleration.py`. The latter retains the full seven-field prefix-cache
ABI, original attention masks/GEMM shapes and FP32 action-head/accumulation path.
Kernels must be compiled separately for each hardware platform; cross-GPU bitwise identity or equal
speedups are not claimed. Numerical validation is performed on the same hardware.

The generic V16 checkpoint preparer records, but does not require fixed values
for, the four action-prefix training-noise fields (enabled/probability/std/excluded
dimensions). Structural, text-encoder and normalizer compatibility checks remain.
