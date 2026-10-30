# go3_agent_preview

**Release layout:** This directory publishes the RoboDojo agent source, evidence and technical report under `GO_3_Nano_preview`. It is not connected to the parent adapter's launchers. The installation and evaluation commands below refer to the original standalone layout at `policy/go3_agent_preview/`; they are retained as reference, not as launch instructions for this nested location.

**Contributor:** AgiBot Research Team | **Paper:** [Appendix: Recursive Self-Improvement for Robotic Manipulation in RoboDojo](docs/Appendix_Recursive_Self-Improvement_for_Robotic_Manipulation_in_RoboDojo.pdf) | **arXiv:** Not provided | **Original code:** Included in this adapter.

**Self-evaluation summary:** [RoboDojo evaluation summary](summary/summary.md) | **Method:** [Program development workflow](workflow/README.md) | **Supporting evidence:** [Task evidence and reproduction checks](evidence/README.md).

Evidence data are hosted on [Hugging Face](https://huggingface.co/datasets/skwang123/go3-nano-preview-evidence); follow the [download instructions](evidence/README.md#download-evidence) before running evidence checks. The reproduction scripts and policy runtime assets remain included here.

`go3_agent_preview` starts every episode with the task's code program in an isolated process. An explicit `ctx.switch_to_wam(...)` checkpoint, declared by the task itself, switches that environment to WAM once; WAM continues until episode reset. The WAM proxy (`wam/`) sends current observations over websocket to a standalone GE-Act backend. The vendored inference source in `wam/runtime/` is MIT (see `wam/runtime/LICENSE.forge-source`), with marked third-party components from HuggingFace diffusers (Apache-2.0), msgpack-numpy ([BSD license](wam/runtime/LICENSE.msgpack-numpy)), Cosmos and DowneyFlyfan/iMFVideo. Task programs sit in `gm/programs/`, with their prebuilt data files in `gm/programs/data/`. For each
task, [`evidence/`](evidence/README.md) shows that the values a program carries can be recomputed from
recorded robot data within the tolerance of the code that consumes them.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

Where things are:

| Path | What it holds |
| --- | --- |
| `model.py`, `skill_switch.py` | The adapter entry point and the controller that runs a task program, then the learned branch if the program hands over. |
| `gm/programs/` | One program per task, named after the task. |
| `gm/programs/support/` | Geometry, measurement and motion helpers several programs share. |
| `gm/programs/data/` | The arrays and tables those helpers read. |
| `gm/lib/` | The shared library: camera, kinematics, motion, perception, scene. |
| `gm/runtime/` | The per-episode runtime: the program stepper and the list of source strings a program may yield. |
| `wam/` | The learned branch: a proxy to a GE-Act backend, plus the vendored inference source. |
| `evidence/` | Per task, a folder that recomputes the program's stored values from recorded robot data. |
| `workflow/` | How the task programs and the evidence folders were produced. |
| `summary/` | RoboDojo summariser output from the contributor's self-evaluation run. |
| `docs/` | The technical appendix, including the method, experiments and limitations. |
| `tests/` | The offline suite; no simulator and no weights. |

## Installation

```bash
cd <robodojo_root>/XPolicyLab/policy/go3_agent_preview
bash install.sh
```

`install.sh` installs XPolicyLab in editable mode and then runs `wam/install.sh`. Both operate on one environment, chosen by `select_python.sh`: `PYTHON_BIN` if set, otherwise a virtual environment at `${GEACT_ENV_DIR:-<robodojo_root>/XPolicyLab/.venv-geact}` built with `${XPL_BASE_PYTHON:-${GEACT_BASE_PYTHON:-python3}}`. The chosen interpreter is checked for Python 3.11 or 3.12 before anything is installed, and `wam/install.sh` reuses it rather than creating a second one. That environment also receives `requirements-code.txt`, which carries the dependencies of the code branch that the shared XPolicyLab package does not declare. `wam/install.sh` runs on Linux only: it installs the pinned packages from `wam/requirements-infer.txt` plus XPolicyLab, and checks the adapter and backend imports. The pins need Python 3.11 or 3.12 (`websockets==17.1` requires 3.11 or newer; `numpy==1.26.4` has no wheels for 3.13 or newer), and the backend needs a CUDA GPU. `bash wam/install.sh --check` repeats only the import check. `INSTALL_WAM=0 bash install.sh` skips `wam/install.sh` and leaves only XPolicyLab plus the code-branch dependencies installed, which is not enough for the backend.

## Data Processing

Not provided, and not needed: the adapter consumes the observations the policy server has already
decoded, and there is no offline dataset in this adapter.

## Training

**Only the evaluation implementation is included in this source release.** There is no `train.sh` and no training entry point for the code
branch; what produced its programs is a procedure, documented in
[`workflow/`](workflow/README.md), not a training run. `wam/runtime/` vendors the GE-Act model,
pipeline, transform and configuration source used for inference; its configs also carry training-time
settings, but it contains no training loop and no dataset loader, and the WAM checkpoint is published
in the asset manifest rather than trained here.

The data files the code branch loads (`gm/programs/data/*.json`, `*.npz`, `*.npy`) ship pre-built.
What each one is, and how it was measured, is in [`evidence/`](evidence/README.md).

## Skill switch

The task name selects `gm/programs/<task_name>.py`; a `<task>_random` task without its own file uses the base task's program. All 54 runnable RoboDojo task configurations resolve this way. There are no branch pins, program
overrides or switch activation settings, and the tree carries no table mapping a task to a branch: a
program decides for itself when its plan is exhausted, from observations only.

Task code yields an explicit checkpoint before abandonment cleanup:

```python
yield ctx.switch_to_wam(
    SwitchReason.PLAN_EXHAUSTED,
    detail="required motion exhausted the caller's recovery",
    source="<registered task.source>",
)
```

Only registered task-local checkpoints switch control. Logs, `ctx.failure`, counters, normal program completion and uncaught infrastructure exceptions do not switch. The controller discards that tick's partial code action, sends the current state, RGB images and instruction to WAM, validates the returned chunk, and consumes it per environment. A failed or invalid WAM request holds current joint and gripper state for the remainder of that episode. Reset clears the latch and cached actions; the next episode starts in code. WAM is constructed lazily at the first switch.

The source allowlist is in `gm/runtime/switch_sources.py`: one entry per task, holding the source
strings that task's program may yield. It is a typo guard, not a selector — nothing there chooses a
branch, and an unregistered string raises. Caller retries remain in task code. Broad sources that can fire after a final task operation were removed or protected with bookkeeping guards; those guards suppress switching and do not constitute success. Only the task-declared abandonment paths listed in the allowlist switch control.

For an infrastructure-only pure-code comparison, instantiate `GmModel(cfg, respect_switch=False)` in the harness, or use `Session(..., respect_switch=False)`. This consumes explicit markers without an extra code tick. It is not a deployment setting.

## Evaluation

The WAM backend is a separate long-running process that the evaluation scripts neither start nor stop. Start it before evaluation so the first explicit switch can use it. Adapter construction and successful code-only episodes do not contact it.

The backend occupies one GPU for its whole lifetime. Its inference code runs on `cuda:LOCAL_RANK`, so `CUDA_VISIBLE_DEVICES` chooses the card and `LOCAL_RANK` must stay `0`; the launch script pins it, because an inherited `LOCAL_RANK` from a distributed run would otherwise select a device that does not exist. Neither belongs to the `policy_gpu_id` argument of `eval.sh`, which applies to the separate policy-server process.

```bash
cd <robodojo_root>/XPolicyLab/policy/go3_agent_preview/wam
CUDA_VISIBLE_DEVICES=<backend_gpu_id> PYTHON_BIN=<robodojo_root>/XPolicyLab/.venv-geact/bin/python \
  bash run_geact_backend.sh
```

Weights are fetched from Hugging Face on first run when `CKPT` is unset; see the variable table below for where they land and how to point at existing copies instead.

`run_geact_backend.sh` reads these environment variables. Path values may be absolute or relative to this adapter directory; the script resolves them before it changes directory.

| Variable | Default | Meaning |
| --- | --- | --- |
| `CKPT` | `<WAM_ASSETS_DIR>/preview_model/checkpoints` | Checkpoint directory (see [Model Assets](#model-assets)). Downloaded when unset and absent. |
| `PYTHON_BIN` | `$WAM_PYTHON_BIN`, else `python3` | Python of the inference environment. |
| `STAT_PATH` | `wam/stats.json` | State and action normalization statistics. |
| `CFG` | `wam/runtime/config/cfg_e2e60k_infer.py` | Default E2E60K inference profile. The previous KA profile remains available via an explicit `CFG` override. |
| `ROBOT_NAME` | `$GEACT_ROBOT_NAME`, else `sl_arxvr` | Body-embedding identity; must equal `geact_robot_name`. |
| `HOST` / `PORT` | `127.0.0.1` / `8001` | Address the backend listens on. Set `HOST=0.0.0.0` only when the policy server runs on another machine, and restrict that port to the peer host. |
| `DENSE_ACTION_STEPS` | `30` | Length the backend truncates its returned action chunk to. The number of generated actions is the config's `action_chunk` (54 in the default `CFG`), and the policy server executes only the first `action_horizon` of them, so any value of at least `action_horizon` gives the same executed actions. |
| `SEED` | `42` | Global random seed set at startup. |
| `DEBUG_VIDEO_DIR` | `<robodojo_root>/outputs/wam_openloop` | The backend writes each request's input and generated video here (`input_video_NNNN.mp4`, `generated_video_NNNN.mp4`). |
| `DEPLOY_CONFIG` | this adapter's `deploy.yml` | Deploy config read by the backend. `setup_eval_policy_server.sh` and `setup_eval_env_client.sh` read the same variable. |
| `GEACT_INFER_LEGACY_STATE_LAYOUT` | `identity` | `identity` normalizes the 14D state with 14D statistics; `robodojo_x5` maps it into a 24D statistics layout. |
| `WAM_ASSETS_DIR` | `<this adapter>/.wam-assets` | Where downloaded assets are kept. Point it at shared storage to download once per machine. |
| `WAM_QWEN_DIR` / `WAM_VAE_DIR` / `WAM_LATENT_STATS_PATH` | under `WAM_ASSETS_DIR` | Asset locations. Downloaded when unset and absent. |
| `WAM_MODEL_REPO` / `WAM_MODEL_REVISION` | `ccccsc/preview_model` / pinned revision | Checkpoint, VAE and statistics source. |
| `WAM_QWEN_REPO` / `WAM_QWEN_REVISION` | `Qwen/Qwen3.5-2B` / `main` | VLM source. |

Nothing has to be downloaded by hand: when a path is unset and the file is absent, the launcher fetches it from Hugging Face with `hf` (or `python -m huggingface_hub.commands.huggingface_cli`), into `WAM_ASSETS_DIR`. That needs the CLI once — `pip install "huggingface_hub[cli]"` — and network access on first run only; later runs reuse what is on disk. Set `CKPT` and the three asset variables to skip the download entirely, for example when weights already sit on shared storage.

Then run the policy server and the simulator client on the same machine:

```bash
cd <robodojo_root>/XPolicyLab/policy/go3_agent_preview
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_env_or_uv_path> <eval_env_conda_env>

# Example
bash eval.sh RoboDojo stack_bowls go3_agent_preview arx_x5 joint 0 0 0 <policy_env> <eval_env>
```

`eval.sh` accepts the standard ten arguments above, or the same list with an `expert_num` inserted after `env_cfg_type` (the convention `robodojo.sh eval` uses when an adapter's `eval.sh` mentions that name). It starts `setup_eval_policy_server.sh`, waits for it, runs `setup_eval_env_client.sh`, and stops the server — the same sequence as `policy/demo_policy/eval.sh`, using this repository's `utils/get_free_port.sh` and `utils/wait_for_policy_server.sh`. From `<robodojo_root>`, `bash scripts/robodojo.sh eval --policy-dir XPolicyLab/policy/go3_agent_preview --task <task_name> --ckpt <ckpt_name> --policy-env <policy_env> --env-cfg arx_x5 --action-type joint` runs the same sequence through RoboDojo's own entry point. `<policy_env>` and `<eval_env>` are either an environment prefix (for example `<robodojo_root>/XPolicyLab/.venv-geact`) or the name of an environment under `<robodojo_root>/.conda/envs/`.

The adapter's scripts do not read `EVAL_ENV_TYPE`. `setup_eval_env_client.sh` always launches the RoboDojo simulator through `<robodojo_root>/scripts/eval_policy.sh`, so no debug client is available through them.

For split-machine deployment, run `setup_eval_policy_server.sh` and `setup_eval_env_client.sh` with the arguments given in the [Deployment Flow](../../../README.md#-deployment-flow), and start the backend with `HOST=0.0.0.0` as described above. The policy server must reach the backend at `geact_server_host:geact_server_port`, so set `geact_server_host` to the backend machine's address when the two run on different machines. The backend listener has no authentication and runs whatever inference it is sent: keep `HOST` on loopback or restrict the port to the peer host. Give the backend and the policy server the same deploy config: before its first inference request, the policy server compares the backend's memory and TTC settings and it fails on a mismatch. It compares robot identity only when `geact_robot_name` is `robodojo` or `robodojo_real`; for the `sl_*` identities a `ROBOT_NAME` mismatch is not detected. The backend takes its identity from `ROBOT_NAME`, not from the deploy config, so set `ROBOT_NAME` to `geact_robot_name` yourself.

### Evaluation results

The **official evaluation** reported in the [technical appendix](docs/Appendix_Recursive_Self-Improvement_for_Robotic_Manipulation_in_RoboDojo.pdf), Section 5.1 and Table 2, has an overall **SR of 56.42%** and **score of 61.90**.

The table below reports the contributor's **self-evaluation**: overall **SR 58.86 ±0.19%** and **score 64.85 ±0.39**. These are separate evaluation runs. [`summary/summary.md`](summary/summary.md) preserves the self-evaluation export generated by RoboDojo's `scripts/internal/summarize_result.py`: 42 tasks x 3 seeds, 126/126 cells filled. It describes the code in this directory. Using the benchmark's summariser does not make a self-evaluation an official evaluation.

| Dimension | SR (%) | Score |
| --- | ---: | ---: |
| Generalization | 38.50 ±1.21 | 42.11 ±1.15 |
| Precision | 76.83 ±1.12 | 79.92 ±1.03 |
| Long-Horizon | 55.58 ±1.45 | 70.52 ±0.20 |
| Memory | 67.44 ±2.18 | 68.10 ±2.14 |
| Open | 55.92 ±0.77 | 63.61 ±0.83 |
| **Overall (five dimensions weighted equally)** | **58.86 ±0.19** | **64.85 ±0.39** |

The self-evaluation run used the shipped 120 s request budget, 162/162 units at native episode counts. Generalization
is reported both as the merged 50-episode number above and as its two 25-episode halves in the same
file.

An earlier revision of this adapter had its own sweep. It is not shipped here: a second result table
whose code is no longer in the tree invites the two to be quoted interchangeably.

### What produced these programs

[`workflow/`](workflow/README.md) records the procedure behind the task programs and the evidence
folders: how a program for a task is written, how a suspected regression is tested, how a constant is
tied to something the robot can measure, and how a candidate is frozen and reviewed. It describes a
method, not a script — an agent sits in the middle of it, so it reproduces the shape of the work and
not a particular number.

## Model Assets

The checkpoint, VAE and latent statistics can be [downloaded here](https://huggingface.co/ccccsc/preview_model). The Qwen3.5 VLM can be [downloaded separately](https://huggingface.co/Qwen/Qwen3.5-2B). `run_geact_backend.sh` downloads whichever of them is missing; the commands below do the same by hand, which is what you want on a machine without network access at evaluation time. These assets are not included in this source repository; only the normalization statistics also ship locally as `wam/stats.json`.

The default `CFG` is `cfg_e2e60k_infer.py`: video/action view embeddings 8/8, body embeddings 12, legacy spatial RoPE, zero null-text tokens, FLA disabled, and VGen compilation left at the profile's default of disabled (`vgen_compile_forward=False`, `vgen_compile_mode="default"`). The previous KA45K profile remains available as `cfg_e2e_from_geactv2p5_new_fix_infer.py` through an explicit `CFG` override, requires its matching checkpoint, and is the profile that enables `reduce-overhead` compilation. The supplied profile is preserved except for `use_fla_kernels=False`, matching the E2E60K checkpoint. No checkpoint metadata or compatibility-check code is changed.

Through its `metadata` request the backend reports the sha256 of both checkpoint `.bin` files, of `CFG` and of `STAT_PATH`, plus the file names of `CFG` and the checkpoint directory. It returns no absolute paths. The [published asset manifest](assets_manifest.json) records the available download at Hugging Face revision `24e3a2553f274b8700f52a3065d7088e80f8da10`, including Git LFS SHA-256 values for the large weights and locally computed hashes for the small configuration/statistics files. This identifies the published assets.

| Published checkpoint file | SHA-256 |
| --- | --- |
| `checkpoints/diffusion_pytorch_model.bin` | `120016afec1b0cb14de3ad79f3353f4b1620d08f668ef224fd24b6fd41a8d540` |
| `checkpoints/action_model/diffusion_pytorch_model.bin` | `c18390b9f2344d5aae832b614361373715841290c74a94bd52018407599f9d5d` |

If `CKPT` or `CKPT/action_model/` also contains `diffusion_pytorch_model.safetensors.index.json` or `diffusion_pytorch_model.safetensors`, that file is loaded instead of the `.bin`, and the reported sha256 values no longer identify the loaded weights; provide only the `.bin` files. Record asset paths, asset hashes and the config hash with results; environment variable overrides do not change `cfg_sha256`. Record the evaluated Qwen revision separately; it is not part of the preview-model asset manifest.

The backend loads:

| Asset | Set by | Download source | Expected content |
| --- | --- | --- | --- |
| Checkpoint | `CKPT` | [Download](https://huggingface.co/ccccsc/preview_model) | `checkpoints/config.json` and `checkpoints/diffusion_pytorch_model.bin` (video model), plus `checkpoints/action_model/config.json` and `checkpoints/action_model/diffusion_pytorch_model.bin` (action expert). |
| Normalization statistics | `STAT_PATH` | [Download](https://huggingface.co/ccccsc/preview_model) or bundled `wam/stats.json` | `stats.json` with `state` and `action` entries, each holding a 14-value `mean` and `std`. It overrides the inference config's own `stat_path`. |
| Qwen3.5 VLM | `VideoModelConfig.vlm_model_id` in `CFG` | [Download](https://huggingface.co/Qwen/Qwen3.5-2B) | Hugging Face directory loaded with `AutoModelForImageTextToText` and `AutoProcessor` (set `WAM_QWEN_DIR`). |
| VAE | `GEActModelConfig.vae_path` in `CFG` | [Download](https://huggingface.co/ccccsc/preview_model) | `vae/` directory with `config.json` and `diffusion_pytorch_model.safetensors`. |
| VAE latent statistics | `GEActModelConfig.stats_tracker_path` in `CFG` | [Download](https://huggingface.co/ccccsc/preview_model) | `vae/latent_stats.pth` state dict. |

Download the assets with the Hugging Face CLI:

```bash
hf download ccccsc/preview_model --revision 24e3a2553f274b8700f52a3065d7088e80f8da10 --local-dir /path/to/preview_model
hf download Qwen/Qwen3.5-2B --local-dir /path/to/Qwen3.5-2B
```

Both bundled configs obtain asset paths from environment variables and contain no internal asset paths. Before launching the backend, set `CKPT` and export `WAM_QWEN_DIR`, `WAM_VAE_DIR` and `WAM_LATENT_STATS_PATH` as absolute paths to the downloaded assets. These variables are required for default WAM inference; they have no bundled asset defaults. Training-only paths are unset. For example:

```bash
export CKPT=/path/to/preview_model/checkpoints
export STAT_PATH=/path/to/preview_model/stats.json
export WAM_QWEN_DIR=/path/to/Qwen3.5-2B
export WAM_VAE_DIR=/path/to/preview_model/vae
export WAM_LATENT_STATS_PATH="$WAM_VAE_DIR/latent_stats.pth"
```

Use the matching VAE assets from the downloaded `vae/` directory. Before loading, `run_geact_backend.sh` runs `check_checkpoint.py`, which refuses to start when the checkpoint's `config.json` files disagree with `CFG` on `num_view_identities`, `goal_mode_null_text_tokens`, `spatial_rope_grid_mode`, `use_fla_kernels` or `num_body_embeddings`. `ckpt_name` only labels the run: the adapter reads no `checkpoints/<bench_name>-<ckpt_name>-...` directory.

## Configuration

`deploy.yml` keys and their shipped values:

| Key | Default | Meaning |
| --- | --- | --- |
| `policy_name` | `go3_agent_preview` | Must equal the directory name. |
| `protocol` | `ws` | Policy-server transport. |
| `host` | `localhost` | Policy-server bind address; `setup_eval_policy_server.sh` overrides it (10th argument, default `localhost`). |
| `port`, `bench_name`, `task_name`, `ckpt_name`, `env_cfg_type` | `null` | Set per run by `setup_eval_policy_server.sh`. |
| `action_type` | `joint` | Set per run; both branches accept only `joint`. |
| `seed` | `0` | Set per run. |
| `gpu_id` | `null` | Standard adapter key; the launch script sets visible devices through `CUDA_VISIBLE_DEVICES`. |
| `eval_batch` | `true` | Batched evaluation across the simulator's parallel environments. |
| `geact_robot_name` | `sl_arxvr` | Body-embedding identity: `sl_agilex`, `sl_acone`, `sl_arxvr`, `robodojo` or `robodojo_real`. Must match the checkpoint and `ROBOT_NAME`; for `robodojo` and `robodojo_real` the policy server checks the backend. |
| `geact_server_host` / `geact_server_port` | `127.0.0.1` / `8001` | Address of the WAM backend. |
| `geact_connect_timeout_s` / `geact_request_timeout_s` | `30` / `600` | Backend connect and per-request timeouts, in seconds. |
| `image_size` | `[256, 384]` | `[height, width]` each camera image is resized to. |
| `camera_names` | `[cam_head, cam_left_wrist, cam_right_wrist]` | Camera order; any other order is rejected. |
| `action_horizon` | `30` | Actions executed per chunk; the backend output is truncated to this length. |
| `num_ae_steps` | `5` | Action-expert denoising steps per request. |
| `memory_enabled` | `false` | Reserved; setting `true` raises `NotImplementedError` in this release. |
| `memory_n_previous` | `8` | Frames per request in memory mode (sampled history plus the current frame). |
| `memory_config_path` | `null` | Reserved; no memory inference config is bundled. |
| `ttc_enabled` | `false` | RCS best-of-N test-time selection among action candidates. |
| `rcs_num_proposals` | `8` | Candidates per observation; `1` skips scoring. |
| `rcs_timestep_ratios` | `[0.25, 0.5, 0.75]` | Scoring points as fractions in `[0, 1)` of the action-expert schedule. |
| `rcs_noise_seed` | `null` | `null` uses the global RNG; an integer `>= 0` fixes the scoring noise. |
| `bbox_enabled` | `false` | Reserved; setting `true` raises `NotImplementedError` in this release. |
| `bbox_vlm_fixed_sequence_length` | `384` | VLM sequence length in bbox mode; must be `384`. |
| `bbox_missing_policy` | `full_image` | When no target is found: `full_image` uses the whole image, `error` raises. |
| `bbox_cache_mode` | `instruction` | `instruction` reuses the first found box per environment and instruction until reset; `frame` annotates every chunk. |
| `bbox_artifact_dir` | `null` | Optional directory for head images, boxed images and `annotations.jsonl`. |
| `code.out_dir` | `null` | Optional directory for code tapes/frames, `events.jsonl`, `episodes.jsonl` and switch observation snapshots. |
| `code.worker_startup_timeout_s` | `30` | Deadline for a spawned code worker to import the adapter and answer its first message. Raising it helps on a cold page cache, a slow shared filesystem or a large batch; the effective value is capped by the observation RPC budget below. |

`XPL_REQUEST_TIMEOUT_S` and `XPL_HANDSHAKE_TIMEOUT_S` are this adapter's own settings, read in `deploy.py` from the client RoboDojo hands the evaluation loop, and applied before the first request. They accept positive values in seconds; leaving them unset keeps the protocol defaults (120 s and 60 s). Nothing outside this directory reads them.

`XPL_REQUEST_TIMEOUT_S` also governs the policy server, which reads it independently (not through `deploy.py`): a WAM request may not run past `min(geact_request_timeout_s, XPL_REQUEST_TIMEOUT_S − 10)`, so the shipped defaults give **110 s**, not the 600 s in `geact_request_timeout_s`. Set the variable in both processes — the policy server is a separate process from the simulator — and note that one WAM deadline is shared by every switching environment in the batch, not granted per environment. A request that misses it holds that environment's joint and gripper state for the remainder of the episode. The same budget caps `code.worker_startup_timeout_s`. Record these settings with the launch configuration.

Offline verification from the repository root:

```bash
# from the XPolicyLab root; PYTHON is the interpreter install.sh selected
PYTHON=<robodojo_root>/XPolicyLab/.venv-geact/bin/python   # or your PYTHON_BIN
"$PYTHON" -m pip install -e . -r policy/go3_agent_preview/requirements-code.txt pytest
PYTHONPATH="$PWD" "$PYTHON" -m pytest -q policy/go3_agent_preview/tests
```

The suite covers lazy construction, explicit one-way hand-over, action discard, batch/reset/chunk
state, marker sourcing, bounded errors, task completion guards and the transport contracts. It uses
synthetic task programs and a mock backend, so it establishes the adapter's wiring and not a success
rate. Run it in the environment `install.sh` created, which carries `requirements-code.txt`; add
`pytest` if it is not already present; `requirements-code.txt` supplies `scipy`, and the shared
package supplies `numpy` and `opencv-python-headless`.

Before a full official sweep, run the simulator smoke test: four tasks × ten layouts on one machine, checking that the environment runs, the code branch executes and any switch is served by the backend. See [tests/smoke](tests/smoke/README.md).

```bash
cd <robodojo_root>/XPolicyLab/policy/go3_agent_preview/tests/smoke
./smoke_test.sh --robodojo-root <robodojo_root> --policy-env <policy_env>
```

## Recording results

RoboDojo's summarizer writes `_summary.md`; keep it next to the episode results rather than in this adapter. To make a published number reproducible, record with it:

| Record | What to capture |
| --- | --- |
| Code and benchmark | XPolicyLab commit, parent RoboDojo commit, per-task layout configuration, `env_cfg_type` and `action_type`. |
| Loaded assets | Checkpoint and Qwen revisions, VAE and latent-statistics hashes, and the normalization statistics used. Compare against [assets_manifest.json](assets_manifest.json). |
| Backend metadata | The backend's `metadata` reply: `checkpoint_files`, `cfg_sha256`, `stats_sha256`, `robot_name`, `state_layout`, `startup_seed`, `dense_action_steps`, memory and TTC settings. |
| Effective launch | The deploy config plus every override: `ROBOT_NAME`, `GEACT_INFER_LEGACY_STATE_LAYOUT`, `XPL_REQUEST_TIMEOUT_S`, horizons, inference steps, and policy/environment runtime versions. |
| Execution | Code revision, selected task program, which switch sources fired, and client timeouts. |

Never include API keys in recorded metadata.

## Notes

- Supported: `env_cfg_type=arx_x5`, `action_type=joint`, 14D state and action `[left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]`, with dimensions read through `get_robot_action_dim_info`. The default controller requires two 6-joint arms and two grippers; WAM output is checked against authored arx_x5 joint limits.
- `memory_enabled` and `bbox_enabled` must remain `false` in this release. Their implementation code is retained for future integration, while unused inference configs have been removed; but enabling either raises `NotImplementedError` before backend/model initialization. External bbox API calls, image publishing and key loading are stubbed out; no external grounding service is used.
- `ttc_enabled` needs no extra assets; action-expert compute grows with `rcs_num_proposals`.
