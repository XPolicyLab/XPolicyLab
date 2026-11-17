# Auriga

**Contributor:** StellarEdge AI Technical Team | **Paper:** [Auriga Technical Report](https://github.com/StellarEdge-AI/Auriga/blob/robodojo-leaderboard/docs/Auriga-Technical-Report.pdf) ([project page](https://auriga.stelledge.com)) | **arXiv:** TBD | **Original code:** [StellarEdge-AI/Auriga](https://github.com/StellarEdge-AI/Auriga/tree/robodojo-leaderboard) (branch `robodojo-leaderboard`)

Auriga is an agent harness in which a language model (`gpt-6-astra`, reasoning effort `high`, served through the OpenAI Responses API) operates the two ARX X5 arms through typed tools: it observes the three RGB cameras and the robot state, previews and executes planned single- or dual-arm motions, waits, and measures the scene with calibrated multi-view geometry and a local 3D perception service. The whole harness runs in the policy server. This directory holds the XPolicyLab glue (`auriga_xpl/`), RoboDojo's own cuRobo planner class and the ARX X5 robot description vendored from RoboDojo under `vendor/robodojo_planner/` (MIT, see its `NOTICE`), and frozen camera calibration constants under `assets/`; `install.sh` checks out the Auriga source at the commit pinned in `deploy.yml`. Evaluation only: `bench_name=RoboDojo`, `env_cfg_type=arx_x5`, `action_type=joint`, `eval_batch: false`.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

Linux with an NVIDIA driver for CUDA 12.8 (>= 570). The policy environment is a conda env with Python 3.12: Auriga and its pinned Codex SDK/CLI (`requirements.lock.txt` of the pinned commit), XPolicyLab (editable), PyTorch 2.7.0 + cuRobo at RoboDojo's pinned commit for the policy-side planner, and the perception service with its weights. The perception service brings its own two uv-managed Python environments (about 12 GB under `VGGT_HOME`, nothing installed system-wide) and 8.1 GB of weights; it needs `git`, `curl` and `python3`, and a GPU with compute capability >= 8.0.

```bash
cd XPolicyLab/policy/Auriga
bash install.sh            # conda env "Auriga"; or: bash install.sh <env_name>
```

`install.sh` creates the untracked `config.env` from `config.env.example`, clones Auriga from `https://github.com/StellarEdge-AI/Auriga` at the commit pinned in `deploy.yml`, installs XPolicyLab editable (`pip install -e`) and `transforms3d`, runs `install_planner.sh` (PyTorch, warp, cuRobo), `download_robot_meshes.sh`, `install_perception.sh` (perception environments) and `download_assets.sh` (see [Model Assets](#model-assets)). The RoboDojo eval environment needs nothing extra: `deploy.py` uses only the standard library and numpy.


## Data Processing

Unsupported. This is an eval-only adapter: `process_data.sh` is omitted and no dataset is read.

## Training

Unsupported. `train.sh` is omitted: there is no trained checkpoint. Actions are produced at evaluation time by the model endpoint and the harness.

## Evaluation

```bash
cd XPolicyLab/policy/Auriga
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_conda_env> <eval_env_conda_env>

# Example (policy server on GPU 0, simulator on GPU 1)
bash eval.sh RoboDojo stack_bowls no-checkpoint arx_x5 joint 0 0 1 Auriga robodojo
```

`ckpt_name` is ignored (`no-checkpoint` by convention). `action_type` must be `joint` and `env_cfg_type` must be `arx_x5`. The model endpoint must be configured first (see [Configuration](#configuration)); without it the policy server refuses to start a simulator evaluation.

GPU memory on the policy GPU (argument 7):

| Component | GPU memory |
| --- | --- |
| cuRobo planner (one X5 planner shared by both arms) | about 2.5–2.7 GB |
| Perception service (VGGT-Omega, SAM 3.1) | about 15 GB |
| Total on one GPU | 11.1 GB at start, 17.5 GB measured after a batch of perception requests (a 24 GB GPU is enough) |

The perception service can be moved to another GPU with `VGGT_GPU`.

Offline wiring check (no simulator, no model endpoint, no perception service; the policy answers hold-still actions):

```bash
cd XPolicyLab/policy/Auriga
EVAL_ENV_TYPE=debug bash eval.sh RoboDojo stack_bowls no-checkpoint arx_x5 joint 0 0 0 Auriga base
DEBUG_OBS_ENCODED=1 EVAL_ENV_TYPE=debug bash eval.sh RoboDojo stack_bowls no-checkpoint arx_x5 joint 0 0 0 Auriga base
```

## Model Assets

The policy has no trained weights. `download_assets.sh` (called by `install.sh`) fetches two groups of files and checks every file against a pinned SHA-256; it needs no Hugging Face token.

| Files | Source | Check | License |
| --- | --- | --- | --- |
| ARX X5 meshes (`vendor/robodojo_planner/Assets/Robots/x5/meshes/`, 24 MB) | Hugging Face dataset `RoboDojo-Benchmark/RoboDojo`, fixed revision (`download_robot_meshes.sh`) | `meshes.sha256` (the dataset's LFS object ids) | MIT |
| `vggt_omega_1b_512.pt` | `facebook/VGGT-Omega` (ModelScope mirror by default) | official upstream SHA-256 | FAIR Noncommercial Research License v1 |
| `sam3.1/sam3.1_multiplex.pt` | `facebook/sam3.1` (ModelScope mirror by default) | official upstream SHA-256 | SAM License |

The two weight files total 8.1 GB and are byte-identical to the files used for the reference runs (pinned revisions and SHA-256 in `download_assets.sh`). In mainland China, `TORCH_SOURCE=aliyun PYPI_INDEX=https://mirrors.aliyun.com/pypi/simple/` (for `install_perception.sh`) avoid slow downloads.

Downloaded files are git-ignored (`.gitignore`).

## Configuration

**Model endpoint.** The endpoint and its key are provided by the Auriga team with the evaluation application; nothing secret is stored in this repository. Put them in the untracked `config.env` (read by `setup_eval_policy_server.sh`) or export them in the environment that runs `eval.sh`; a variable exported with a non-empty value takes precedence over the same line in `config.env`:

| Variable | Role |
| --- | --- |
| `AURIGA_API_BASE_URL` | Base URL of an OpenAI Responses API endpoint (`POST <base>/responses`), provided by the Auriga team |
| `AURIGA_API_KEY` or `AURIGA_API_KEY_FILE` | The key, or a 0600 file that holds only the key, provided by the Auriga team |
| `AURIGA_MODEL`, `AURIGA_REASONING_EFFORT` | `gpt-6-astra`, `high` (fixed for leaderboard evaluation) |
| `AURIGA_SRC` | Auriga checkout (default `third_party/Auriga`, cloned by `install.sh`) |
| `VGGT_HOME`, `VGGT_PORT`, `SAM31_PORT`, `VGGT_GPU` | Perception service install root, first loopback ports to try (8331 / 8333), GPU (default: the policy GPU) |
| `VGGT_SOURCE`, `SAM31_SOURCE` | Weight sources for `download_assets.sh` (`modelscope` by default; no token needed) |

**Perception service.** The service code is `services/vggt_height` of the pinned Auriga checkout (VGGT-Omega geometry; SAM 3.1 for point- and text-prompted masks). Nothing has to be started by hand. The policy server starts the VGGT and SAM 3.1 services as its own child processes on free loopback ports, waits for their health check (about 25 s), and stops them when it exits, also after SIGKILL. A crashed service is restarted once; after a second crash it is marked unavailable and the affected tools report that, as in the reference runs.

Extra `deploy.yml` keys:

| Key | Role |
| --- | --- |
| `auriga_src`, `auriga_commit` | Auriga checkout and the commit it is pinned to |
| `auriga_config` | deployment-only overrides layered over the pinned Auriga configuration (`auriga_eval.json`) |
| `model`, `reasoning_effort` | model id and effort (overridden by `AURIGA_MODEL` / `AURIGA_REASONING_EFFORT`) |
| `result_dir` | per-episode run records written by the harness |
| `poll_s` | seconds before `get_action` answers WAIT (below the RoboDojo client's 120 s request timeout) |
| `wrap_up_s` | seconds `on_trial_end` waits for the run record |
| `verdict_wait_s` | seconds the episode waits for the client's final check after the agent stops |
| `episode_budget_s` | optional wall-clock cap per episode (empty: none) |
| `perception` (optional) | mapping that overrides the perception variables above (`home`, `gpu`, `vggt_port`, `sam31_port`, ...) |

## Notes

- `deploy.py` follows `policy/demo_policy/deploy.py` with three divergences: a WAIT answer is asked again without stepping the simulator (physics stays paused while the model reasons); an empty answer ends the episode through RoboDojo's own final check; each observation carries `additional_info["auriga"]` with the native step count and limit and robot proprioception that the stock observation lacks (measured gripper jaw position, arm joint velocities). No object, layout, seed or evaluator state is read or sent. The verdict goes back through `trial_end`, for the run record only.
- Camera calibration comes from constants of the stock `arx_x5` cell (head camera) and forward kinematics of the measured joints (wrist cameras); no depth is used.
- The motion planner is RoboDojo's own `CuroboPlanner`, run in the policy server with cuRobo at RoboDojo's pinned commit; its sampler state is reset at the start of every episode.
- Each episode starts with an empty cross-episode memory; nothing learned in one episode is visible to the next.
- One environment at a time (`eval_batch: false`). Most of the wall-clock time of an episode is model latency.
- Test setup: `gpt-6-astra`, reasoning effort `high`, at most 170 model requests per episode, the same system prompt, tools and configuration for every task. Reference result (our runs, three seeds, 6,300 episodes): 57.7% SR / 64.0 Score.
- Adapter check: the adapter was run end to end with `eval.sh` (`EVAL_ENV_TYPE=sim`, real model endpoint) on an unmodified RoboDojo main checkout after a fresh `install.sh`.
