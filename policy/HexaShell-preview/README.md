# HexaShell-preview

**Contributor:** HexaFuture | **Paper:** HexaShell: A Thin Harness for Exploring the Limits of VLM Agents as Robot Coding Agents | **arXiv:** [2610.23479](https://arxiv.org/abs/2610.23479) | **Original code:** [HexaFuture/HexaShell](https://github.com/HexaFuture/HexaShell)

Evaluation-only agent policy. An unmodified coding agent (Codex CLI, `gpt-6-astra`) operates the ARX X5 through one command-line program, `robo`; a server in this directory turns each command into a timed joint path with cuRobo inverse kinematics and streams it through the standard action contract. Task competence is in small Python tools under `tasks/<task>/tools/`, written by a second copy of the same model during development and frozen here; the acting agent only chooses which tool to call and with what arguments. The agent runs in a Docker container with no network beyond the model endpoint; the API key never enters the container.

The 8 tasks of the Open capability dimension (`profiles/open/tasks.txt`) are handled differently, because a method may not see their environments before the evaluation: they have no `tasks/<task>/` directory and run with one generic profile. Every Open task gets the same manual (`agent/INTERFACE_open.md`), the same tools (`tools/` and `library/`, listed in `profiles/open/enabled_tools.txt`), no task notes, and a Python kernel in which the agent writes the task logic itself during the episode. These tools were built and tested on the other 34 tasks only, and the profile was frozen before any Open episode was run.

The observation is RGB only (stock `arx_x5` config) and no depth is used, neither the simulator's depth nor a depth network. Positions come from the colour images and the fixed camera calibration: a pixel defines a ray, and the ray is intersected with a plane whose height is measured in the episode, or with the ray of the same point seen by a second camera.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

Requirements on the policy machine: the RoboDojo checkout this XPolicyLab lives in, with its Isaac Sim 5.1 Python environment (cuRobo is used for IK); Docker; the Codex CLI binaries (`codex` and `codex-code-mode-host` side by side; the standalone installer puts both under `~/.codex/packages/standalone/current/bin/` and `CODEX_BIN` may point to the `codex` symlink; `install.sh` records the version it builds with in `agent/codex/CODEX_VERSION`); an OpenAI Responses API endpoint that serves `gpt-6-astra`, and its key in a file (the key is read by the egress proxy on the host and never enters the agent container).

```bash
cd XPolicyLab/policy/HexaShell-preview
bash install.sh                      # first run only creates config.env from config.env.example
vi config.env                        # fill ISAAC_PYTHON, CODEX_BIN, MODEL_UPSTREAM, MODEL_BASE_PATH, MODEL_KEY_FILE; ROBODOJO_REPO defaults to the checkout
bash install.sh                      # builds the agent image and installs websockets>=13 for the policy server
```

Nothing upstream is modified and the stock `arx_x5` observation config is used. The policy server runs in the Isaac Sim Python (`ISAAC_PYTHON`), so `<policy_env>` below may be that interpreter's path or any value when `config.env` sets `ISAAC_PYTHON`.

## Data Processing

Unsupported. This is an eval-only adapter: `process_data.sh` is omitted, and it does not convert datasets.

## Training

Unsupported. `train.sh` is omitted. The policy has no weights: the model is called through the provider API at evaluation time, and the task tools in `tasks/` are frozen.

## Evaluation

RoboDojo's official evaluation, run by RoboDojo on its own machines with a hidden seed: 50 episodes per task (Generalization tasks: 25 standard and 25 cluttered), scored by RoboDojo's checkers with the leaderboard formula. The 34 tasks outside Open use their tools in `tasks/`; the 8 Open tasks use the generic profile.

| Task type | Success rate (%) | Progress score |
| --- | ---: | ---: |
| Generalization | 34.50 | 39.22 |
| Precision | 32.00 | 41.25 |
| Long-Horizon | 31.50 | 42.91 |
| Memory | 81.00 | 84.53 |
| Open | 55.25 | 57.97 |
| **Overall** | **46.85** | **53.18** |

Per task: `eval/official_robodojo.md`; the same numbers are in `eval/official_robodojo.json`.

```bash
cd XPolicyLab/policy/HexaShell-preview
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_env> <eval_env>

# Example: stack_bowls, seed 0, simulator and policy server on GPU 0
bash eval.sh RoboDojo stack_bowls gpt-6-astra arx_x5 joint 0 0 0 unused /path/to/RoboDojo/isaacsim-5.1.0/.venv
```

- `env_cfg_type` is the stock `arx_x5` (three 640x480 RGB cameras, no depth). `action_type` must stay `joint`. `ckpt_name` is a label only; `gpt-6-astra` is the convention.
- Cluttered Generalization variants are task names with the `_random` suffix (`stack_bowls_random`); they share the tools of the base task.
- The policy server and the simulator may share one GPU (about 14 GB together). Batch evaluation is not supported (`eval_batch: false`).
- The agent's full session for each episode (commands, observations, transcript) is written under `runs/bridge/<task>/`. `python3 eval/official_validate.py .` flags episodes in which the agent never issued a command (an infrastructure failure, not a result).
- Almost all of an episode's wall-clock time is model latency, and it grows with the reasoning effort; `agent_wall_s` caps the agent's session at one hour.

Offline wiring check (no simulator). The policy server starts the real agent, so Docker, the Codex binaries and the endpoint are needed; the agent sees placeholder observations and the loop ends at the debug client's step limit:

```bash
export EVAL_ENV_TYPE=debug
bash eval.sh RoboDojo stack_bowls gpt-6-astra arx_x5 joint 0 0 0 unused <eval_env>
```

`./roboshell.sh check-task stack_bowls` checks the tools of one task: the tools load, and the manual the agent reads carries no task or object words.

## Configuration

`config.env` (not tracked; see `config.env.example`):

| Variable | Role |
| --- | --- |
| `ROBODOJO_REPO` | RoboDojo checkout; defaults to the one this XPolicyLab is in |
| `ISAAC_PYTHON` | Python of the Isaac Sim 5.1 environment (policy server) |
| `CODEX_BIN` | Codex CLI binary; `codex-code-mode-host` must be next to it |
| `MODEL_UPSTREAM`, `MODEL_BASE_PATH` | OpenAI Responses API endpoint and base path |
| `MODEL_KEY_FILE` | file holding the API key; read by the egress proxy on the host only |
| `MODEL`, `EFFORT` | `gpt-6-astra`, `xhigh` (the setting for the leaderboard evaluation) |
| `SIM_CACHE`, `TMPDIR`, `EXTRA_LD_LIBRARY_PATH` | optional simulator-side settings |
| `ROBOSHELL_RECIPE` | `1` (default): append the task recipe to the agent's prompt; `0`: leave it out |
| `ROBOSHELL_NOTES` | `1` (default): append `tasks/<task>/notes.md` after the recipe; `0`: leave it out |
| `ROBOSHELL_PROFILE` | unset (default): the Open tasks in `profiles/open/tasks.txt` use the generic profile, all other tasks the default one; `default` or `open` forces one profile for every task |

Extra `deploy.yml` keys: `command_budget` (0 = only the official step limit), `agent_wall_s` (wall-clock cap on the agent's session, 3600 s), `request_timeout_s` (120).

## Notes

- The agent's prompt is the official instruction followed by `TASK RECIPE:` and the task recipe from RoboProbe (commit `5230075`, Apache-2.0, copied unchanged into `agent/recipes/`, including RoboProbe's notes on six tasks), the same text the GPT-6 Astra baseline gives its agent; `roboshell/recipe.py` refuses a recipe file whose SHA-256 differs from `agent/recipes/manifest.json`.
- After the recipe come `TASK NOTES (from earlier runs of the same tools in other scenes):` and `tasks/<task>/notes.md`: at most 20 lines under four fixed headings (time, failed commands, checks before stopping, known problems), written from the development-layout runs of the frozen tools. `eval/check_notes.py` rejects lengths, numbers with two or more decimals, coordinates, pixel positions and any mention of a layout; `roboshell/recipe.py` runs the same check before the notes reach the prompt.
- `_random` variants use the recipe and the notes of the base task.
- Open tasks (generic profile, chosen per task by `roboshell/server/tools.py` `select_profile`): the prompt is the official instruction and the task recipe, without notes. The manual `agent/INTERFACE_open.md` plus the `interface.md` of each tool in `profiles/open/enabled_tools.txt` is the same file for every Open task (`agent/build/AGENTS_open.md`). Besides the base commands the agent has `goto` (absolute pose, with nearby reachable poses when the target is not reachable), `reach` (inverse kinematics only, nothing moves), `pixels` (the images as arrays) and `robo py`, a persistent Python kernel with `roboshell/client/robolib.py` (robot commands, camera rays and projections, triangulation, image marking), mounted read-only into the agent container. Gripper feedback (`fingers_stopped`, `opening_m`) is read from the official observation. `tools/scene` finds object regions in one image and measures an object in 3D from two camera views (plane sweep of the two silhouettes; RGB, camera calibration, the measured table plane and the robot's own URDF only). `library/` holds 8 tools taken from the 34 non-Open tasks and rewritten without task terms (`library/README.md` lists sources and rejected candidates). No tool in this profile was taken from, tuned on or tested on an Open task.
- The server answers each 45 s of agent silence with one hold step, within the client's 120 s request timeout; hold steps cost about 1% of the step budget.
- Observation: RGB images, joint angles, end-effector poses and the camera calibration (`agent/camera_model.json`: fixed intrinsics, fixed head pose, wrist poses from the arm's end pose). The depth slot of the observation is always empty; a tool that tries to read it gets an error that names the alternatives. The shared tool `table_plane` (`tools/table_plane/`) measures the table height from the images once per episode, the first time it is called; other heights are measured in the episode by the task tools or by two-view triangulation.
- Tools cannot import the simulator or read object poses: they see joint angles, TCP poses, the three RGB images with the calibration, and the same motion primitives as the base commands (`roboshell/server/tools.py`, `EpisodeAPI`). They hold no object dimensions or object positions: the sizes, heights and positions of task objects are measured in the episode, by the tool itself or by the agent, which passes them as arguments; the constants left in the code are robot and camera geometry, design margins and generic plausibility bounds.
- `agent/INTERFACE.md` is the manual the agent reads; each enabled tool's `interface.md` is appended to it (`agent/build/AGENTS_<task>.md` shows the result for each task; `python3 roboshell/docs.py build --task <task>` regenerates it). The delivery check (`eval/check_task.py`) rejects tool text that names the task or its objects, and runs `eval/check_notes.py` on the notes.
- In a few tasks some enabled tools still have a depth path; called without depth they stop with a readable error, and the results above were measured with exactly this tool set.
- `tasks/<task>/` holds only what is used at evaluation time: `enabled_tools.txt`, the enabled tools with the modules they import, and `notes.md` (all tasks except `swap_blocks`). The Open tasks have no `tasks/<task>/`.
- Images: `model.py` decodes nothing; it receives the RGB arrays the policy server has decoded. The robo server writes each observation once as lossless PNG files for the agent to view (`obs/head.png`, ...), and the task tools read the same PNG bytes in memory. These PNGs are HexaShell-preview's own copies of the current observation inside the policy process: no XPolicyLab trajectory bits or datasets are decoded or written, so the `cv2.imencode` / `cv2.imdecode` calls in `roboshell/server/`, `tools/` and `tasks/*/tools/` never see the two stored byte formats.
- The internal Python package, entry script and run directories keep the development name `roboshell` (`roboshell/`, `roboshell.sh`, `runs/`); the policy name is HexaShell-preview.
- Data processing and training are unsupported.
