# RoboBigWater

**Contributor:** RoboBigWater authors | **Paper:** not yet available | **arXiv:** not yet available | **Original code:** not yet available

Evaluation-only agent policy. An unmodified coding agent (Codex CLI, `gpt-6-astra`) operates the ARX X5 through one command-line program, `robo`; a server in this directory turns each command into a timed joint path with cuRobo inverse kinematics and streams it through the standard action contract. Task competence is in small Python tools under `tasks/<task>/tools/`, written by a second copy of the same model during development and frozen here; the acting agent only chooses which tool to call and with what arguments. The agent runs in a Docker container with no network beyond the model endpoint; the API key never enters the container. Depth is used: the adapter ships an RGB-D observation config (`env_cfg/arx_x5_rgbd.yml`).

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

Requirements on the policy machine: the RoboDojo checkout this XPolicyLab lives in, with its Isaac Sim 5.1 Python environment (cuRobo is used for IK); Docker; the Codex CLI binaries (`codex` and `codex-code-mode-host` side by side; the standalone installer puts both under `~/.codex/packages/standalone/current/bin/` and `CODEX_BIN` may point to the `codex` symlink; all results here used Codex CLI 0.159.0; `install.sh` records the version it builds with in `agent/codex/CODEX_VERSION`); an OpenAI Responses API endpoint that serves `gpt-6-astra`, and its key in a file (the key is read by the egress proxy on the host and never enters the agent container).

```bash
cd XPolicyLab/policy/RoboBigWater
bash install.sh                      # first run only creates config.env from config.env.example
vi config.env                        # fill ISAAC_PYTHON, CODEX_BIN, MODEL_UPSTREAM, MODEL_BASE_PATH, MODEL_KEY_FILE; ROBODOJO_REPO defaults to the checkout
bash install.sh                      # builds the agent image, installs websockets>=13 for the policy server, adds the RGB-D env configs
./roboshell.sh check                 # optional, no GPU: one Codex turn against a fake server, verifies image, endpoint and key
```

`install.sh` copies `env_cfg/arx_x5_rgbd.yml` and `env_cfg/camera_config_rgbd.yml` next to the stock RoboDojo configs; nothing upstream is modified. The policy server runs in the Isaac Sim Python (`ISAAC_PYTHON`), so `<policy_env>` below may be that interpreter's path or any value when `config.env` sets `ISAAC_PYTHON`.

## Data Processing

Unsupported. This is an eval-only adapter: `process_data.sh` is omitted, and it does not convert datasets.

## Training

Unsupported. `train.sh` is omitted. There is no checkpoint: the tools in `tasks/` are the trained artefact, and the model is called through the provider API at evaluation time.

## Evaluation

```bash
cd XPolicyLab/policy/RoboBigWater
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_env> <eval_env>

# Example: stack_bowls, seed 0, simulator and policy server on GPU 0
bash eval.sh RoboDojo stack_bowls gpt-6-astra arx_x5_rgbd joint 0 0 0 unused /path/to/RoboDojo/isaacsim-5.1.0/.venv
```

- `env_cfg_type` is `arx_x5_rgbd` (three 640x480 RGB-D cameras). `action_type` must stay `joint`. `ckpt_name` is a label only; `gpt-6-astra` is the convention.
- Cluttered Generalization variants are task names with the `_random` suffix (`stack_bowls_random`); they share the tools of the base task.
- The policy server and the simulator may share one GPU (about 14 GB together). Batch evaluation is not supported (`eval_batch: false`).
- The agent's full session for each episode (commands, observations, transcript) is written under `runs/bridge/<task>/`. `python3 eval/official_validate.py .` flags episodes in which the agent never issued a command (an infrastructure failure, not a result).
- A typical episode takes 5 to 10 minutes of wall-clock time, almost all of it model latency.

Offline wiring check (no simulator). The policy server starts the real agent, so Docker, the Codex binaries and the endpoint are needed; the agent sees placeholder observations and the loop ends at the debug client's step limit:

```bash
export EVAL_ENV_TYPE=debug
bash eval.sh RoboDojo stack_bowls gpt-6-astra arx_x5_rgbd joint 0 0 0 unused <eval_env>
```

Development without the official client (direct executor, resets and replays, seed-0 layouts):

```bash
./roboshell.sh serve 0 stack_bowls &      # robo-server in direct mode
./roboshell.sh episode stack_bowls 3      # one episode on layout 3, output under runs/episodes/
./roboshell.sh check-task stack_bowls     # delivery check: files present, tools load, manual carries no task words
```

## Configuration

`config.env` (not tracked; see `config.env.example`):

| Variable | Role |
| --- | --- |
| `ROBODOJO_REPO` | RoboDojo checkout; defaults to the one this XPolicyLab is in |
| `ISAAC_PYTHON` | Python of the Isaac Sim 5.1 environment (policy server and direct executor) |
| `CODEX_BIN` | Codex CLI binary; `codex-code-mode-host` must be next to it |
| `MODEL_UPSTREAM`, `MODEL_BASE_PATH` | OpenAI Responses API endpoint and base path |
| `MODEL_KEY_FILE` | file holding the API key; read by the egress proxy on the host only |
| `MODEL`, `EFFORT` | `gpt-6-astra`, `medium` |
| `SIM_CACHE`, `TMPDIR`, `GPU_LOCK`, `EXTRA_LD_LIBRARY_PATH` | optional simulator-side settings |

Extra `deploy.yml` keys: `command_budget` (0 = only the official step limit), `agent_wall_s` (wall-clock cap per episode, 3600), `request_timeout_s` (120).

## Notes

- The server answers each 45 s of agent silence with one hold step, within the client's 120 s request timeout; hold steps cost about 1% of the step budget.
- Tools cannot import the simulator or read object poses: they see joint angles, TCP poses, the three RGB-D images with calibration, and the same motion primitives as the base commands (`roboshell/server/tools.py`, `EpisodeAPI`).
- `agent/INTERFACE.md` is the manual the agent reads; each enabled tool's `interface.md` is appended to it. The delivery check (`eval/check_task.py`) rejects tool text that names the task or its objects.
- Per-seed success counts from our own runs of this adapter on seeds 0--2 are in `eval/official_results.json` (27 task variants, 3,286 episodes); they are not leaderboard entries.
- The server writes each decoded observation as PNG files for the agent (`obs/head.png`, ...), and the task tools read those files back; this is RoboBigWater's own archive inside one process, so the `cv2.imencode` / `cv2.imdecode` calls in `roboshell/server/` and `tasks/*/tools/` never touch XPolicyLab trajectory bits. `model.py` decodes nothing.
- The internal Python package, entry script and run directories keep the development name `roboshell` (`roboshell/`, `roboshell.sh`, `runs/`); the policy name is RoboBigWater.
- Data processing and training are unsupported.
