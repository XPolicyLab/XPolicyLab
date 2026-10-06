# EmbodiedRSI

**Contributor:** [EinsiaAI](https://lab.einsia.ai/) | **Paper:** Coming soon | **Original code:** [EmbodiedRSI](https://github.com/Einsia/EmbodiedRSI)

EmbodiedRSI is a **self-evolving robotic agent** operating in an **SWE-style workspace**. It writes, executes and revises Python programs to control robots.

It has two phases:

1. **Playground:** The agent explores the environment, iteratively develops control programs, and consolidates its experience into Skills and Lessons.
2. **Test:** The agent uses the frozen Skills and Lessons to solve tasks in new scenes, adapting its programs to live feedback within each episode.

This **eval-only adapter** provides the evaluation runtime and Agent Code Workspace for Test. The bundled Skills and Lessons are empty; a learned workspace can be supplied separately.

Shared argument and deployment conventions are in the [XPolicyLab README](../../README.md).

## Installation

Use Linux, Python **3.12+** and Docker. Place this release at `<RoboDojo>/XPolicyLab` in the public [RoboDojo workspace](https://github.com/RoboDojo-Benchmark/RoboDojo). Reuse your installed simulator and assets.

Run from this policy directory:

```bash
EMBODIEDRSI_PYTHON=python3.12 bash install.sh
bash runtime/docker/build.sh

export EMBODIEDRSI_CODEX_HOME="$PWD/.venv/provider"
mkdir -p "$EMBODIEDRSI_CODEX_HOME"
cp runtime/agent/codex.config.example.toml "$EMBODIEDRSI_CODEX_HOME/config.toml"
# Edit base_url in config.toml if using a compatible provider.
read -rsp 'API key: ' OPENAI_API_KEY; echo
export OPENAI_API_KEY
```

The provided [Dockerfile](runtime/docker/Dockerfile) builds `embodiedrsi-agent:xpolicy` from public sources and downloads the complete [Codex 0.159.2 release](https://github.com/openai/codex/releases/tag/rust-v0.159.2). No prebuilt private image or host Codex installation is needed. For a clean rebuild, add `embodiedrsi-agent:xpolicy --no-cache` to the build command.

The config defaults to `https://api.openai.com/v1`; your account must support `gpt-6-astra`, Responses API, images and tool calls. This uses the copied example and your API key, without reading your personal Codex configuration.

## Data Processing

Not required. This adapter consumes live observations and workspace files; `process_data.sh` is omitted.

## Training

Unsupported. `train.sh` is omitted. There is no checkpoint directory and no training release to schedule. Actions are produced at eval time by the provider API.

## Evaluation

Run from this directory in an official RoboDojo workspace, with XPolicyLab beside `env_cfg/` and `scripts/`. The agent uses the configured model API; only the simulator needs a GPU.

Each Test episode launches a Codex agent in Docker with a fresh workspace mounted at `/workspace`. The agent submits Python programs from that workspace; the host executes them and exchanges robot actions and observations with RoboDojo.

Standard evaluation entry:

```bash
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_env> <eval_env>

# Example: Test on stack_bowls
EVAL_NUM=1 EVAL_ENV_TYPE=sim bash eval.sh RoboDojo stack_bowls embodiedrsi_astra_xhigh arx_x5 ee 1 0 0 \
  "$PWD/.venv" /path/to/robodojo-env
```

`EVAL_NUM=1` runs one complete episode; unset it for the full benchmark. Completion is marked by `[MAIN] eval finished`; task success is scored separately.

`env_cfg_type` is `arx_x5`; `action_type` accepts `ee` or `joint`. `ckpt_name=embodiedrsi_astra_xhigh` selects the bundled [Workspace](Workspace/); use a workspace path to load another set of Skills and Lessons. No neural checkpoint is required.

`policy_env` (arg 9) and `eval_env` (arg 10) accept a virtualenv directory, Python executable, conda environment name or `current`. Use the installed policy environment for arg 9 and the RoboDojo simulator environment for arg 10. Set `eval_batch: true` in `deploy.yml` for independent agents across parallel environments.

Offline interface check (no model API, Docker or simulator):

```bash
EVAL_ENV_TYPE=debug bash eval.sh RoboDojo stack_bowls diagnostic arx_x5 ee 1 0 0 \
  "$PWD/.venv" "$PWD/.venv"
```

This runs a deterministic Python probe through the action/observation loop. Re-run with `DEBUG_OBS_ENCODED=1` to check server-side image decoding; a successful run ends with `[MAIN] eval finished`.

Additional checks:

```bash
.venv/bin/python -m unittest discover -s tests -v

# Both debug image modes, using robot configuration from another installation
.venv/bin/python tests/run_debug_checks.py \
  --env-cfg /path/to/RoboDojo/env_cfg --run-dir /path/to/new/debug-run
```

Add `--batch` to the debug runner to check parallel episode workspaces. For split-machine evaluation, see the [deployment guide](../../README.md#-deployment-flow); the server/client scripts accept an optional final `bind_host`/`server_host`.

## Configuration

Policy-specific settings in `deploy.yml`:

| Key | Default / role |
| --- | --- |
| `model`, `reasoning_effort` | `gpt-6-astra`, `xhigh` |
| `agent_image` | `embodiedrsi-agent:xpolicy`; override with `EMBODIEDRSI_AGENT_IMAGE` |
| `agent_cpus`, `agent_memory_mb` | 4 CPUs / 8192 MB per agent |
| `execution_budget` | 100 code executions per episode |
| `submission_timeout_sec` | 1200 seconds per code execution |
| `feedback_fps` | 25 frames per second in execution feedback videos |
| `agent_recovery` | Retry interrupted model connections up to 3 times |
| `run_dir` | Fresh directory under `XPolicyLab/runs/EmbodiedRSI/`; override with `EMBODIEDRSI_RUN_DIR` |
| `diagnostic` | Enabled by `EVAL_ENV_TYPE=debug` to use the Python probe |
| `ws_ping_interval_s`, `ws_ping_timeout_s` | 20 / 120 seconds |

The simulator sets the native action limit; there is no agent wall-clock cap. Each run saves its configuration, input hashes, episode workspaces and execution feedback in the run directory.

## Notes

- Task experience belongs in `Workspace/skills/<task>/` and `Workspace/lessons/<task>/`; `*_random` variants use the base task name. These directories currently contain placeholders; add the frozen files before evaluating with learned experience.
- Each Test episode starts with a fresh agent and workspace. Python and robot state persist between submissions; Skills and Lessons stay read-only, and agent-initiated reset is disabled. The official evaluator controls episode boundaries and scoring.
- `deploy.py` forwards terminal observations to waiting Python programs so their final `step(action)` completes. Debug checks verify this interface; task success is measured in the official simulator.
