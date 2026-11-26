# KnowinAgent

**Contributor:** [zcyverygood](https://github.com/zcyverygood) | **Paper:** Not yet available | **arXiv:** Not yet available | **Original code:** Bundled [knowin_agent/](knowin_agent/)

KnowinAgent 1.0.0 is a bimanual robot-agent system for RoboDojo. A task agent writes short robot programs and delegates local interactions to a second agent. Both use one runtime for motion, perception, observations and execution receipts. Practice produces task skills and lessons; evaluation can use explicitly supplied frozen knowledge. This framework integration supports RoboDojo with `env_cfg_type=arx_x5` and `action_type=joint`. It includes training code but no learned task or shared assets, and does not reproduce the full system's scores.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

This directory bundles the XPolicyLab adapter, Python runtime and framework resources. Run the commands below from `XPolicyLab/policy/KnowinAgent/`; no separate KnowinAgent checkout or export step is required.

Install into the client interpreter and, optionally, a separate policy-server interpreter:

```bash
cd <RoboDojo>/XPolicyLab/policy/KnowinAgent
bash install.sh <client_python> [<server_python>]

# Example: use one prepared environment for both processes
bash install.sh "$VIRTUAL_ENV/bin/python"
"$VIRTUAL_ENV/bin/python" -m knowin_agent.cli --help
```

The client environment needs RoboDojo and its Isaac Sim dependencies. The policy server needs XPolicyLab and its websocket transport. `install.sh` installs the bundled runtime without replacing the simulator's pinned dependencies; the client interpreter must already provide NumPy and Pillow. The runtime and its supplied reference resources are required for evaluation. Learned knowledge is optional.

Simulation additionally needs a Codex CLI with working model access, a SAM3 segmentation service and a MoGe-2 point-map depth service. The service implementations and weights must be installed separately; their reproducible packaging is pending. No service implementations, credentials or weights are bundled here. Configure their locations before simulation:

```bash
export KNOWIN_AGENT_CODEX_BIN=/path/to/codex
export KNOWIN_AGENT_CODEX_HOME=/path/to/codex-home
export KNOWIN_AGENT_SAM3_DIR=/path/to/sam3-service
export KNOWIN_AGENT_DEPTH_DIR=/path/to/depth-service
```

The service launcher expects SAM3's `service.py`, `.venv` and weights in its project directory, and the depth project's `scripts/start.sh` and `.venv`. To use services already managed elsewhere, set `KNOWIN_AGENT_START_SERVICES=0` and provide `KNOWIN_AGENT_SAM3_URL` / `KNOWIN_AGENT_DEPTH_URL`.

## Data Processing

No demonstration dataset conversion is needed. Knowledge is learned through simulator practice and review. The standard entry point explains this and exits without creating a dataset:

```bash
bash process_data.sh --help
```

## Training

Training runs simulator practice, a real review-agent session, and knowledge freezing. The first six arguments follow XPolicyLab's convention; `--task` selects the task and `--out` selects a separate output directory:

```bash
bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id> \
  --task <task_name> --out <new_output_dir>

# Example (requires the simulator, model access and perception services)
bash train.sh RoboDojo no-checkpoint arx_x5 joint 0 0 \
  --task stack_bowls --out /path/to/training-output
```

The default budgets are 150 accepted practice executions, 10,800 agent wall seconds, and 2,400 review-agent wall seconds. Set `--practice-max-executions`, `--practice-wall-seconds` and `--review-wall-seconds` explicitly when needed; `bash train.sh --help` lists all options. The practice seed encodes `bank * 1_000_000 + ordinal`. `--out` must be a new directory outside the policy source. With no `--knowledge-from`, practice starts without task-specific learned experience or a learned shared library. The final reviewed assets are written to `<out>/review/frozen`; pass that directory as `KNOWIN_AGENT_KNOWLEDGE_DIR` for a later evaluation of the same task. Training does not launch Test or promote assets into the source tree. Running this entry point consumes model and simulator resources.

The [training guide](docs/training.md) describes the output receipts and freeze boundaries. Real simulator practice and review have not been validated in this submission.

## Evaluation

```bash
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_python_or_env> <client_venv_or_conda_env>

# Example (requires the simulator, model access and perception services)
bash eval.sh RoboDojo stack_bowls no-checkpoint arx_x5 joint 0 \
  <policy_gpu_id> <env_gpu_id> <policy_python_or_env> <client_venv_or_conda_env>
```

Leave `EVAL_ENV_TYPE` unset or set it to `sim` for normal evaluation. The task/local agents run in the environment-client process. The policy server loads a lightweight adapter; it does not run a learned action model. `deploy.py` reads official observations, uses rig-constant camera calibration, and sends actions through `take_action`. The evaluator owns reset and scoring. If the agent finishes early, the adapter holds the arms until the evaluator ends the episode.

For a CPU transport check using the official XPolicyLab debug client, keep the normal RoboDojo layout: the policy is under `XPolicyLab/policy/KnowinAgent/`, and `env_cfg/arx_x5.yml` is beside the `XPolicyLab/` directory. `XPOLICYLAB_ROOT` can locate an external checkout but does not register an arbitrary policy directory; copy or link this policy into that checkout first.

```bash
EVAL_ENV_TYPE=debug bash eval.sh RoboDojo stack_bowls no-checkpoint arx_x5 joint 0 0 0 \
  <policy_python_or_env> <client_venv_or_conda_env>
DEBUG_OBS_ENCODED=1 EVAL_ENV_TYPE=debug bash eval.sh RoboDojo stack_bowls no-checkpoint arx_x5 joint 0 0 0 \
  <policy_python_or_env> <client_venv_or_conda_env>
```

If the shell inherits an HTTP/SOCKS proxy, add `NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1` to the evaluation command so the local websocket connection bypasses it. This is a per-command setting.

Debug validates any explicitly supplied knowledge, assembly, raw/encoded observations, a client/server round trip and legal hold actions. It starts neither Codex nor perception services. `KNOWIN_AGENT_DRY_RUN=1` is rejected outside debug. Debug results are not RoboDojo scores.

## Model Assets

Paths below are relative to this policy directory.

- `knowledge/index.json`: empty knowledge index; learned skills, lessons, task tools and shared knowledge are withheld.
- `knowin_agent/skills/builtin.py` and `knowin_agent/program/library/`: built-in framework helpers, including basic manipulation and pouring, retained for cold-start execution and training.
- `knowin_agent/assets/robodojo/`: scoring notes, checksum-pinned task recipes and task-source references supplied to the agent.
- `knowin_agent/backends/robodojo/rig_constants.json`: camera and robot rig constants.

`ckpt_name=no-checkpoint` means there is no trained action-network checkpoint. Model API access, the simulator, and separately installed perception services and weights are still required for simulation. Frozen knowledge is optional; no full-system score reproduction is claimed.

## Configuration

`test_config.json` fixes the shipped evaluation configuration: `gpt-6-astra`, `xhigh`, constant calibration, service depth, no learned shared library by default, at most 100 accepted program executions and a 5,400-second agent wall limit. The official evaluator's action and episode limits also apply. Use one environment per client (`eval_batch: false`).

| Variable | Purpose |
| --- | --- |
| `KNOWIN_AGENT_MODEL`, `KNOWIN_AGENT_EFFORT` | Explicit model/effort overrides; record changes when comparing results |
| `KNOWIN_AGENT_PYTHON` | Interpreter for the agent's MCP bridge |
| `KNOWIN_AGENT_CODEX_BIN`, `KNOWIN_AGENT_CODEX_HOME` | Codex executable and authentication/configuration source |
| `KNOWIN_AGENT_SAM3_URL`, `KNOWIN_AGENT_DEPTH_URL` | Perception endpoints; defaults `http://127.0.0.1:8035` and `http://127.0.0.1:8034` |
| `KNOWIN_AGENT_START_SERVICES` | `0` uses externally managed services |
| `KNOWIN_AGENT_KNOWLEDGE_DIR` | Optional explicit self-contained frozen directory for the selected task; absent means cold start |
| `XPOLICYLAB_ROOT` | Optional external XPolicyLab checkout for adapter launch scripts |
| `KNOWIN_AGENT_TRACE_DIR` | Trace output root; keep outputs outside the policy source |
| `KNOWIN_AGENT_VIDEO_STRIDE` | Optional per-program frame recording; `0` disables it |
| `KNOWIN_AGENT_AGENT_ISOLATION` | Optional JSON configuration for containerised agent sessions |
| `KNOWIN_AGENT_AGENT_SANDBOX` | Optional local Codex sandbox override |

## Notes

[Architecture](docs/architecture.md) describes the runtime, knowledge and evaluator boundaries. [Training](docs/training.md) explains practice and review. Validation for this submission is limited to CPU/debug checks; real simulator training and evaluation remain unverified. CPU/debug checks do not establish simulator success or benchmark performance.

The technical report has not yet been written. Publish results and videos separately from the policy source, with the exact source revision and knowledge hashes.
