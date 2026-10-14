# EmbodiedEvo

**Contributor:** [Einsia.AI](https://lab.einsia.ai/) | **Paper:** [arXiv:2610.18867](https://arxiv.org/pdf/2610.18867) | **Project Page:** [EmbodiedEvo](https://lab.einsia.ai/embodiedevo/) | **Original code:** [EmbodiedEvo](https://github.com/Einsia/EmbodiedRSI)

EmbodiedEvo is a **self-evolving robotic agent** operating in an **SWE-style workspace**. It writes, executes and revises Python programs to control robots.

It has two phases:

1. **Playground:** The agent explores the environment, iteratively develops control programs, and consolidates its experience into Skills and Lessons.
2. **Test:** The agent uses the frozen Skills and Lessons to solve tasks in new scenes, adapting its programs to live feedback within each episode.

This adapter includes the [Playground source and execution workflow](playground/README.md), the Test runtime, and frozen Playground experience for all 42 tasks: 293 skill files and 165 lesson files. Its XPolicyLab policy identifier and directory name are `EmbodiedEvo`.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

Use Linux, Python **3.12+** and Docker. Place this release at `<RoboDojo>/XPolicyLab` in the public [RoboDojo workspace](https://github.com/RoboDojo-Benchmark/RoboDojo). Reuse your installed simulator and assets.

Run from this policy directory:

```bash
EMBODIEDEVO_PYTHON=python3.12 bash install.sh
bash runtime/docker/build.sh

export EMBODIEDEVO_CODEX_HOME="$PWD/.venv/provider"
mkdir -p "$EMBODIEDEVO_CODEX_HOME"
cp runtime/agent/codex.config.example.toml "$EMBODIEDEVO_CODEX_HOME/config.toml"
# Edit base_url in config.toml if using a compatible provider.
read -rsp 'API key: ' OPENAI_API_KEY; echo
export OPENAI_API_KEY
```

The provided [Dockerfile](runtime/docker/Dockerfile) builds `embodiedevo-agent:xpolicy` from public sources and downloads the complete [Codex 0.159.2 release](https://github.com/openai/codex/releases/tag/rust-v0.159.2). No prebuilt private image or host Codex installation is needed. For a clean rebuild, add `embodiedevo-agent:xpolicy --no-cache` to the build command.

The config defaults to `https://api.openai.com/v1`; your account must support `gpt-6-astra`, Responses API, images and tool calls. This uses the copied example and your API key, without reading your personal Codex configuration.

To explicitly reuse the provider and authentication of your local Codex instead,
set this before launching the server:

```bash
export EMBODIEDEVO_CODEX_HOME="${CODEX_HOME:-$HOME/.codex}"
```

Each episode copies `auth.json` and the provider/model settings from that directory
into a temporary container home. Personal plugins, MCP servers, project settings
and conversation history are excluded. The model and reasoning effort are selected
by `deploy.yml`; keep them aligned with the provider's supported models. If the
provider declares `env_key`, export that variable in the server environment too.

## Data Processing

Not required. This adapter consumes live observations and workspace files; `process_data.sh` is omitted.

## Training

There is no gradient-based training or neural checkpoint. `train.sh` is omitted. Playground learns reusable code and lessons through live interaction; both phases call the configured model API.

## Playground

The bundled [source](playground/) comes from the EmbodiedEvo repository supplied as
`EmbodiedRSI`. It includes the orchestrator, persistent Python sessions, RoboDojo
adapters, task templates, provisioning scripts and CPU tests. See its
[source manifest](playground/SOURCE_MANIFEST.json) for the upstream revision and
file hashes. Runtime assets and environments are provisioned separately.

Run these commands from `policy/EmbodiedEvo/`, after Installation above. Prepare
an independent simulator snapshot using a working RoboDojo checkout and conda
environment (this copies the assets and clones the environment):

```bash
bash playground/scripts/prepare_robodojo.sh /path/to/RoboDojo /path/to/conda/envs/RoboDojo
```

The default [task configuration](playground/tasks/RoboDojo/task.toml) uses
`gpt-6-astra`, `xhigh`, the same `embodiedevo-agent:xpolicy` image as Test, a fixed
Playground scene and 100 execution requests. Edit it before a new run when needed.
`EMBODIEDEVO_CODEX_HOME` selects the same external provider configuration as Test.
The Playground launcher defaults to this policy's `.venv/bin/python`; override
it with `EMBODIEDEVO_PLAYGROUND_PYTHON` to use a separate host interpreter.

```bash
PLAYGROUND_RUN="$PWD/playground/runs/$(date +%Y%m%d%H%M)_stack_bowls"
bash playground.sh explore --task RoboDojo --env-task stack_bowls \
  --run-dir "$PLAYGROUND_RUN" --gpu 0

# Export only the verified frozen Skills/Lessons into a fresh Test workspace.
TEST_WORKSPACE="${PLAYGROUND_RUN}_workspace"
bash playground.sh export --run-dir "$PLAYGROUND_RUN" --output "$TEST_WORKSPACE"

# Use the exported directory as ckpt_name in the existing official Test entry.
EVAL_NUM=1 EVAL_ENV_TYPE=sim bash eval.sh RoboDojo stack_bowls "$TEST_WORKSPACE" \
  arx_x5 ee 1 0 0 "$PWD/.venv" /path/to/robodojo-env
```

Playground allows same-scene `reset()` and writable Skills/Lessons. Only those two
directories transfer; Test starts with fresh programs, observations and agent
state, and frozen knowledge. Export rejects incomplete runs, modified frozen
files and existing output directories. Public recipes remain supplied knowledge;
setting `EMBODIEDEVO_USE_RECIPE=0` before exploration also omits the recipe from the
exported Test instruction. Keep Test scenes disjoint from the Playground scene.

`bash playground.sh test ...` and `bash playground.sh run ...` retain the upstream
direct-simulator Test and combined workflow. Their reports differ from the
official `eval.sh` reports. See [protocol details](playground/docs/PROTOCOL.md).

CPU checks for the bundled source (no simulator, Docker or model calls):

```bash
(cd playground && ../.venv/bin/python -m unittest discover -s tests -v)
```

## Evaluation

Run from this directory in an official RoboDojo workspace, with XPolicyLab beside `env_cfg/` and `scripts/`. The agent uses the configured model API; only the simulator needs a GPU.

Each Test episode launches a Codex agent in Docker with a fresh workspace mounted at `/workspace`. The agent submits Python programs from that workspace; the host executes them and exchanges robot actions and observations with RoboDojo.

Standard evaluation entry:

```bash
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_env> <eval_env>

# Example: Test on stack_bowls
EVAL_NUM=1 EVAL_ENV_TYPE=sim bash eval.sh RoboDojo stack_bowls embodiedevo_astra_xhigh arx_x5 ee 1 0 0 \
  "$PWD/.venv" /path/to/robodojo-env
```

`EVAL_NUM=1` runs one complete episode; unset it for the full benchmark. Completion is marked by `[MAIN] eval finished`; task success is scored separately.

`env_cfg_type` is `arx_x5`; `action_type` accepts `ee` or `joint`. `ckpt_name=embodiedevo_astra_xhigh` selects the bundled [Workspace](Workspace/); use an exported workspace path to load newly learned Skills and Lessons. No neural checkpoint is required. The old `EmbodiedRSI` policy/import path and `EMBODIEDRSI_*` settings have been replaced by `EmbodiedEvo` and `EMBODIEDEVO_*`; update launch commands and rerun `install.sh` if an existing virtualenv still refers to the old directory.

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

### One service for multiple tasks

Use `all` as the server's task argument to expose the bundled tasks through one
WebSocket address. The server snapshots each task's recipe, Skills and Lessons at
startup. It creates **no agent process, Docker container or episode workspace**
until that task has received an observation and the client first calls
`get_action` (or batched inference), preserving the single-task launch behavior.
Provider configuration, the agent image and robot metadata are still required
for real inference; listening on the port alone does not validate those services.

```bash
# Server: run from this policy directory after the Installation steps.
EVAL_ENV_TYPE=sim bash setup_eval_policy_server.sh \
  RoboDojo all embodiedevo_astra_xhigh arx_x5 ee 1 \
  0 "$PWD/.venv" 19000 0.0.0.0

# Client: run from the same policy directory in the remote RoboDojo workspace.
EVAL_NUM=1 EVAL_ENV_TYPE=sim bash setup_eval_env_client.sh \
  RoboDojo stack_bowls embodiedevo_astra_xhigh arx_x5 ee 1 \
  0 /path/to/robodojo-env "ckpt_name=embodiedevo_astra_xhigh,action_type=ee" \
  19000 <server-ip>
```

Another client can use `push_T` at the same address. The official RoboDojo client
already sends `action_case_id=<task>_case`, so its RPC sequence is unchanged.
The server also accepts explicit `payload.task_name`; clients with a custom case
ID must send it on their first `RESET` or `PREPARE_CASE`. Subsequent requests use
the bound `(evaluation_id, action_case_id)`. Conflicting metadata and unknown
tasks return protocol errors. `*_random` variants route to the base task's agent
and frozen experience; the client still selects the actual simulator variant.

The 42 task names are **workspace templates, not a container limit**. Every
`(evaluation_id, action_case_id)` has independent model state; each `env_idx` and
episode within it gets a separate worker and container. Thus two clients running
`stack_bowls` with different layouts can run simultaneously; even the same layout
in independent evaluations is isolated. The official client now sends
`environments: [{env_idx: 0, layout_id: <seed>}]` for single-environment episodes
(the batched client already supplies these fields). Legacy clients without
layout metadata remain accepted; their episodes are isolated but logged with an
unknown layout. IDs and layout fields are WebSocket protocol metadata, not HTTP
request headers, and are not exposed to the coding agent.

Concurrent clients must use distinct evaluation IDs (or distinct case IDs with
explicit task metadata). The client launcher generates a UUID-based
`ROBODOJO_RUN_ID` unless one is supplied. Retain that ID when reconnecting the
same episode. Using the same evaluation/case IDs means **resume the existing
session**, not launch another layout. With `eval_batch: true` on both peers, one
case can contain up to 16 independent environment agents.

All layouts of a task copy the same startup snapshot of its prompt, recipe,
Skills, Lessons, primitives and scripts. Initial workspace contents are identical;
observations, Python state, `scratch/` and submissions are private to each episode.
Skills and Lessons are mounted read-only. No writable directory is shared across
layouts, so their programs and observations cannot overwrite one another.

`server_max_active_agents` defaults to **256 across all tasks and layouts**.
`prepare_case` reserves capacity for each environment atomically, before any
container starts. Exhaustion returns a retryable error; retry preparation later
with a new message ID. Startup has 16 lanes, each held until Docker reports the
container running. Queued/in-progress starts return empty action chunks while
WebSocket heartbeats remain responsive. Cancellation removes queued starts;
startup failure/timeout retires the case and returns its capacity after cleanup.
Model calls and cleanup use separate bounded thread pools; API waiting does not
hold one thread per client. Calls within each case remain sequential; overlapping
calls with different IDs are rejected, while retries with the same ID share the
original execution. Receive queues, frame size, connection/session counts and
total in-flight requests are also bounded.

`reset` clears only that client case's episode state. `trial_end` retires its completed
workers; the next episode gets a fresh workspace and agent. An explicit protocol
`CLOSE` retires the cases still owned by that connection. A dropped network
connection preserves the agents and Python state for
`server_disconnect_grace_sec` (**900 seconds** by default), measured after
WebSocket detects the disconnect. Reconnect with the same evaluation/case IDs
and retry the same message ID to continue. A newer connection takes ownership;
late calls or `CLOSE` from the old connection cannot operate on its agent.
After the grace period, the server closes the abandoned agents and releases
capacity; that case is expired and must restart with a new evaluation ID.
Keep WebSocket pings enabled so half-open connections can be detected.

The server shields in-flight execution from connection cancellation and caches
the latest 64 replies per case, keeping active cases separate from busy peers.
This covers the bundled sequential client's immediate retry, including a lost
action response; it is not an unbounded history for replaying old requests.
Request timeouts remain fatal for that trial: do not issue the uncertain action
again under a new message ID. A server process restart loses episode state; the
client detects its changed server ID and aborts instead of continuing silently.
Server shutdown closes active workers and containers. Connected but idle cases
expire after 1800 seconds, and an episode has a 7200-second wall timeout.
Unexpected worker death is reaped; a supervisor watchdog terminates orphaned
workers if the server dies. Failed Docker cleanup retains the capacity reservation
and is retried rather than admitting replacement containers without bounds.
Connection recovery and expiration are recorded in `connections.jsonl`.

`server_tasks: null` enables every base task in the workspace's recipe manifest
(42 in the bundled release); set it to a list such as `[stack_bowls, push_T]` to
serve a subset. Runs are saved under one fresh server directory, with
`server.json`, `service-status.json`, `routes.jsonl`, and separate
`tasks/<task>/sessions/<session>/episodes/` trees. `session.json` and
`episode-map.jsonl` record the client IDs and layout mapping.
`EMBODIEDEVO_RUN_DIR`, when set, names that fresh server directory.

For an offline service check, use `EVAL_ENV_TYPE=debug` and `ckpt_name=diagnostic`
on the server and client commands above, with the policy `.venv` on both sides.
This uses the same routing and worker lifecycle with deterministic probes; it
does not launch Docker or call a model API. Passing a concrete task instead of
`all` to the server script retains the original single-task entrypoint.

### Startup load check without model charges

The opt-in load runner opens real WebSockets, sends three encoded 640×480 cameras
per client, starts one real Docker container per layout, and checks the first
action is routed back to the matching client. It holds all containers concurrently
before testing explicit close and abrupt-disconnect cleanup. Containers use the
production worker, workspace, RPC bridge and lifecycle, with a deterministic
agent command, `--network none`, and no mounted provider credentials. It does not
run full episodes or call a model API.

```bash
# From the XPolicyLab checkout, after installing the policy and Docker image.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 policy/EmbodiedEvo/.venv/bin/python -m \
  XPolicyLab.policy.EmbodiedEvo.tests.run_server_load \
  --clients 256 --run-dir /tmp/embodiedevo-load-unique-run
```

The report records concurrent running containers, first actions, heartbeat
latency, workspace hashes and cleanup. This validates service startup and routing
capacity; full Codex inference throughput also depends on provider quotas and host
resources. Production defaults cap each agent at 4 CPUs / 8192 MB; 256 agents could
reach 2 TiB of memory. The lightweight load probe uses 1 CPU / 256 MB per container.
Increase `server_max_active_agents` above 256 only with corresponding host and API
capacity; simulator resources belong to the clients.

## Configuration

Policy-specific settings in `deploy.yml`:

| Key | Default / role |
| --- | --- |
| `model`, `reasoning_effort` | `gpt-6-astra`, `xhigh` |
| `agent_image` | `embodiedevo-agent:xpolicy`; override with `EMBODIEDEVO_AGENT_IMAGE` |
| `agent_cpus`, `agent_memory_mb` | 4 CPUs / 8192 MB per agent |
| `execution_budget` | 100 code executions per episode |
| `submission_timeout_sec` | 1200 seconds per code execution |
| `feedback_fps` | 25 frames per second in execution feedback videos |
| `agent_recovery` | Retry interrupted model connections up to 3 times |
| `run_dir` | Fresh directory under `XPolicyLab/runs/embodiedevo/`; override with `EMBODIEDEVO_RUN_DIR` |
| `diagnostic` | Enabled by `EVAL_ENV_TYPE=debug` to use the Python probe |
| `ws_ping_interval_s`, `ws_ping_timeout_s` | 20 / 120 seconds |
| `server_tasks` | `null`: all manifest tasks when the server task argument is `all`; otherwise a nonempty list of task names |
| `server_max_active_agents` | 256 reserved environment agents across all tasks/layouts; positive integer |
| `server_disconnect_grace_sec` | 900 seconds to resume a disconnected case |
| `server_max_sessions`, `server_max_connections` | 1024 / 1024 |
| `server_max_inflight_requests` | 4096 total requests; one distinct task request per case |
| `server_startup_parallelism`, `server_startup_timeout_sec` | 16 concurrent starts / 120 seconds per start |
| `server_idle_session_sec`, `server_episode_timeout_sec` | 1800 seconds idle / 7200 seconds per episode |
| `server_model_threads`, `server_cleanup_threads` | 64 / 16 separate executor threads |
| `server_poll_interval_sec` | 0.05 seconds minimum empty-action response wait |
| `server_max_message_bytes` | 67108864 (64 MiB); receive queue 2, pending replies 4 per connection |

The simulator sets the native action limit. The shared service additionally enforces the episode wall timeout; the single-task entrypoint retains its original behavior. Each run saves its configuration, input hashes, episode workspaces and execution feedback in the run directory.

## Notes

- The bundled experience comes from Playground run `202610052245_robodojo_D_recipe_astra_xhigh_v01_a01` (`gpt-6-astra`, `xhigh`). [The manifest](Workspace/experience-manifest.json) records each task's variants and file hashes. All files are included in this repository; the original run directory is unnecessary.
- At startup, `Workspace/skills/<task>/` and `Workspace/lessons/<task>/` are copied into that episode's `skills/` and `lessons/`. Only the selected task is loaded; `*_random` variants use the base task's experience. The default `ckpt_name=embodiedevo_astra_xhigh` enables this automatically.
- Observation numbers in the frozen experience refer to past Playground evidence. Those historical logs are not runtime dependencies; each Test episode receives fresh observations.
- Each Test episode starts with a fresh agent and workspace. Python and robot state persist between submissions; Skills and Lessons stay read-only, and agent-initiated reset is disabled. The official evaluator controls episode boundaries and scoring.
- `deploy.py` forwards terminal observations to waiting Python programs so their final `step(action)` completes. Debug checks verify this interface; task success is measured in the official simulator.
