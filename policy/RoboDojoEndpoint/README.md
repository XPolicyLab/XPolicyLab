# RoboDojoEndpoint

**Contributor:** Ro | **Paper / technical report:** Planned for October 5–11, 2026 | **arXiv:** To be provided | **Original code:** Planned for October 5–11, 2026

A system involving agent and code as policy. The system receives instructions, RGB observations and robot state, and returns robot actions. This adapter supports remote evaluation on RoboDojo with `arx_x5`, joint actions and up to eight parallel environments, each with an independent remote session.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

In a Python 3.10+ virtual environment or conda environment, install the adapter dependencies:

```bash
cd XPolicyLab/policy/RoboDojoEndpoint
bash install.sh

# Example, from an existing RoboDojo workspace:
python3 -m venv .venv-endpoint
source .venv-endpoint/bin/activate
bash install.sh
```

The transport requires `websockets>=16,<17`. The simulator is installed separately using RoboDojo's official instructions; CPU debug mode does not require a simulator or GPU. Keep XPolicyLab inside the benchmark workspace, with the benchmark's existing `env_cfg/` directory beside it.

## Data Processing

Unsupported: this is an evaluation-only remote-service adapter. It consumes runtime observations and does not convert or publish datasets. There is no `process_data.sh`.

## Training

Eval-only remote evaluation; this adapter does not run data processing or training, so `process_data.sh` and `train.sh` are omitted for the declared scope. We plan to release the evaluated implementation and technical report during October 5–11, 2026, and meet RoboDojo’s artifact-publication requirements within one week of leaderboard listing. Report links will be added upon release. The remote checkpoint / eval-only arrangement is subject to maintainer review.

## Evaluation

Obtain the service URL privately from the contributor and set `POLICY_ENDPOINT_URL`. Keep that URL out of public logs and repository files.

```bash
cd XPolicyLab/policy/RoboDojoEndpoint
export POLICY_ENDPOINT_URL='wss://<endpoint supplied privately>'
bash eval.sh <bench_name> <task_name> code arx_x5 joint <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_env> <eval_env>

# Eight-environment simulator evaluation:
EVAL_NUM=8 EVAL_NUM_ENVS=8 bash eval.sh RoboDojo stack_bowls code arx_x5 joint 0 0 0 \
  /absolute/path/to/policy-venv /absolute/path/to/eval-venv

# CPU interface check, after setting a real endpoint and environment paths:
EVAL_NUM_ENVS=8 EVAL_ENV_TYPE=debug bash eval.sh RoboDojo stack_bowls code arx_x5 joint 0 0 0 \
  /absolute/path/to/policy-venv /absolute/path/to/eval-venv

# Exercise the shared server's encoded-image decoding as well:
DEBUG_OBS_ENCODED=1 EVAL_ENV_TYPE=debug bash eval.sh \
  RoboDojo stack_bowls code arx_x5 joint 0 0 0 \
  /absolute/path/to/policy-venv /absolute/path/to/eval-venv
```

Leaving `EVAL_ENV_TYPE` unset runs the official RoboDojo simulator client. `code` is a result label; no local checkpoint is loaded and no checkpoint directory is created. A policy-side GPU is unnecessary; the simulator uses the environment GPU argument. Environment arguments accept a conda name or an absolute virtual-environment directory.

The normal scripts start the official XPolicyLab policy server with a transport-only Model adapter. It forwards each environment to an independent remote session and executes those requests concurrently. The simulator uses the official evaluation entry point with an explicit environment count. RGB decoding remains in the shared policy server; the adapter performs no image decoding or channel conversion. `deploy.py` is identical to the upstream demo episode loop. When task metadata is absent, the agent operates from the supplied instruction, RGB observations and robot state. Task names in launch scripts construct the simulator environment. Selected official tasks have been rehearsed with and without task identifiers; this does not imply benchmark-wide equivalence for all inputs or layouts. Supported instructions follow the official task templates; unsupported or ambiguous instructions fail explicitly, and arbitrary paraphrases are not guaranteed.

For applications that instantiate `Model` directly, `model.py` exposes both standard single-environment and batch APIs through the same official client library. Supply `env_cfg_type=arx_x5` and `action_type=joint`; set `POLICY_ENDPOINT_URL`. An optional `task_name` is forwarded using the public RoboDojo convention `action_case_id=<task_name>_case`; otherwise the case identifier is opaque. Call `reset()`, then `update_obs(obs)` before `get_action()`; call `close()` when finished. Already decoded RGB arrays are forwarded without image decoding or channel conversion, and returned action dimensions are validated using the shared robot helper. Evaluation and trial identifiers generated by the direct Model API remain opaque. In batch calls, each observation must carry the official integer `env_idx`. Call `update_obs_batch(obs_list)` and `get_action_batch(env_idx_list)`; returned chunks follow the requested ID order and have equal lengths. Unused action tails are retained. Finished environments are removed when absent from the next observation batch; reset, trial end and server shutdown close the remote sessions. Call `reset()` before switching between single and batch APIs.

## Configuration

- `POLICY_ENDPOINT_URL`: required `ws://` or `wss://` service address, supplied privately. Unavailable services fail the evaluation; there is no dummy-action fallback.
- `EVAL_NUM_ENVS`: simultaneous simulator/debug environments, default `8`, range `1`–`8`. Use `1` for a single environment. `EVAL_NUM` remains the official total episode target; a vectorized round runs all active environments, so use at least eight episodes for eight environments. Direct official-client launches must also configure at most eight simulator environments.
- `DEBUG_EVAL_EPISODES`: CPU debug episodes, default `2`. These synthetic observations validate interface wiring and do not produce a benchmark success rate.
- `DEBUG_OBS_ENCODED`: upstream CPU-client image encoding switch, default `0`.
- `DEBUG_INSTRUCTION`: CPU fixture instruction, default `Stack the three bowls together.`. The fixture uses opaque evaluation/trial IDs and no case identifier. Images and joint states are synthetic; this check does not evaluate physical task success.
- `deploy.yml` retains the standard key set, fixes `policy_name=RoboDojoEndpoint`, and defaults to `joint`, `arx_x5`, `ckpt_name=code`, `eval_batch=true`, and `max_batch_size=8`.
- `serve_pool.sh`: optional eight-listener pool, with ports `19100`–`19107` by default (local bind only). Run `bash serve_pool.sh RoboDojo auto code arx_x5 joint 0 0 /path/to/policy-venv 19100`; the `auto` case label leaves task selection to the remote instruction/RGB handling. These are local adapter ports on the evaluator’s machine, each forwarding to the privately supplied WSS URL. Keep total active environments within the agreed remote capacity; eight listeners do not reserve 64 environments. Public non-loopback binds additionally require network/firewall configuration.
- `setup_eval_policy_server.sh` accepts the standard optional tenth argument for its bind address, default `127.0.0.1`.

## Notes

- Eight environments share the same service URL; separate public TCP ports are not required. Concurrent jobs still require distinct sessions and sufficient total service capacity. Each invocation of `eval.sh` allocates its own local policy-server port. Only transport concurrency changes; every environment invokes the same remote inference implementation.
- This PR prepares the initial remote-evaluation stage. Full evaluated artifacts will follow the benchmark's verified-publication requirements. Maintainer review of eligibility and the remote evaluation procedure is pending.
- The contributor supplies service addresses and any access details through the evaluation contact channel, not in this public adapter.
