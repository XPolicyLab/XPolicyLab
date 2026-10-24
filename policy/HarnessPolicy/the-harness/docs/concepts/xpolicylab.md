# Evaluation through XPolicyLab

RoboDojo's leaderboard evaluates a policy through XPolicyLab: a folder
`XPolicyLab/policy/<Name>/` whose `Model` answers the official client's `reset`,
`update_obs` and `get_action`. Harness Policy is not shaped like that (the model's
programs ask for one control tick at a time; XPolicyLab asks the policy for action
chunks), so support has three parts, each usable on its own:

| Part | Where | What it does |
| --- | --- | --- |
| Observation contract | `core/contract.py`, `runtime.contract` | development runs see exactly what an evaluated policy sees |
| Connector | `connectors/xpolicylab.py`, `connectors/package.py` | the harness as an XPolicyLab `Model` |
| Export | `export/xpolicylab.py`, `harness export xpolicylab` | a policy folder that follows XPolicyLab's adapter standard |

Nothing in the runtime, the skills or the services knows about XPolicyLab; another
evaluator would get a contract, a connector and an export of its own.

## 1. The agent runs under the contract

The official client hands a policy less than our simulator worker reports (checked
against RoboDojo cb2a37d and the leaderboard protocol of 2026-10-08, unchanged in that of
2026-10-14):

| The worker reports | The official client gives | Under `contract = "xpolicylab"` |
| --- | --- | --- |
| measured gripper openings | the last commanded gripper values | state carries the commands; `pick` checks whether the object rose with the jaws (SAM 3), and `holding(arm)`, or a pick whose rise cannot be measured, asks the VLM about the wrist camera's view; prompt feature `measured_gripper` off |
| camera intrinsics and extrinsics | none | from the embodiment's calibration (`embodiments/arx_x5`): head camera fixed in the world, wrist cameras fixed on link6 |
| the step limit | none | left out of the profile and the state; prompt feature `step_limit` off; `run_policy` without `max_seconds` runs until the episode ends or the arms stop moving |
| the task name | none (only the instruction) | no task descriptions (the `with-task-description` ablation adds them in simulation and does not export) |
| the outcome | none (the host scores) | the policy never reads it |
| fk, ik, plan | none | the kinematics service: RoboDojo's own cuRobo planner without the simulator (`services/kinematics.toml`) |

The agent runs under that contract by default (`contract = "xpolicylab"` in
`configs/default.toml`), on development layouts as on evaluation layouts: the suite
starts its workers with `--contract xpolicylab --embodiment arx_x5`, and the worker
filters everything it returns. Development results are therefore what the exported
policy should reach. The contract is part of the lock; `--set
agent.runtime.contract=native` lets a development run see everything the simulator
measures (another identity).

```bash
harness run configs/experiments/dev.toml --deploy mine --output runs/dev
```

The kinematics service answers exactly as the simulator worker does (same planner
code; checked on general_pickup: FK, IK joints and planned waypoints identical).

## 2. Export

```bash
harness export xpolicylab --name HarnessPolicy \
    --out <RoboDojo>/XPolicyLab/policy/HarnessPolicy --deploy mine [--vendor] \
    [--check --gpu 5 --env harness-policy]
```

The agent (`configs/default.toml` unless another file is given) resolves strictly (every
skill and service it names must be runnable), must run under the contract, and may not
use task descriptions. The export refuses uncommitted code unless
`--allow-dirty`. It writes:

```text
policy/HarnessPolicy/
├── model.py               Model(ModelTemplate) -> harness_policy.connectors.package.HarnessModel
├── deploy.py, eval.sh, setup_eval_policy_server.sh, setup_eval_env_client.sh, __init__.py
│                          copied unchanged from demo_policy
├── deploy.yml             the standard keys (protocol ws, action_type joint, eval_batch true)
│                          and poll_seconds, result_dir, max_wall_seconds
├── agent.json             what the agent was resolved from (paths relative to the
│                          checkout), its lock and identity, the code fingerprint, the commit
├── deploy.toml            where weights are; optionally where a service runs instead
├── install.sh             clones the-harness at the commit (unless vendored), installs XPolicyLab
├── download_checkpoints.sh  `harness fetch` of the services' weights into checkpoints/
├── README.md              XPolicyLab's adapter template
└── the-harness/           with --vendor: the files the agent was resolved from
```

`--check` runs XPolicyLab's PR checks: `bash -n` on every script, `py_compile`, no
image decoding in `model.py`/`deploy.py`, and the closed loop of XPolicyLab's debug
environment (`EVAL_ENV_TYPE=debug`, 10 episodes, scripted model replies from
`configs/scripts/xpolicylab_check.json`, no model calls).

## 3. What runs on the evaluator's machine

`setup_policy_server.py` imports `policy/HarnessPolicy/model.py`, whose `Model` builds
a `HarnessModel`:

1. Resolves `agent.json` again, strictly, against a deploy written for this instance,
   and refuses to run unless the identity equals the exported one; warns when the code
   fingerprint differs. A hosted model without its API key stops it here.
2. Places the services on the GPUs the server sees (`policy_gpu_id`, comma-separated
   for several), by their free memory, on ports of its own (several instances may run
   on one machine). When they do not all fit: policies and world models load on demand
   in one swap group; then every model but the kinematics. The agent's models
   (kinematics, SAM 3, Simate: 23.2 GB) fit one 24 GB GPU resident; when a GPU has less
   room, the simulator waits while a model loads, so only wall time changes.
3. Starts the resident services, the broker of the on-demand ones, and a reaper that
   stops them all if the server dies without cleaning up.
4. Per episode, on the first observation after `reset`: runs the driver (kernel,
   skills, model loop) against a `Bridge`, which is a robot worker to the session and
   the `Model` to XPolicyLab. Programs run in the kernel's sandbox where the machine
   allows user namespaces (`isolation = "auto"`); in a container that forbids them, the
   guard inside the kernel still limits programs to their services and forbids
   processes.

The bridge meets the two loops: a tick puts one action up for `get_action` and waits
for the observation after it (the client sends one after every action), so the
reactive layer sees every observation. While the model thinks or a model loads,
`get_action` returns an empty chunk after `poll_seconds` (90 s, under the client's
120 s request limit); the standard episode loop then takes no action and asks again,
and the simulator does not advance. After the program finishes, the arms hold until the
client ends the episode. A `reset` ends the running program's ticks (they report the
episode over) and returns at once.

Each instance writes its traces to `results/<task>-<time>-<pid>/`: the instance's
deploy, the lock, the service logs, and one directory per episode with the model
requests, programs, events and summary.

## Checks done (2026-10-12)

- `tests/test_contract.py`: the contract filter, and prompts and resolution per
  contract. `tests/test_xpolicylab.py`: package placement (4, 2, 1 GPUs, too little
  memory, external services, distinct ports), the bridge under the standard loop (empty
  chunks, holding, a reset ending a waiting tick), the export, and an exported policy
  for three episodes under XPolicyLab's loop with double resets, the last stopped
  mid-program.
- `harness export ... --check` on one 24 GB GPU: every check passed; the package
  placed itself on one GPU (kinematics resident, four models on demand) and finished
  10/10 debug episodes.
- RoboDojo's official client (`eval.sh`, Isaac Sim, grok-4.7 low) against the exported
  policy on a development layout served as eval seed 9000 (no evaluation layout used):
  200/200 steps through the bridge, the client ended the episode, the episode's summary
  was written and every service the policy started was stopped. A wiring check, not a
  result ([how to run it](../tutorials/submit-xpolicylab.md#4-run-the-official-client)).

## Checks done for agent 1.4.1 (2026-10-15)

The configuration 1.5.0 made the default (no pointer role, no world model), exported
with `--vendor` and the two `--set`: identity checked at start, the services placed resident on one GPU
(kinematics, SAM 3, Simate: 23.2 GB).

- `--check`: XPolicyLab's PR checks and the debug closed loop passed.
- RoboDojo's official client with DeepSeek V4.1 Flash on one development layout per
  task served as eval seed 9000: `stack_bowls`, `general_pickup` and `make_toast`
  succeeded, `solve_equation` did not; no request reached the client's 120 s limit.
- Agent 1.4.2, which hands the conveyor tasks to the policy, was exported the same way
  (identity `cf7c5c6cbaf6ed9d`) for the full local evaluation through the official
  client.

## Submitting

[Submit to RoboDojo's leaderboard](../tutorials/submit-xpolicylab.md) goes through it step
by step: develop under the contract, export, check, run the official client, and open
the pull request.
