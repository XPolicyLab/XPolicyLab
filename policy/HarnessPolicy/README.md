# HarnessPolicy

**Contributor:** Harness Policy team | **Paper:** Harness Policy | **arXiv:** TBD | **Original code:** https://github.com/EvolvingLMMs-Lab/the-harness

HarnessPolicy is the Harness Policy agent: a language model (gt-6.1-sol-a) reads the learned
policy's description and hands it whole tasks it was trained on (wall-oss), from the
start pose; for what the policy cannot do (picking the one object a name, a label or a
picture identifies) it writes Python programs that perceive (SAM 3 segmentation, the model's own look at the images), plan and move
the arms through bounded skills. A reactive layer runs guards every control tick and a
fixed shield checks every action.
The agent is frozen in `agent.json` (identity `4b97417504695258`): the policy refuses to run
if the machine it lands on would resolve a different one.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

```bash
cd XPolicyLab/policy/HarnessPolicy
bash install.sh <python of the policy environment>
bash download_checkpoints.sh <python of the policy environment>
export CODEX_BASE_URL=http://HOST:PORT/v1   # model id gt-6.1-sol-a
# export PROXY_OPENAI_API_KEY=...   # only if that server requires a key
```

`the-harness/` holds the-harness at the exported commit (eb936793fa76); install.sh uses it as it is.

## Data Processing

Not supported: this is an evaluation-only adapter (no `process_data.sh`). The learned
policies it calls are RoboDojo's own released checkpoints.

## Training

Not supported (no `train.sh`): the agent has no trained component of its own.

## Evaluation

```bash
bash eval.sh RoboDojo <task_name> HarnessPolicy arx_x5 ee <seed> <policy_gpu_ids> <env_gpu_id> <policy_env> <eval_env>

# Offline wiring check with scripted model replies (no simulator, no model calls)
HARNESS_SCRIPT=$PWD/the-harness/configs/scripts/xpolicylab_check.json EVAL_ENV_TYPE=debug \
  bash eval.sh RoboDojo stack_bowls HarnessPolicy arx_x5 ee 0 0 0 <policy_env> <eval_env>
```

`policy_gpu_ids` may list several GPUs (`2,3`): the services (kinematics, sam3, wall-oss) take
about 28 GB in all; when they do not fit at once, the learned policies and the
world model load on demand, one at a time (the simulator waits while a model loads, so
only wall time changes). `action_type` must be `ee`; `ckpt_name` is not read.

## Configuration

| deploy.yml key | Meaning |
| --- | --- |
| `poll_seconds` | while the model thinks, `get_action` returns an empty chunk after this long (default 90, below the client's 120 s request limit) |
| `result_dir` | where each episode's trace (model requests, programs, events) goes; default `results/` |
| `max_wall_seconds` | wall-time limit of one episode |

Environment: `CODEX_BASE_URL` (the server for `gt-6.1-sol-a`); `PROXY_OPENAI_API_KEY` if the server requires a key. `HARNESS_SCRIPT` replays model replies
instead of calling the model (wiring checks).

## Notes

- A batch of environments (`eval_batch: true`): one program per environment. While any
  of them is still waiting on the model, every chunk comes back empty and the
  simulator does not step. The learned policy's server still serves one environment
  at a time; the lock in `services/policy_server.py` keeps those calls from mixing.
- One fixed agent for every task: the policy uses only the observation XPolicyLab
  hands it (RGB cameras, measured arm joints, link6 poses, the instruction, the last
  gripper commands); it is told neither the task nor its step limit (`task_name` only
  labels the traces). Camera calibration and kinematics come from the robot's
  description (`the-harness/embodiments/arx_x5`, RoboDojo's own cuRobo planner).
- The policy never reads the outcome: an episode that continues after the program
  finished is not taken as a failure; the arms hold until the client ends it.

## This version

One fixed system on every task. The program keeps the episode, and
`run_policy("wall-oss")` is one bounded call inside it. WALL-OSS 1.0 lite is the
only learned policy. Its native end-effector actions are passed through, and
RoboDojo's client does the IK (`action_type=ee`), the same action path as official
WallX. There is no task-name router: `task_name` only labels traces. Five
instruction phrasings the policy cannot read stay on the agent's own tools.
Image questions and `verify` send temperature 1, which this model accepts. A later
policy slice continues from the frames that were stored when steps in between were
not recorded.
