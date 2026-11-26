# Add a VLA as a learned skill

Any policy XPolicyLab can serve becomes a learned skill (`run_policy`) with one
manifest: XPolicyLab's own policy server runs it, and `run_policy` speaks its protocol,
handing it the observation RoboDojo's evaluation client would. The agent's policy is
Simate (`[services] policies = ["simate"]` in `configs/default.toml`); this tutorial
adds G0.5 next to it, or in its place.

## 1. The manifest

`services/g05.toml`:

```toml
--8<-- "services/g05.toml"
```

| Key | What it is for |
|---|---|
| `protocol = "xpolicylab"` | no `command`: the server is XPolicyLab's `setup_policy_server.py` |
| `[xpolicylab]` | the policy folder under `XPolicyLab/policy/`, the checkpoint, the action type (`joint`, or `ee` for a policy that outputs link6 poses, which `run_policy` solves by IK every step, as for OpenWAM-alpha in `services/openwam-policy.toml`), `overrides` of its `deploy.yml`, `pythonpath` and `cwd` when the policy needs them |
| `vram_gb`, `load_seconds` | measured: placement and swapping use them |
| `client.description` | what the model is told (`list_policies()`), and what it decides the strategy from: the kinds of manipulation the policy does reliably, partly or not at all, what it cannot do, and the signs that it is failing; never task names (step 3) |
| `client.scoped` | the policy keeps a memory per evaluation id: `run_policy` gives each episode its own (Simate) |
| `notes` | for people: where the description came from |
| `[[weights]]` | where `harness fetch` gets the checkpoint (ModelScope, hf-mirror); several entries for several directories |
| `needs_files` | what must exist before it can start |

`{weights}`, `{xpolicylab}`, `{robodojo}`, `{repo}`, `{python}`, `{port}`, `{url}` are
filled in from the deploy.

## 2. Weights and placement

```bash
harness fetch g05 --deploy mine
```

In the deploy, give it a GPU, or put it on demand in one swap group with Simate:

```toml
[services.g05]
gpu = 2
```

```bash
harness doctor g05 --deploy mine
harness up g05 --deploy mine
```

## 3. Its description

The model decides from the description alone, so it says what the policy does when
handed a whole task from the start pose, measured, never what we hope. The descriptions
in `services/` come from development layouts: the policy alone on every task, then its
per-task scores sorted into kinds.

```bash
harness run configs/experiments/policy-alone-dev.toml --set 'agent.services.policies=["g05"]' \
  --script configs/scripts/policy_alone.json --output runs/g05-alone-dev
python scripts/policy_card.py runs/g05-alone-dev
```

`policy_alone.json` hands the configured policy the whole task from the start pose and
lets it run to the end of the episode, with no model calls. `policy_card.py` sorts the
tasks into reliable (mean score at least 0.7), partial (at least 0.1) and unreliable,
with the tasks behind each. Write the description from that, generalised into kinds of
manipulation ("nesting bowls or cups", not `stack_bowls`): the leaderboard allows no
router keyed on tasks, and the model meets tasks it has no line for.

What it decides: the prompt keeps strategy B (the programs' own tools) for picking one
object out of several by what identifies it, among objects that stand still, and hands
every other task to the policy. The kinds choose between A (hand over and supervise:
the policy usually finishes that kind) and C (the policy first, then finish what it
leaves). Say also what the policy cannot do, so that the model keeps those tasks, and
the signs that it has stopped helping.

## 4. Use it in the agent

```toml
# configs/experiments/g05-dev.toml
description = "the agent with Simate and G0.5"
include = ["dev.toml"]

[overrides.agent]
services = { policies = ["simate", "g05"] }
```

Programs then see both in `list_policies()`. `start("A" or "C", kind)` hands the task
to the first one unless it names another (`policy="g05"`), and `run_policy` takes them
by name:

```python
print(list_policies())                     # name, description, arms of each policy
trace = run_policy("g05", max_seconds=8)   # the task's own instruction, both arms
print(trace.event.kind, get_state()["step"])
```

With `policies = ["g05"]` it replaces Simate (another identity, as any change of the
agent).

`run_policy(name, instruction=None, arms=ARMS, max_seconds=None, region=None, until=None,
reset=False, stall_seconds=5.0, from_anywhere=False)` is a bounded execution like any
motion: guards, `until`, `region`, the shield and handlers apply. Every tick the policy
gets RoboDojo's own observation, and each action chunk is executed whole, as RoboDojo's
evaluation client does. A fresh run (the first, another instruction or arms,
`reset=True`) starts only from the start pose the demonstrations start from; elsewhere
it ends at once with `SKILL_ERROR` ("not at the start pose"; `from_anywhere=True` lifts
that for R1). Calls with the same policy, instruction and arms continue where the last
one stopped (the policy keeps its state, and the unused rest of the action chunk is
kept), so slices of `max_seconds` equal one long run. Without `max_seconds` it runs until
the episode ends, `until` holds, or the arms stop moving for `stall_seconds` once they
have moved (a normal end: `COMPLETED`, detail `stopped = "stalled"`). The agent's
`runtime.policy_progress_seconds` (0, off, in the default agent) also ends a run in
which the objects the last locate tracks have not moved 3 cm for that long
(`stopped = "no_progress"`).

## 5. Check it without a model

`configs/scripts/run_policy_check.json` is a scripted program that lists the policies,
runs the first one for 6 s and prints the joints and the wall time. Replay it against a
running worker (the experiment's `endpoint`) with the new policy configured:

```bash
harness program configs/experiments/dev.toml --set 'agent.services.policies=["g05"]' \
  --script configs/scripts/run_policy_check.json --task stack_bowls
```

`configs/scripts/run_policy_rules_check.json` checks the start-pose rule the same way: a
fresh run refused away from the start pose, run after `home()`, continued until the arms
stop.

!!! note "One fixed agent"
    RoboDojo's leaderboard allows an agent that chooses among policies inside one fixed
    system, not a router keyed on task names. The harness never keys anything on task
    names: the model decides from the policies' descriptions.
