# Check a change without a model

A change to a skill, the runtime or a worker can be checked end to end in the
simulator without model calls: replay model replies instead of calling the model.

## Scripted replies

`--script FILE` takes a JSON list of replies, or `{"replies": [...], "verifier": [...]}`
with the verifier's answers too (without them the verifier answers YES):

```json
--8<-- "configs/scripts/run_policy_check.json"
```

```bash
harness run configs/experiments/dev.toml --script configs/scripts/pipeline_check.json --output runs/check
harness program configs/experiments/dev.toml --task stack_bowls --script my_replies.json
```

| Script | Checks |
|---|---|
| `configs/scripts/pipeline_check.json` | state, cameras, a motion, a guard stopping a motion, grippers, a controller, an error report, the verifier, finish |
| `configs/scripts/run_policy_check.json` | the first configured learned policy for 6 s |
| `configs/scripts/run_policy_rules_check.json` | the same policy: a fresh run refused away from the start pose, run after `home()`, continued until the arms stop |
| `configs/scripts/policy_alone.json` | the configured learned policy alone: the whole task from the start pose to the end of the episode (`configs/experiments/policy-alone-dev.toml` measures it on every task; `scripts/policy_card.py` sorts the scores for its description) |
| `configs/scripts/eef_check.json` | the tools by object id on stack_bowls: `locate`, `pick`, `release`, `push`, then `home()` |
| `configs/scripts/xpolicylab_check.json` | an exported XPolicyLab policy under the debug environment |

When the replies run out, the episode finishes (`<finish>scripted replies exhausted</finish>`).

## Replay a recorded episode

```bash
python scripts/replay_script.py runs/dev/general_pickup/seed-0 > replay.json
harness run configs/experiments/dev.toml --tasks general_pickup --seeds 0 --script replay.json \
  --output runs/replay-check
```

`replay_script.py` turns an episode's model replies and verifier answers into a script
and prints the recorded status, reward and steps on stderr. Replaying them on the same
task and development seed runs the same programs against the changed code. Expect
small differences: contact-rich motions are not deterministic from run to run even on
unchanged code.

## Try code against a running worker

```bash
python scripts/dev_console.py --reset stack_bowls 0
python scripts/dev_console.py --code "print(move_eef('left_arm', get_state()['left_ee_pose'], max_seconds=2))"
python scripts/dev_console.py --exec my_program.py --images /tmp/shots
```

`dev_console.py` builds a session with every layer against `--endpoint` (default
`http://127.0.0.1:8771`), runs the code, prints the executions and saves `show()`
images to `--images`. The episode persists between calls; Python variables do not.
`--vlm`, `--segmenter`, `--policy` and `--world-model` take the services' URLs.

## Follow an episode

```bash
python scripts/episode_status.py runs/dev             # the newest episode, one line per turn
```
