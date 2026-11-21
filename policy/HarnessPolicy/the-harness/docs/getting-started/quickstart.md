# Quickstart

From a machine with the environment installed ([Installation](installation.md)) to a
scored episode. Commands run from the repository root after
`source environment/activate.sh`; `mine` is your deploy file
(`configs/deploy/mine.toml`).

## 1. Start the services

```bash
harness up --deploy mine
harness status --deploy mine
```

```text
service  kind       mode      gpu  url                    up    pid
sam3     segmenter  resident  1    http://127.0.0.1:8810  True
simate   policy     resident  1    ws://127.0.0.1:19008   True
```

`harness up` starts the resident services of the deploy, each in its own process group
with its log under the deploy's `state` directory, and leaves alone a service whose port
already answers. On-demand services start later, when a skill first calls them.

## 2. See what a run resolves to

An experiment file names a bench, tasks and seeds, and may pick a model (or an ablation
of the agent). `harness config` resolves it against your machine:

```bash
harness config configs/experiments/dev.toml --deploy mine
```

```text
agent default 1.6.0  identity 3786ebbcd0e640ae  mode program
main model: deepseek-flash (openai_chat)
bench robodojo, 1 tasks x 1 seeds, procedural layouts, deploy mine, strict False
skill         on   why
core          yes
geometry      yes
kinematics    yes
pointing      yes
vlm           yes
segmentation  yes
grasping      yes
memory        yes
verify        yes
motion        yes
manipulation  no   not in this configuration
eef           yes
reactive      yes
scene         yes
handlers      yes
vla           yes
wam           no   no world_model service
uses       service  status
segmenter  sam3
policies   simate
```

(`manipulation`, the pick and place by name, is the `analytic-manipulation` ablation's:
the agent's `pick` takes the object ids of `locate`. `wam`, the world model's
`predict_future`, comes with `--set 'agent.services.world_models=["openwam"]'`. The
agent has no pointer role: `pointing` asks the vlm role, the main model.)

On a machine that lacks a service, the skills that need it drop out, and the table says
why (`no segmenter service`). The identity is the hash of everything that decides what
the run computes; where things run is not part of it.

## 3. Check the pipeline without a model

Replay fixed model replies instead of calling a model. This starts the simulator,
runs a short program that lifts one arm, stops a second lift with a guard, works
the grippers, runs a controller and finishes, and costs nothing:

```bash
harness run configs/experiments/dev.toml --deploy mine \
  --script configs/scripts/pipeline_check.json --output runs/check
```

`harness run` starts the simulator workers itself on the deploy's `[sim]` GPUs and
stops them at the end.

## 4. Run with a model

```bash
harness run configs/experiments/dev.toml --deploy mine --output runs/dev
```

`dev` runs the agent with its default model (DeepSeek V4.1 Flash; set `DEEPSEEK_API_KEY`) on one task and one
development layout, with a $4 cost cap per episode. It prints one line per episode, such
as these two of an evaluation run (agent 1.4.1):

```text
[slot 0] stack_bowls seed 0: episode_ended success=True reward=1.0 calls=2 cost=$0.002 wall=272s
[slot 0] general_pickup seed 4: finished success=True reward=1.0 calls=4 cost=$0.003 wall=245s
```

`episode_ended`: the learned policy, handed the task, finished it and RoboDojo ended the
episode; `finished`: a program called `finish()`.

`harness run` resumes: an episode with a `summary.json` is not run again. Change the
task, the seeds, the model, or anything in the agent for one run with flags:

```bash
harness run configs/experiments/dev.toml --deploy mine --tasks stack_bowls,stack_blocks --seeds 0-4 \
  --model grok-medium --set agent.runtime.keep_images=1 --output runs/dev-5
```

## 5. Look at the results

```bash
harness report runs/dev                 # RoboDojo Score and success rate, per task and dimension
python scripts/view_runs.py runs        # HTML pages next to the traces: runs/index.html
python scripts/serve_runs.py runs 8000  # serve them (videos can seek); http://127.0.0.1:8000/
```

Every episode directory holds `manifest.json` (configuration, lock, identity, code
fingerprint), `events.jsonl` (each model request and reply, each program cell and its
events, the outcome), `summary.json` and `kernel.log`.
[Read a run](../tutorials/read-a-run.md) walks through one.

## Next

- [Quick tour](quicktour.md): one episode from the first model request to the outcome.
- [Run the agent on RoboDojo](../tutorials/run-the-agent.md): development and evaluation layouts, suites, costs.
- [Run on a small machine](../tutorials/small-machine.md): one GPU, or none.
