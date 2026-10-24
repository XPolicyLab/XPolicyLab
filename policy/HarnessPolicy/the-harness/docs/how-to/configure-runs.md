# Configure runs

## Layering: include

A configuration file may `include` others: they merge in order, and the file's own keys
win (tables merge key by key; anything else replaces). Paths in a file (`prompt`,
`agent`, `runtime.task_descriptions`, `runtime.library.path`, and the same keys under
`[overrides.agent]`) are relative to the file that names them.

```toml
# configs/ablations/no-verifier.toml
description = "without the verifier"
include = ["../default.toml"]

[runtime]
verifier = false
```

## Changing one run: --model, --agent, --set

`--model NAME` picks `configs/models/<NAME>.toml` (or a `.toml` file), `--agent FILE` an
agent file in place of `configs/default.toml` (an ablation). `--set key.path=value`
writes into the experiment's tree for one invocation, as often as needed; `agent.` and
`model.` keys go to the experiment's overrides of the agent and the model roles:

```bash
harness run configs/experiments/dev.toml --output runs/x \
  --model grok-medium \
  --set budget.max_cost_usd=2 \
  --set agent.runtime.keep_images=1 \
  --set 'agent.services.policies=["simate", "g05"]' \
  --set model.main.model=z-ai/glm-5 \
  --set model.main.reasoning_effort='"medium"'
```

The value is read as TOML (`2`, `true`, `"text"`, `[1, 2]`, `{k = 1}`), and as a plain
string when it is not valid TOML (`z-ai/glm-5`). In an experiment file, the same
changes are `[overrides.agent]` and `[overrides.model]` tables.

## Tasks and seeds

`tasks` (or `--tasks`) takes task names, separated by commas, and these groups:

| Name | Tasks |
|---|---|
| `all` | the 42 scored RoboDojo tasks |
| `Generalization`, `Memory`, `Precision`, `Long-Horizon`, `Open` | the tasks of one dimension |
| `moving` | the two conveyor tasks |

`seeds` (or `--seeds`) is a list or a string of numbers and ranges: `"0-9"`, `"0,2,5"`,
`"0-3,7"`.

## Layouts

| `layouts` | Seeds are | Use |
|---|---|---|
| `procedural` (default) | development layouts in `dev_layout_dir` (default `<RoboDojo>/harness_dev_layouts/arx_x5`), generated on first use | all development |
| `eval` | layouts of the official pack `layout_pack_seed` (0, 1, 2) | measuring a frozen configuration; strict by default |

## Budgets

```toml
[budget]
max_wall_seconds = 5400      # per episode (default 3600); checked before each model call
max_cost_usd = 4.0           # per episode
max_model_calls = 30         # per episode (default: the agent's budget, 30)
max_total_cost_usd = 20.0    # the whole suite: no new episode once spent
```

## Worker and suite options

```toml
[worker]
record_video = true          # one MP4 per camera in each episode's videos/
ground_truth = false         # write the simulator's object poses to truth/<task>.jsonl (analysis only)

[suite]
boot_timeout = 600           # seconds a worker may take to start
```

## Against a worker that already runs: harness program

```bash
harness serve --factory benches.robodojo.runtime:create_environment \
  --options configs/workers/robodojo_runtime_dev.toml --port 8771      # in another shell
harness program configs/experiments/dev.toml --task stack_bowls --seeds 0,1 --output runs/p
```

`harness program` runs the seeds of one task against the experiment's `endpoint` (or
`--endpoint`) and writes `runs/p/<time>-<id>/seed-<n>/`. The worker serves one task;
its options file names it.

All keys: [configuration reference](../reference/configuration.md).
