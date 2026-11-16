# Running RoboDojo

Complete [installation](INSTALL.md) and [Codex API configuration](CODEX_API.md),
then run from the repository root. Use an
explicit new `--run-dir` for each experiment. Directory prefixes use
`YYYYMMDDHHMM`; append a descriptive suffix to distinguish same-minute runs.

## Local Playground and Test

```bash
RUN_DIR="runs/$(date +%Y%m%d%H%M)_stack_bowls"
.venv/bin/embodiedevo run --task RoboDojo --env-task stack_bowls \
  --run-dir "$RUN_DIR" --gpu 0
```

`explore` and `test` can also be run separately against the same directory.
The configured default is 25 Test scenes. `--test-scenes N` sets a run-specific
count; use the same count for separate explore and test commands. Counts and
task settings are pinned, so do not change them during resume. A small smoke or
subset run is not a full benchmark evaluation.

```bash
.venv/bin/embodiedevo test --task tasks_baseline/RoboDojo --env-task stack_bowls \
  --run-dir "runs/$(date +%Y%m%d%H%M)_stack_bowls_baseline" --gpu 0
```

The default templates explicitly select a model and reasoning effort; use values
supported by your provider. No provider URL or key is included. Model credentials
remain in an external CLI home.

## Native official evaluator

After obtaining a frozen Playground run, use the
[XPolicyLab integration](../integrations/robodojo_policy/README.md) to run the
native evaluator. The direct session runner and native evaluator produce
different report formats. Keep their provenance and scope explicit when
comparing results. Full official scene counts and aggregation rules come from
the selected upstream benchmark revision, not from a short local diagnostic.

## Concurrency and output

Start with one simulator per GPU. `--test-gpus 0,1 --test-gpu-slots 1` enables
bounded concurrent Test scenes on resources already assigned to you. The optional
SIF route validates the selected physical GPU against the current Slurm allocation;
no cluster scheduler or node list is shipped.

Inspect the run's `protocol.json`, `resolved.json`, `runtime.json`, per-scene
results, agent sessions and raw submission feedback. Ledgers and Test plans
remain host-only. The repository ignores all run output.

If launching a command in the background, redirect stdin, stdout and stderr to
`/dev/null`, or pass `subprocess.DEVNULL` for all three streams in Python. Use the
run's existing status and feedback files for monitoring; do not create a separate
launcher log. Stop only processes belonging to your own run.
