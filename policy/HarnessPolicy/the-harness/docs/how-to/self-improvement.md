# Grow a library by self-improvement

A **library** is what the agent learns from its own development episodes: lessons for
the prompt, memories to recall, and code. `scripts/self_improve.py` grows one on
development seeds and keeps an entry only if it raises the score on other development
seeds; evaluation then uses the library frozen.

## The library

```text
library/
├── notes/*.md       lessons, added to the system prompt ("Lessons from earlier episodes")
├── memory/*.md      entries programs read with recall(query, k=5) (ranked by word overlap)
├── skills/*.py      functions run in the kernel before the first program; listed in the prompt
└── reactive/*.py    guards and controllers; loaded only when the reactive skill is on
```

Its **version** is the sha256 of every entry it uses (path and content). The agent uses a
library with

```toml
[runtime.library]
path = "../../library/r4"        # relative to this file
layers = ["notes", "skills", "reactive"]
```

The version goes into the lock, and the driver refuses to run an episode if
the library changed after the configuration was loaded: *evaluation needs a frozen
library*.

## The loop

```bash
python scripts/self_improve.py --experiment configs/experiments/dev.toml --deploy mine \
  --tasks stack_bowls,classify_objects --explore-seeds 0-3 --validate-seeds 10-13 \
  --library library/r4 --gpu 3 --output runs/self-improve
```

For each task (and `--rounds` times):

1. run the **explore** seeds with the current library;
2. for each explored episode, ask the main model (temperature 0.3) for one candidate:
   `<note|memory|skill|reactive name="...">...</...>` or `<none/>`. Notes have at most
   600 characters; code must parse and define exactly one public function with a
   docstring;
3. run the **validate** seeds with the candidate added, and keep it if the mean score
   rises by at least `--min-gain` (default 0.05) over the current library's.

Explore and validate seeds must not overlap; both are development layouts. Decisions
go to `OUTPUT/decisions.json`, the final library's manifest to `OUTPUT/library.json`.
Arguments: `--experiment` (the agent, its model, budgets), `--deploy`, `--set`, `--tasks`,
`--explore-seeds`, `--validate-seeds`, `--library`, `--min-gain`, `--rounds`, `--gpu`
and `--port` of the worker, `--dev-layout-dir`, `--output`.

!!! tip
    Validate on tasks where the agent is below its ceiling: a candidate cannot gain on
    seeds the agent already solves.
