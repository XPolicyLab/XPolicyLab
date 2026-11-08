# Run on RoboTwin 2.0

RoboTwin 2.0 (SAPIEN) is the second bench: 50 dual-arm tasks, each served by its own
worker (`benches/robotwin/runtime.py`) with the same robot API as RoboDojo's.

## Setup

1. Check out RoboTwin at the revision in `third_party/sources.toml` (to
   `third_party/RoboTwin`, or set `HARNESS_ROBOTWIN_ROOT`) with its assets (the Hugging
   Face dataset `TianxingChen/RoboTwin2.0`; sha256 values in `sources.toml`).
2. `bash environment/setup.sh --only robotwin`: SAPIEN 3.0.1, mplib, toppra in the
   project environment, RoboTwin's two source patches, and RoboTwin's own cuRobo 0.7.8,
   built in place in `third_party/curobo-0.7.8` (`environment/robotwin_curobo.sh`;
   `HARNESS_ROBOTWIN_CUROBO` points elsewhere). Workers put it first on `PYTHONPATH`.

## Seeds

A RoboTwin seed is valid only if RoboTwin's expert solves it. Episode index *n* is the
*n*-th valid seed: evaluation seeds from 100000 × (1 + group) (the official ones),
development seeds from 10,000,000 + 100000 × group. Valid seeds are cached in
`seed_cache_dir/<split>-<group>/<task_config>/<task>.json` (`seed_cache_dir` defaults to
`<RoboTwin>/harness_seeds`). In a run file,
`layouts = "procedural"` selects the development split and `layout_pack_seed` the group.

## Run

```toml
--8<-- "configs/experiments/robotwin-dev.toml"
```

```bash
harness run configs/experiments/robotwin-dev.toml --deploy mine --output runs/robotwin-dev
harness run configs/experiments/robotwin-dev.toml --deploy mine --tasks all --seeds 0-9 --output runs/robotwin-all
```

or by hand, against one worker:

```bash
harness serve --factory benches.robotwin.runtime:create_environment \
  --options configs/workers/robotwin_runtime_dev.toml --port 8772
harness program configs/experiments/robotwin-dev.toml
```

## How it differs from RoboDojo

| | RoboDojo | RoboTwin |
|---|---|---|
| Control rate | 25 Hz | 10 Hz |
| Fingertip offset | 0.150 m | 0.12 m |
| Arms back to the start pose after `finish` | yes (the tasks require it) | no |
| Learned skills | Simate (the agent's), G0.5, other XPolicyLab policies; OpenWAM for experiments | none served |
| Manipulation tools | by object id (`locate`, `pick`, `release`, ...) | the manipulation skill's `pick` and `place` by name |
| Prompt pack | `prompts/robodojo` | `prompts/robotwin` (own `core.md`, other sections shared) |
| Self-improvement script | yes | no (it starts RoboDojo workers) |

The experiment runs the agent with the RoboTwin prompt pack (`[overrides.agent]`):
learned layer off (so no strategy step), the manipulation skill in place of the tools by
object id (those are tuned to RoboDojo's ARX X5), SAM 3, no task descriptions, the
agent's 30 model calls.
