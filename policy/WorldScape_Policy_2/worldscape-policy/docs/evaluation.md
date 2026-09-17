# Evaluation

`wsp-eval` is the common entry point for offline HDF5 replay, LIBERO, and
guarded native AgileX evaluation. RoboTwin uses its native manager launcher.
Optional backend packages are imported only after their backend is selected.

## Install

```bash
pip install -e .                         # policy runtime and common CLI
pip install -e ".[robotwin2]"            # RoboTwin 2 adapter dependencies
pip install -e ".[libero]"               # LIBERO adapter dependencies
pip install -e ".[agilex]"               # HDF5/real-robot dependencies
```

LIBERO must be installed from its supported release. RoboTwin source is
vendored at `third_party/RoboTwin`, but its assets, cuRobo checkout, and
simulator-specific binary dependencies must be installed separately by
following that directory's README.

## RoboTwin 2

RoboTwin uses its native `script/eval_policy.py` loop. A manager dynamically
assigns task/phase jobs to long-lived GPU workers, each of which loads one WSP2
model and reuses it across jobs.

```bash
export ROBOTWIN2_EVAL_MODEL_PATH=/checkpoint_path
export ROBOTWIN_ROOT="$(pwd)/third_party/RoboTwin"
export ROBOTWIN_GPU_IDS='[0,1,2,3]'
WSP_MODE=auto ./recipes/eval/eval_robotwin2.sh
```

Use `EVALUATION.task_name=adjust_bottle` for a single task. Seed filtering,
expert checks, instruction generation, video capture, and success accounting
remain owned by RoboTwin.

## LIBERO

The LIBERO recipe uses 50 deterministic trials, a 600-step horizon, and 20 Hz
control. It constructs `OffScreenRenderEnv` once per trial from each task's
BDDL metadata.

```bash
export WORLDSCAPE_CHECKPOINT=/checkpoints/worldscape
export LIBERO_BDDL_FILE=/datasets/libero/libero_spatial/task_0.bddl
WSP_MODE=auto wsp-eval --config configs/eval/libero.yaml
# Equivalent release launcher (defaults to interactive):
./recipes/eval/eval_libero.sh
# Auto checkpoint:
WSP_MODE=auto ./recipes/eval/eval_libero.sh
```

Add one task entry per BDDL file and record `suite_name` and `task_index` in
metadata. Benchmark success is read from LIBERO's success method. Numeric
`subgoal_success`, `subgoal_completion`, `subgoal_progress`,
`subgoals_completed`, and `subgoals_total` values exposed by an environment or
step info are normalized into common metrics.

## HDF5 replay

```bash
export WORLDSCAPE_CHECKPOINT=/checkpoints/worldscape
export WORLDSCAPE_HDF5_EPISODE=/data/episode.hdf5
wsp-eval --config configs/eval/agilex.yaml
./recipes/common/eval_agilex.sh
```

Replay is read-only and reports no task success unless the source environment
provides it. CI generates a small HDF5 episode and replays it through the same
backend, so no repository fixture or external data download is needed.
The AgileX release recipe intentionally selects this recorded-HDF5 path; it
does not actuate a robot.

## AgileX

The native recipe uses the WSP-owned robot contract, defaults to dry-run, and
writes the common artifacts plus `action_previews.jsonl`:

```bash
wsp-eval --config configs/eval/agilex.yaml
```

Evaluation reads image range conversion, ordered state/action fields,
normalization statistics, relative-action semantics, per-horizon statistics,
and embodiment IDs from the checksum-validated `transform_bundle.json`.
Evaluation fails closed when that artifact is absent or invalid. Convert an
older checkpoint with `wsp-convert-checkpoint` before evaluation.

Live hardware requires `backend_config.transport: manifold` and the explicit
`--live-hardware` flag. The standalone wheel packages WorldScape-owned HDF5 and
Manifold transports. The Manifold deployment must additionally provide
`manifold_msg`; DreamZero evaluation modules remain source-only oracle code.

### Migrated real-robot recipes

All AgileX task launchers resolve the common `configs/eval/agilex.yaml` through
strict mode and visual-prompt profiles:

```bash
./recipes/eval/eval_agilex_fold_shirt_text.sh
WSP_GOAL_IMAGE=/data/goal.png ./recipes/eval/eval_agilex_build_block_goal.sh
./recipes/eval/eval_agilex_build_block_demo.sh
./recipes/eval/eval_agilex_shell_game_demo.sh
```

The Fold Shirt recipe selects `VISUAL_PROMPT=none` and never sends a visual
prompt. The goal recipe sends
exactly one head-camera frame, loaded from an explicit path by default; Python
callers may instead select `source: hdf5` or `source: upload`, and
`source: first_observation` is accepted only with the explicit
`goal_from_first_observation: true` opt-in. The demo recipe uniformly samples
exactly 50 frames. `ctx_head_only: true` produces `V=1`; setting it to false
produces high/left/right `V=3`.

For live demo upload, use `source: transport` with `poll_transport: true`.
Every completed Manifold upload starts a new policy session and clears visual,
WAM, and event memory before the 50-frame prompt is sent once. HDF5 context is
preloaded through the WSP-owned replay transport. All three recipes require the
checksum-validated `agilex` EEF transform fields in exact
high/left/right and left/right state/action order.

## Artifacts

Every backend writes the same versioned directory:

- `config.yaml` or `config.json`: resolved run recipe;
- `summary.json`: aggregate success, subgoal, latency, and per-task metrics;
- `episodes.jsonl`: trial records with suite/task metadata, seed, horizon,
control frequency, success, subgoal metrics, and step latency;
- `per_task.csv`: task success and mean subgoal completion;
- `videos/<episode-id>.mp4`: present when frame capture is enabled.

Recipes fix trial count and seed. Override them only when intentionally
creating a separately named evaluation protocol.