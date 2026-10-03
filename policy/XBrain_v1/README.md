# XBrain-v1

This is an evaluation-only RoboDojo/XPolicyLab adapter. Training code and
training data-processing code are intentionally not included. The adapter
loads the selected Piper X release checkpoint through an inference-only
runtime and returns action chunks through XPolicyLab. Model observations and
predictions remain in joint space. All three robots, including every PiperX
task, currently return joint actions by default.

Supported targets: `env_cfg_type=piper_x`, `piper`, and `arx_x5`, with
`action_type=joint`. The selected `env_cfg_type` chooses the matching
checkpoint, normalization statistics, and model embodiment.

The release package must provide a checkpoint download method, model config,
normalization files, tokenizer files, and inference-only dependencies. Do not
put private training paths, credentials, or training-only source code in this
directory.

For local development, the inference runtime can be supplied outside the
policy directory. The final PR must package or install only the source code
needed for inference, and must not rely on an author's local source tree.

## Release resources

The policy does not include model weights. Its checkpoint-matched inference
runtime is bundled under `policy/XBrain_v1/runtime`; do **not** substitute an
arbitrary upstream `giga-models` version. Install the policy and bundled
runtime with:

```bash
export XBRAIN_PYTHON=/path/to/conda/env/bin/python
bash policy/XBrain_v1/install.sh
```

Download and verify all three robot checkpoints and shared resources with:

```bash
export XBRAIN_RELEASE_REPO='wuchong617/robodojo_xbrain_real'
export XBRAIN_RELEASE_DIR=/path/to/xbrain-v1-release-0.1.0
bash policy/XBrain_v1/download_checkpoint.sh
source "$XBRAIN_RELEASE_DIR/xbrain_env.sh"
```

The downloaded release directory contains `piper_x/model_ema/`,
`piper/model_ema/`, `arx_x5/model_ema/`, `norm/`, `tokenizer/`, and
`fast_tokenizer/`. The generated `xbrain_env.sh` exports these resource
locations:

```bash
export XBRAIN_PIPERX_CHECKPOINT_PATH=/path/to/release/piper_x/model_ema
export XBRAIN_PIPERX_NORM_STATS_PATH=/path/to/release/norm/robodojo_piperx_0902.json
export XBRAIN_PIPER_CHECKPOINT_PATH=/path/to/release/piper/model_ema
export XBRAIN_PIPER_NORM_STATS_PATH=/path/to/release/norm/robodojo_piper_0903.json
export XBRAIN_ARX_X5_CHECKPOINT_PATH=/path/to/release/arx_x5/model_ema
export XBRAIN_ARX_X5_NORM_STATS_PATH=/path/to/release/norm/robodojo_arx5_0903.json
export XBRAIN_TOKENIZER_PATH=/path/to/release/tokenizer
export XBRAIN_FAST_TOKENIZER_PATH=/path/to/release/fast_tokenizer

# The policy server accepts either a Python executable, a conda environment
# path, or (when Conda is available) a conda environment name.
export XBRAIN_PYTHON=/path/to/conda/env/bin/python
# Optional functional smoke-test override; official evaluation should use CUDA.
# export XBRAIN_DEVICE=cuda
```

The ARX X5 checkpoint selected for this submission uses
`robodojo_arx5_0903.json` as its final normalization statistics. The policy
returns standard XPolicyLab action dictionaries directly. The checkpoint is
trained with delta targets for the twelve arm-joint dimensions and absolute
targets for the two grippers. The checkpoint-matched inference Pipeline
converts the joint predictions to absolute targets before `get_action()`
returns. The action dictionaries therefore contain:

```text
left_arm_joint_state:  absolute joint targets (6)
left_ee_joint_state:   absolute gripper value (1)
right_arm_joint_state: absolute joint targets (6)
right_ee_joint_state:  absolute gripper value (1)
```

The official executor must execute these values directly; it must not add the
current joint state or apply a second joint-delta conversion.

### Joint output by default; optional PiperX endpose actions

Keep the server/evaluation argument `action_type=joint` for all three robots.
It describes the checkpoint and input-state contract. The separate
`output_action_type: joint` setting disables FK and Z adjustment for every
task. Omitting this setting or using `auto` also selects joint output for all
three robots. The existing endpose path is retained for an explicit future
`output_action_type: ee` opt-in, supported only for PiperX. The `env_cfg_type` supplied
at server startup selects the robot and checkpoint; observations do not need
to include an extra robot identifier.

The model predicts 50 joint actions, and the adapter selects the first 20 and
applies the task's gripper thresholds and per-arm scales. Only when endpose is
explicitly enabled does it convert PiperX joint targets through the
bundled `piperx_fk.py`. This NumPy-only implementation and its adapter ship in
the policy directory and require no external SDK checkout or absolute code path.
Input joint angles are radians. Each pose is in its own arm's base frame, with
XYZ in meters, a WXYZ quaternion, and zero additional flange-to-tool offset.
The executor must consume that pose convention; the left and right poses are
not coordinates in a shared world frame.

For endpose output, each action dictionary contains:

```text
left_ee_pose:         [x, y, z, qw, qx, qy, qz]
left_ee_joint_state:  [processed_gripper]
right_ee_pose:        [x, y, z, qw, qx, qy, qz]
right_ee_joint_state: [processed_gripper]
```

RoboDojo's `take_action` distinguishes `joint` and `ee` by the action dictionary
keys. Each row contains one arm representation only; the observation still
supplies the joint fields needed by the checkpoint.

In the optional endpose path, after FK each arm's Z is adjusted independently
for each outgoing action:

```text
0.13 <= z <= 0.17: z_out = max(z - 0.02, 0.13)
otherwise:        z_out = z
```

Only Z is adjusted; X/Y, quaternion and the processed grippers are retained.
There is no stateful carry-over between rows or tasks.

When endpose is explicitly enabled, PiperX's joint-output exception remains
the `pack_objects_into_backpack` prompt:

```text
Place all the objects on the table into the backpack.
```

This task returns the first 20 joint actions with the same gripper processing,
skipping both FK and Z adjustment. Every request resolves its output type from
its own prompt, including batch items and task changes without restart. All
other PiperX prompts then use endpose. With the current default `joint` (or
`auto`) configuration, every PiperX task stays in joint mode. Piper and ARX X5
also retain their joint output.

Before returning the first 20 actions, the adapter matches the client-provided
`instruction` (or `task_instruction`) against the bundled
`gripper_thresholds.json`. Left gripper dimension 6 and right gripper dimension
13 use independent per-task thresholds. A value strictly below its threshold
is replaced with zero; a value equal to or above the threshold is multiplied by
that task's `left_scale` or `right_scale`. Each scale defaults to 1.3 when omitted
from the JSON, and a scale of 1.0 leaves eligible values unchanged. Thresholds
still apply even with scale 1.0. The comparison uses the original prediction,
and scaling occurs once before
returning the actions. This postprocessing does not clip the scaled values.
Prompt matching ignores case, repeated whitespace, and a trailing period. An
unknown prompt is not assigned a guessed task: its predicted gripper values are
preserved and the server logs a warning once.

Task-specific settings updated on 2026-09-29:

| Robot | Task | Left threshold | Right threshold | Left scale | Right scale |
|---|---|---:|---:|---:|---:|
| PiperX | `pack_objects_into_backpack` | 0.15 | 0.15 | 1.1 | 1.1 |
| PiperX | `classify_objects` | 0.25 | 0.25 | 1.3 | 1.3 |
| PiperX | `sweep_blocks` | 0.35 | 0.30 | 1.3 | 1.0 |
| Piper | `fill_pen_holder` | 0.35 | 0.35 | 1.1 | 1.1 |
| Piper | `put_objects_into_basket` | 0.33 | 0.33 | 1.15 | 1.15 |
| Piper | `insert_charger` | 0.20 | 0.20 | 1.1 | 1.1 |
| Piper | `stack_and_cover_blocks` | 0.45 | 0.45 | 1.1 | 1.1 |
| ARX X5 | `pack_and_pour_fruit` | 0.30 | 0.30 | 1.1 | 1.1 |

Other task thresholds and their default 1.3 scales remain unchanged. Edit
`left`/`right` for thresholds and `left_scale`/`right_scale` for multipliers in
`gripper_thresholds.json`.

The training dataset metadata records these control frequencies:

```text
piper_x: 25 Hz
piper:   30 Hz
arx_x5:  30 Hz
```

These values describe the dataset/control contract. The official hardware
client remains responsible for enforcing its supported control loop and should
confirm the final real-robot timing contract before execution.

Environment variables take precedence over `deploy.yml`. This keeps the
committed policy portable and prevents local absolute paths from entering a
submission.

## Local checks

```bash
cd XPolicyLab
bash -n policy/XBrain_v1/*.sh
python -m py_compile policy/XBrain_v1/model.py policy/XBrain_v1/deploy.py
```

The official installation should use the bundled inference runtime. The public
upstream `giga-models` package does not contain the checkpoint-matched changes
required by this release. `install_runtime.sh` remains only an optional local
helper for an existing conda-pack archive and is not part of the official
setup.

The standard XPolicyLab `eval.sh` arguments are documented in the repository
README. The checkpoint predicts 50 actions per inference call, but the adapter
returns only the first 20 actions to the client. After those actions execute,
the client collects a fresh observation and requests a new prediction. The
`action_horizon` setting may shorten this execution horizon, but cannot exceed
20.

## Evaluation-only submission

This policy is submitted for evaluation only. Training code, training data,
private training configurations, and the private model-development repository
are intentionally excluded. The PR contains the complete checkpoint-matched
inference runtime and the standard XPolicyLab adapter required for evaluation.

## Debug validation

The adapter has been exercised through the official WebSocket debug client for
all three supported targets:

```text
piper_x:  EVAL_ENV_TYPE=debug closed loop passed
piper:    EVAL_ENV_TYPE=debug closed loop passed
arx_x5:   EVAL_ENV_TYPE=debug closed loop passed
```

Each run completed server startup, WebSocket connection, `reset`, repeated
`update_obs`, repeated `get_action`, action validation, and client/server
cleanup. This debug result verifies the policy protocol and tensor contracts;
it is not a real-robot task score.
