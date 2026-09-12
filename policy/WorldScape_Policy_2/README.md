# WorldScape_Policy_2

**Contributor:** WorldScape Team | **Paper:** WorldScape Policy 2.0: Empowering Steerable World Action Modeling with Reasoning-Augmented Memory | **arXiv:** [https://arxiv.org/abs/2607.18840](https://arxiv.org/abs/2607.18840) | **Original code:** [https://github.com/manifoldai-research/WorldScape-Policy](https://github.com/manifoldai-research/WorldScape-Policy) (included as `worldscape-policy/`; [project page](https://manifoldai-research.github.io/WorldScape-Policy/), [checkpoints](https://huggingface.co/manifoldai-research/WorldScape-Policy-2)).

`WorldScape_Policy_2` is the XPolicyLab adapter for WorldScape Policy 2.0; the
WorldScape source (training, data tooling, and inference runtime) is included
as `worldscape-policy/`. It supports `auto` (default) and `interactive` modes
(see [`worldscape-policy/README.md`](worldscape-policy/README.md) for details),
RoboTwin dual-arm absolute joint control, stateful WAM memory, and 24-step
action horizons.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). RoboTwin 2.0 results: [RoboTwin Leaderboard](https://robotwin-platform.github.io/leaderboard).

## Installation

Create the WorldScape policy-server environment (The validated environment uses Python 3.11, PyTorch 2.8.0, CUDA 12.9,
TorchVision 0.23.0, and DeepSpeed 0.18.9.):

```bash
conda create -n worldscape-policy python=3.11 -y
conda activate worldscape-policy
cd XPolicyLab/policy/WorldScape_Policy_2
bash install.sh
```

`install.sh` runs `pip install -e worldscape-policy[server]` followed by
`pip install -e ../..`. Set `WORLDSCAPE_POLICY_ROOT` only if you want to use
a WorldScape checkout other than `worldscape-policy/`. Keep
RoboTwin/SAPIEN in its separate evaluation environment.

## Model Assets

The official Hugging Face repository publishes:

- `wsp_2_pretrain`: pretrained WorldScape Policy 2.0 checkpoint.
- `wsp_2_posttrain_robotwin2_c2r`: RoboTwin 2.0 C2R post-trained checkpoint
used for RoboTwin evaluation.

Download the complete checkpoint directory and preserve its upstream layout;
the adapter reads the checkpoint metadata, policy weights, visual-memory
artifacts, and RoboTwin checkpoint transform through
`worldscape-policy/src/worldscape_policy/native_builder.py` and
`worldscape-policy/evals/robotwin2/checkpoint.py`. Place it under
`checkpoints/` so the shared XPolicyLab checkpoint naming applies:

```bash
cd XPolicyLab/policy/WorldScape_Policy_2
huggingface-cli download manifoldai-research/WorldScape-Policy-2 \
  --include "wsp_2_posttrain_robotwin2_c2r/*" \
  --local-dir checkpoints
```

The requested interaction mode, action horizon, and diffusion view layout must
be supported by the checkpoint.

## Data Processing

WorldScape Policy 2.0 trains on LeRobot v2 data in the upstream-native
WorldScape layout (not the official XPolicyLab LeRobot export). Follow the
[data preparation guide](worldscape-policy/tools/data/README.md) in
`worldscape-policy/` (or the
[original code](https://github.com/manifoldai-research/WorldScape-Policy))
to produce `DATA_ROOT` and `dataset_stats.json`.

`process_data.sh` does not rewrite episodes; it links an existing dataset to
`data/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>` and writes the
WorldScape native metadata with `--embodiment robotwin2`:

```bash
cd XPolicyLab/policy/WorldScape_Policy_2
LEROBOT_DATA_PATH=/path/to/robotwin2_lerobot \
bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [extra_args...]
```



## Training

Training runs from the code in `worldscape-policy/`; see its
[README](worldscape-policy/README.md#-robotwin-20-evaluation) and
[post-training guide](worldscape-policy/docs/posttraining.md) (or the
[original code](https://github.com/manifoldai-research/WorldScape-Policy)) for
the `[train]` extra, `DATA_ROOT` / `ZSCORE_STATS_PATH` /
`PRETRAINED_MODEL_PATH`, and recipe options.

```bash
cd XPolicyLab/policy/WorldScape_Policy_2
DATA_ROOT=<lerobot_root> ZSCORE_STATS_PATH=<dataset_stats.json> \
PRETRAINED_MODEL_PATH=<wsp_2_pretrain> \
bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id> [extra_args...]
```

Example:

```bash
bash train.sh RoboTwin wsp2_clean arx_x5 joint 0 0,1,2,3,4,5,6,7
```



## Evaluation

Run evaluation from a RoboTwin workspace in which this checkout is the
`XPolicyLab/` submodule. The standard local entrypoint takes the current
XPolicyLab 10-argument form:

```bash
cd XPolicyLab/policy/WorldScape_Policy_2
bash eval.sh \
  <bench_name> <task_name> <ckpt_name> \
  <env_cfg_type> joint <seed> \
  <policy_gpu_id> <env_gpu_id> \
  <worldscape_policy_env> <robotwin_eval_env>
```

Example, with the checkpoint downloaded to
`checkpoints/wsp_2_posttrain_robotwin2_c2r` as above:

```bash
bash eval.sh \
  RoboTwin adjust_bottle wsp_2_posttrain_robotwin2_c2r \
  arx_x5 joint 0 0 0 worldscape-policy robotwin
```

`<ckpt_name>` may also be an absolute checkpoint directory, or you can set
`WORLDSCAPE_CHECKPOINT=/path/to/checkpoint` to force one.

The wrapper starts a websocket policy server in its own process group, waits
for the listening port, runs the shared XPolicyLab environment client, and
cleans up the complete server process group on exit.

For split-machine evaluation, install the appropriate environment on each
machine and use a reachable address:

```bash
# Policy-server machine (checkpoint under checkpoints/ or via WORLDSCAPE_CHECKPOINT)
bash setup_eval_policy_server.sh \
  <bench_name> <task_name> <ckpt_name> \
  <env_cfg_type> joint <seed> <policy_gpu_id> \
  <worldscape_policy_env> <port> 0.0.0.0

# Environment-client machine
bash setup_eval_env_client.sh \
  <bench_name> <task_name> <ckpt_name> \
  <env_cfg_type> joint <seed> <env_gpu_id> \
  <robotwin_eval_env> \
  "ckpt_name=<ckpt_name>,action_type=joint" \
  <port> <policy_server_ip>
```

Open the chosen TCP port between the machines. Transport is websocket
(`protocol: ws`).

For a protocol/action-shape smoke test without loading WorldScape weights, set
`WORLDSCAPE_ALLOW_DUMMY_POLICY=true` and use the XPolicyLab debug environment.
Dummy mode returns zero actions and is not a model-quality evaluation.

## Configuration

The server wrapper consumes these environment variables:

- `WORLDSCAPE_POLICY_ROOT`: WorldScape source checkout. Defaults to
`worldscape-policy/` in this directory.
- `WORLDSCAPE_CHECKPOINT`: optional explicit checkpoint directory. It
overrides the `ckpt_name`-based resolution described in Model Assets.
- `WORLDSCAPE_DEVICE`: model device, default `cuda`.
- `WSP_MODE`: `auto` or `interactive`, default `auto`.
- `ROBOTWIN_ACTION_HORIZON`: action horizon, default `24`.
- `ROBOTWIN_REPLAN_STEPS`: actions executed from each prediction. It must be
in `[1, action_horizon]` and divisible by the observation interval.
- `ROBOTWIN_OBSERVATION_INTERVAL`: sampling interval for the rolling
nine-frame WAM history, `3` for horizon 24.
- `ROBOTWIN_MEMORY_RESET_CHUNKS`: reset visual/event memory after this many
committed chunks; `0` disables periodic resets.
- `ROBOTWIN_VLM_HISTORY_NUM_FRAMES`: VLM anchor-history length, default `4`.
- `ROBOTWIN_DIFFUSION_VIEW_LAYOUT`: optional compatibility override,
`mosaic_2x2` or `robotwin_concat`.
- `ROBOTWIN_VLM_COT_PROMPT` and `ROBOTWIN_T5_PROMPT_TEMPLATE`: optional
prompt-template overrides for `auto` and `interactive` modes.
- `WSP_VALIDATE_CHECKPOINT_ARTIFACTS`: enable full upstream checkpoint
validation, default `false`.
- `WSP_LOG_INFERENCE`: print per-inference timing, default `true`.
- `WORLDSCAPE_ALLOW_DUMMY_POLICY`: skip source/checkpoint loading and return
zeros for debug-only wiring checks, default `false`.

Supported replan steps for horizon `24` (observation interval `3`) are
`3, 6, 9, 12, 15, 18, 21, 24`.

The adapter receives three already-decoded HWC `uint8` RGB arrays:
`cam_head`, `cam_left_wrist`, and `cam_right_wrist`. It performs no
OpenCV/PIL decoding and no RGB/BGR channel swap. Robot state and action
conversion use XPolicyLab's shared helpers in the dual-arm `6+1+6+1` packing
order.

## Limitations

- `eval_batch` and both batch APIs are disabled. One policy-server instance
owns one mutable rollout/history state and serves one active environment.

