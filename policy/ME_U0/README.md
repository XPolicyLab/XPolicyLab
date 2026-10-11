# ME-U0

**Contributor:** [anesthesia3690](https://github.com/anesthesia3690) | **Paper:** MachEmbodied-U0: Unified Understanding and Generation Model for Embodied Intelligence | **arXiv:** [2609.25627](https://arxiv.org/abs/2609.25627) | **Original code:** [MachEmbodied/ME-U0](https://github.com/MachEmbodied/ME-U0)

ME-U0 jointly predicts future video and robot actions. This policy directory contains the RoboDojo `arx_x5` joint-action adapter and the matching GPU post-training entry points. The vendored `me_u0/` directory contains the model, data pipeline, Lance/Qwen/Wan components, preprocessing, and action codec.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

Place this directory at `RoboDojo/XPolicyLab/policy/ME_U0` and install the policy environment:

```bash
cd /path/to/RoboDojo/XPolicyLab/policy/ME_U0
bash install.sh me_u0
```

The installer creates a Python 3.12 environment with PyTorch 2.8 and FlashAttention. The simulator uses its own RoboDojo environment. For an existing compatible policy environment, install `requirements.txt` there and retain its matching PyTorch / torchvision / FlashAttention builds.

## Data Processing

RoboDojo post-training consumes an existing LeRobot v3 video dataset with three RGB camera streams, `action`, and `observation.state`. `process_data.sh` validates the dataset and creates the policy-local data link; it does not convert RoboDojo simulator episodes.

```bash
export ME_U0_ROBODOJO_DATA_ROOT=/path/to/RoboDojo_lerobot_v30_video
bash process_data.sh RoboDojo all arx_x5 joint
```

The dataset uses the delta-joint H48 contract and the included q01/q99 statistics.

## Training

Install the policy dependencies and the additional GPU training dependencies in the same CUDA environment:

```bash
bash install.sh me_u0
python -m pip install -r requirements-training.txt
```

Set the Lance assets, pretraining checkpoint, and training output root:

```bash
export LEAP_MODEL_ROOT=/path/to/lance-assets
export ME_U0_PRETRAINED_PTH=/path/to/ME-U0-Pretrained/model.pt
export RUN_ROOT=/path/to/train_runs
```

Run the XPolicyLab-style GPU entry point. The first six arguments are the shared policy convention; the GPU list controls the local `torchrun` world size.

```bash
bash train.sh RoboDojo all arx_x5 joint 0 0,1,2,3
```

Additional arguments are forwarded as training configuration overrides. Use `--dry-run` to print the resolved launcher command without starting training.

## Model Assets

```bash
bash download_checkpoint.sh /path/to/ME-U0-RoboDojo
```

The checkpoint directory must contain:

```text
ME-U0-RoboDojo/
├── model.pt
├── config.yaml
├── stats/robodojo_sim_arx_x5_delta_joint_horizon48.json
└── assets/
    ├── Wan2.2_VAE.pth
    ├── Lance_3B_Video/       # llm_config.json and tokenizer files
    └── Qwen2.5-VL-ViT/config.json
```

The adapter reconstructs the architecture from these configuration files and strictly loads the complete trained state from `model.pt`. It does not load the original Lance or ViT weight files before loading the policy checkpoint.

## Evaluation

```bash
bash eval.sh RoboDojo stack_bowls /path/to/ME-U0-RoboDojo arx_x5 joint 0 0 1 me_u0 robodojo
```

The checkpoint argument can be an absolute directory or a name under this adapter's `checkpoints/`, resolved by XPolicyLab's shared checkpoint resolver. The policy environment argument also accepts an absolute Python executable path, such as `/path/to/policy-env/bin/python`. See the [common workflow](../../README.md#-common-workflow) for the standard argument order.

For XPolicyLab's simulator-free debug client:

```bash
EVAL_ENV_TYPE=debug bash eval.sh RoboDojo stack_bowls /path/to/ME-U0-RoboDojo arx_x5 joint 0 0 0 me_u0 base
```

Use a current XPolicyLab checkout for encoded-image debug requests; image decoding belongs to the shared policy server. The adapter receives RGB arrays and performs no RGB/BGR channel swap.

### Multiple machines

Run the following on every machine, using one shared output directory. Replace
`--node-rank 0` with each machine's rank, from 0 to 7. Each GPU pair runs one
independent official `eval.sh` invocation; tasks and seeds are assigned across
all 32 GPU pairs.

```bash
ME_U0_POLICY_ENV=me_u0 ME_U0_SIM_ENV=robodojo \
bash eval_distributed.sh \
  --checkpoint /path/to/ME-U0-RoboDojo \
  --output-root /path/to/evaluation/me_u0_robodojo \
  --task-suite all --seeds 0,1,2 --eval-num native \
  --num-nodes 8 --node-rank 0 \
  --server-gpus 0,1,2,3 --sim-gpus 0,1,2,3 \
  --num-inference-steps 12 --action-chunk-size 36
```

`--node-rank auto` claims a rank in the shared output directory. `--tasks` accepts
a comma-separated task list instead of `--task-suite`; `--dry-run` writes the
assignments without starting a server or simulator. Results retain RoboDojo's
format under `results/RoboDojo`; logs are under `nodes/node_N/lanes/lane_N/tasks`.
Re-running the same command skips task/seed invocations whose exit code is zero.

For an existing simulator setup script, set `ME_U0_SIM_SETUP` to its absolute
path. It is sourced only on the simulator side. `ME_U0_POLICY_ENV` also accepts
an absolute Python executable; `ME_U0_SIM_ENV` accepts a conda environment
prefix. Multi-machine evaluation does not use the ME-U0 repository's external
policy-server launcher or its client installer.

## Configuration

Model architecture and dataset metadata come from the checkpoint's `config.yaml`. Runtime defaults are in `deploy.yml`:

| Key | Default | Meaning |
| --- | --- | --- |
| `num_inference_steps` | `12` | Denoising steps. |
| `action_chunk_size` | `36` | Actions executed before replanning; use 1–48 for this H48 checkpoint. |
| `request_timeout_s` | `300` | WebSocket request timeout in seconds. |
| `ws_ping_interval_s` | `null` | WebSocket keepalive interval; disabled by default. |
| `ws_ping_timeout_s` | `null` | WebSocket keepalive timeout; disabled by default. |

`eval_batch: true` enables the benchmark's batch loop; the adapter evaluates active environments sequentially. `ME_U0_NUM_INFERENCE_STEPS` and `ME_U0_ACTION_CHUNK_SIZE` override the runtime defaults. `ME_U0_POLICY_PORT` selects a local server port; otherwise a free port is chosen.

## Observation and action contract

- Cameras: `cam_head`, `cam_left_wrist`, `cam_right_wrist`.
- State/action ordering at the simulator boundary: left arm, left gripper, right arm, right gripper. Dimensions come from XPolicyLab's robot metadata.
- Images: 224×224 head view for ViT, 384×320 three-camera pyramid for the video branch, with the same uint8 resize conversion as post-training.
- Model outputs: H48 chunk-start-relative joint predictions, converted to absolute simulator targets using source q01/q99 statistics and the current joint state. The gripper transform is inverted before returning native simulator commands.
- Prompt: task, control mode, and action representation; the instruction prefix precedes the ViT tokens.
