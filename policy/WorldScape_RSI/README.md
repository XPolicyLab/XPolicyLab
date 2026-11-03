# WorldScape_RSI

[中文](README_zh.md) · [Website](https://r2si.worldscape.io) · [Model weights](https://huggingface.co/manifoldai-research/WorldScape-R2SI)

**Contributor:** WorldScape team  
**Supported:** RoboDojo · ARX X5 (`arx_x5`) · joint actions

WorldScape_RSI is a robot-control Agent that combines visual reasoning, generic manipulation Skills and WorldScape Flash/Memory policies. At each stage, the Agent chooses native tools or a learned policy using current observations and progress. It can take over at an acknowledged action boundary and discard the remaining policy actions.

This directory contains the Agent Harness, generic library, WAM implementation, RoboDojo adapter and training scripts. It runs independently of the outer RSI loop.

## Components

| Path | Role |
|---|---|
| `model.py`, `server.py` | Official Agent adapter and single-environment server |
| `harness/` | Agent, generic Skills, native tools, WAM dispatch and release |
| `wam_model.py`, `xpolicylab_adapter/` | WorldScape model integration |
| `worldscape-policy/` | Model source, data pipeline and training recipes |
| `deploy.yml`, `eval.sh` | Official evaluation configuration and entrypoint |
| `wam_deploy*.yml`, `serve_robodojo.sh` | Standalone WAM deployment |

The library contains **34 generic Skills, four tools and two primitives**. All tasks use the shared library. WAM receives the original environment instruction; its action history advances with execution acknowledgements.

## Installation

Use Linux, Python 3.11 and NVIDIA CUDA. Follow [INSTALLATION.md](INSTALLATION.md) to create the environment, then run:

```bash
bash install.sh
# Also install training dependencies when training models:
python -m pip install -e './worldscape-policy[train]'
```

Set the simulator assets and Agent provider:

```bash
export ROBODOJO_ROOT=/path/to/RoboDojo
export R2SI_API_BASE=https://your-provider.example/v1
export R2SI_API_KEY=your-api-key
```

The Agent configuration is `harness/configs/inner_agent.json`: `gpt-6-astra`, reasoning effort `high`, and full episode context. Install the official robot configuration in the parent workspace's `env_cfg/`.

## Model assets

```bash
bash download_checkpoints.sh /data/models/WorldScape-R2SI
export WRSI_WAM_MEMORY_CHECKPOINT=/data/models/WorldScape-R2SI/wsp_2_memory_posttrain_robodojo
export WRSI_WAM_NOMEMORY_CHECKPOINT=/data/models/WorldScape-R2SI/wsp_2_flash_posttrain_robodojo
export WRSI_WAM_MEMORY_DEVICE=cuda:1
export WRSI_WAM_NOMEM_DEVICE=cuda:0
python -B preflight.py --check-runtime --check-cuda --check-robot-assets --check-checkpoints
```

Reserve at least 60 GB for the weights. Device indices are relative to the server's `CUDA_VISIBLE_DEVICES`. Checkpoint file hashes are recorded with the Harness release.

## Data processing

RoboDojo training reads official HDF5 trajectories through the XPolicyLab image helpers. Preparation produces normalization statistics and a temporal index:

```bash
DATA_ROOT=/data/RoboDojo ROBODOJO_PREPARED_DIR=/data/rsi-prepared \
  ROBODOJO_MODEL=flash bash process_data.sh RoboDojo baseline arx_x5 joint
```

Use `ROBODOJO_MODEL=memory` for the Memory pipeline.

## Training

Download the Stage-1 initialization:

```bash
bash download_pretrained.sh /data/models/WorldScape-Policy-2
export PRETRAINED_MODEL_PATH=/data/models/WorldScape-Policy-2/wsp_2_pretrain
```

Set `DATA_ROOT`, `ROBODOJO_INDEX_PATH` and `ROBODOJO_ZSCORE_STATS` to the prepared data, then train:

```bash
ROBODOJO_MODEL=flash OUTPUT_DIR=/data/rsi-training/flash \
  bash train.sh RoboDojo baseline arx_x5 joint 62 0,1,2,3
```

For Memory, use `ROBODOJO_MODEL=memory` for both preparation and training. Additional arguments are passed to the training recipe as configuration overrides.

[Native training](worldscape-policy/docs/native_training.md) · [Data hot loading](worldscape-policy/docs/robodojo_hotload.md)

## Evaluation

After binding the weights and provider, use the standard ten arguments:

```bash
bash eval.sh RoboDojo stack_bowls public arx_x5 joint 0 0,1 0 r2si base
```

Arguments: benchmark, task, checkpoint label, robot configuration, action type, seed, policy GPUs, simulator GPU, policy conda environment, simulator conda environment.

For split-machine deployment, use `setup_eval_policy_server.sh` and `setup_eval_env_client.sh`. Each Agent server handles one environment (`eval_batch: false`). Official evaluation loads `model.py`; `serve_robodojo.sh` provides the separate direct-WAM interface.

For protocol debug, use `EVAL_ENV_TYPE=debug` with the same evaluation command. Repeat with `DEBUG_OBS_ENCODED=1` to exercise encoded observations.

## Configuration

| Setting | Purpose |
|---|---|
| `deploy.yml` | Agent server and standard XPolicyLab fields |
| `release_path`, `release_sha256` | Select and optionally pin the Harness release |
| `harness/configs/inner_agent.json` | Agent model and provider environment keys |
| `harness/configs/wam_skills.json` | Flash/Memory capabilities and checkpoint bindings |
| `R2SI_SESSION_RELEASE_PATH` | Optional provider session-release endpoint |
| `WRSI_POLICY_CLIENT_CONFIG` | Optional remote WAM client configuration |
| `WRSI_INTEGRITY_MODE=full` | Full content verification between actions |

Both WAMs use the published rolling setup: 24 actions per chunk, observation interval 3 and eight history segments. Flash's checkpoint capacity is one chunk, so it receives the latest visual frame; Memory has a 24-chunk capacity. Completing a chunk preserves continuity. Agent takeover or switching policies clears the unused prediction and old context.

## Tests

From the XPolicyLab parent workspace:

```bash
export PYTHONPATH="$PWD:$PWD/XPolicyLab"
export R2SI_TEST_XPL_ROOT="$PWD/XPolicyLab"
python -m pip install -r XPolicyLab/policy/WorldScape_RSI/requirements-test.txt
python -B -m pytest -p no:cacheprovider -q XPolicyLab/policy/WorldScape_RSI/harness/tests
```

See [Harness internals](harness/README.md), [component notices](harness/THIRD_PARTY.md) and `VALIDATION.json` for versioned test records.

