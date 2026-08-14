# G05 RoboDojo Real Policy Adapter

**Contributor:** OpenGalaxea | **Training support:** evaluation only

This adapter serves the G0.5 FM-only ARX-X5 checkpoint for official RoboDojo
real-robot evaluation. The submission contains the policy server only; the
official evaluator owns camera acquisition and robot control.

Shared XPolicyLab conventions are documented in the [repository README](../../README.md).
Official results are published on the [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

Point the adapter to the checkpoint-compatible G0.5 inference checkout and
install the policy packages:

```bash
export G05_REAL_ROOT=/path/to/GalaxeaVLA_Private
export G05_REAL_TOKENIZER_ROOT="$G05_REAL_ROOT/third_party/galaxea_tokenizer"
export G05_REAL_DATASET_ROOT="$G05_REAL_ROOT/third_party/galaxea_dataset"
export G05_PYTHON=/path/to/python3.10
bash policy/G05_real/install.sh
```

No Python environment, model weights, training data, or simulator assets are
vendored in this adapter.

## Data Processing

Not included. The official real-robot client sends decoded RGB observations and
14-dimensional bilateral joint state through the standard XPolicyLab protocol.

## Training

Not included in this eval-only submission. Training release timing will be
coordinated separately with the XPolicyLab maintainers.

## Model Assets

```bash
modelscope download --model ZhyRobert/g05-robodojo-real \
  --include 'arx_x5/**' \
  --local_dir ./checkpoints/g05_robodojo_real
```

Keep the downloaded directory intact:

```text
arx_x5/
├── checkpoint.pt
├── checkpoint.pt.sha256
├── .hydra/config.yaml
├── dataset_stats.json
├── hf_processor/
└── action_tokenizer_hf/
```

## Evaluation

The supported configuration is `bench_name=RoboDojo`,
`env_cfg_type=arx_x5`, `action_type=joint`, continuous FM inference at 25 Hz.
The model predicts 32 actions and returns the first 16 actions per request.

Start the policy server through the standard XPolicyLab entrypoint:

```bash
export G05_CKPT_PATH=/path/to/arx_x5/checkpoint.pt
export G05_PYTHON=/path/to/python3.10
bash policy/G05_real/setup_eval_policy_server.sh \
  RoboDojo real_robot checkpoint arx_x5 joint 0 0 \
  "$G05_PYTHON" 6061 0.0.0.0
```

The official RoboDojo real-robot client connects to the WebSocket policy
server and is responsible for executing the returned actions at 25 Hz.

## Configuration

- `G05_CKPT_PATH`: path to `arx_x5/checkpoint.pt`.
- `G05_PYTHON`: optional Python executable; defaults to the packaged runtime.
- `G05_REAL_ROOT`: optional G0.5 runtime source override.
- `action_steps`: `16`.
- `frequency`: `25`.
- `action_source`: `fm`.
