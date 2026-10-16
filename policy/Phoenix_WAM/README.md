# Phoenix_WAM

本版本仅支持评测。

Phoenix_WAM is an evaluation-only policy supporting **ARX X5, Piper and Piper X** using the same code. `env_cfg_type` automatically selects the matching checkpoint, normalization statistics and soft prompt ID.

Each robot has its own posttraining step **40000** checkpoint. We provide the policy; **RoboDojo provides the real-robot client and hardware configuration**.

## Installation

Run on the GPU server from `XPolicyLab/policy/Phoenix_WAM` (Linux, CUDA/BF16):

```bash
bash install.sh phoenix-wam
conda activate phoenix-wam
bash download_checkpoint.sh checkpoints/phoenix-wam-step40000
```

The download script fetches all three policy checkpoints and the shared frozen-model dependencies from [ModelScope](https://www.modelscope.cn/models/ssuuuuu/phoenix-wam). Model configuration, tokenizer and all three robots' normalization statistics are included in this policy directory.

To download only one robot and its shared dependencies, add `--robot-type arx_x5`, `--robot-type piper` or `--robot-type piper_x` to the download command.

| `env_cfg_type` | Selected checkpoint | Soft prompt ID |
| --- | --- | --- |
| `arx_x5` | `phoenix-wam-arx-x5-step_040000.pt` | 2 |
| `piper` | `phoenix-wam-piper-step_040000.pt` | 4 |
| `piper_x` | `phoenix-wam-piper-x-step_040000.pt` | 15 |

Keep the published filenames inside `WAM_CHECKPOINT_PATH`; do not rename or interchange the files. Loading verifies the selected checkpoint's SHA256 and training metadata. A missing or mismatched checkpoint stops startup.

## Evaluation

Use the official RoboDojo workspace, including its external `env_cfg/` robot definitions, on the policy server. Prepare the checkpoint settings:

```bash
export WAM_CHECKPOINT_PATH="$PWD/checkpoints/phoenix-wam-step40000"
export WAM_EXPECTED_STEP=40000
export WAM_ALLOW_DUMMY_POLICY=false
```

Choose **one** server command below. Replace the example task with the official evaluation task. These examples use GPU `0`, port `9000` and conda environment `phoenix-wam`.

```bash
# ARX X5
WAM_AUDIT_DIR="$PWD/logs/arx_x5" bash setup_eval_policy_server.sh \
  RoboDojo cover_blocks phoenix-wam-step40000 arx_x5 joint 0 0 phoenix-wam 9000 0.0.0.0

# Piper
WAM_AUDIT_DIR="$PWD/logs/piper" bash setup_eval_policy_server.sh \
  RoboDojo fill_pen_holder phoenix-wam-step40000 piper joint 0 0 phoenix-wam 9000 0.0.0.0

# Piper X
WAM_AUDIT_DIR="$PWD/logs/piper_x" bash setup_eval_policy_server.sh \
  RoboDojo cap_pen phoenix-wam-step40000 piper_x joint 0 0 phoenix-wam 9000 0.0.0.0
```

RoboDojo's official client connects to the GPU server's reachable IP on port `9000`, using `policy_name=Phoenix_WAM`, the matching `env_cfg_type`, and `action_type=joint`. For the standard client wrapper, use `EVAL_ENV_TYPE=real` and the official hardware configuration. Shared deployment conventions are in the [XPolicyLab README](../../README.md#-deployment-flow).

The official client controls execution timing: ARX X5/Piper use a 30 Hz action timeline, Piper X uses 25 Hz. The policy returns 32 joint actions per call by default. Use the standard `deploy.py` loop, which acknowledges each executed action with an observation. Call `reset()` between episodes. If the official client uses the single-environment loop, set `eval_batch: false` in its copy of `deploy.yml`.

Camera inputs are decoded RGB. Each view is center-cropped to a square and resized to 256×256 to match the prepared training images; 256×256 inputs retain their geometry. `replan_steps: 32` executes all 32 predicted actions before replanning; `num_inference_steps: 10` controls denoising.

Optional offline GPU smoke test, from the **parent directory of `XPolicyLab`**:

```bash
CUDA_VISIBLE_DEVICES=0 python -m XPolicyLab.policy.Phoenix_WAM.smoke_joint96 \
  XPolicyLab/policy/Phoenix_WAM/checkpoints/phoenix-wam-step40000 \
  --expected-step 40000 --robot-type piper --rounds 3 --audit-dir logs/phoenix-smoke
```

Replace `--robot-type` with `arx_x5` or `piper_x` to check the other profiles. This uses synthetic inputs and does not measure hardware task success.
