# WSA

**Contributor:** WSA Team | **Paper:** [WSA: A 3D-Centric World-Spatial-Action Model for Generalizable Robot Control](https://openreview.net/pdf?id=bfphFLldXq) | **Original code:** https://github.com/zaleni/WSA | **Checkpoint:** [zaleni/wsa-b-clean2rand-delta-causal](https://huggingface.co/zaleni/wsa-b-clean2rand-delta-causal)

`WSA` integrates the WSA-Base RoboTwin 2.0 clean-to-random checkpoint with XPolicyLab. It supports the dual-arm `aloha_agilex` embodiment, joint-space delta actions, the three standard RGB cameras, action chunks, and WSA-Base image history. Integration files live at this directory level; the vendored upstream source is in `WSA/`. The adapter also recognizes WSA-Large checkpoint layouts, but the reproducible checkpoint and benchmark evidence in this submission are for WSA-Base.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official benchmark protocol: [RoboTwin 2.0 Leaderboard](https://robotwin-platform.github.io/leaderboard).

## Installation

Install RoboTwin and its assets by following the [RoboTwin repository](https://github.com/RoboTwin-Platform/RoboTwin), and prepare the policy environment by following the [WSA repository](https://github.com/zaleni/WSA). Then install the vendored WSA adapter and XPolicyLab dependencies:

```bash
conda activate wsa
cd XPolicyLab/policy/WSA
bash install.sh
```

`install.sh` installs the vendored WSA and XPolicyLab in editable mode, installs the websocket/runtime dependencies, and applies WSA's Qwen3-VL cached-inference patch. The policy and RoboTwin simulator may use separate environments or the same environment; use the corresponding environment name in the evaluation commands below.

The vendored tree does not embed WSA's external Git submodules. Released-checkpoint evaluation disables the DA3 teacher and does not need its source. For Base/Large training with the upstream nonzero 3D loss, clone [Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3) separately and export `DA3_CODE_ROOT=/path/to/Depth-Anything-3` before `train.sh`.

## Data Processing

WSA trains on LeRobot v3.0. `process_data.sh` calls XPolicyLab's official `scripts/transform_lerobot_v30_format.py`; its keys exactly match the official mapping (`observation.state`, `action`, and `observation.images.cam_high/cam_left_wrist/cam_right_wrist`). No custom image decoder is used. Because the official converter labels its potentially mixed-robot output as `unified_robot`, the wrapper retags this adapter's single supported `aloha_agilex` export as WSA's equivalent `aloha` metadata schema before computing normalization statistics; tensor and image fields are unchanged.

Place raw XPolicyLab trajectories under the parent workspace's `data/<bench>/<task>/<env_cfg>/data/` tree, then run:

```bash
cd XPolicyLab/policy/WSA
bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [max_episode]

# Example
WSA_ACTION_MODE=delta \
  bash process_data.sh RoboTwin cotrain aloha_agilex joint 200
```

The default output is `data/lerobot/<bench>-<ckpt>-<env>-<action>/`; stats are written to `data/norm_stats/aloha/<delta|abs>/stats.json`. `WSA_SOURCE_DATA_TYPE` defaults to `RoboDojo`, the shared XPolicyLab v1.0 trajectory reader used for both benchmark directory names. To use the published RoboTwin LeRobot-v3.0 export instead, set `WSA_ROBOTWIN_ROOT` during training and skip conversion.

## Training

The wrapper preserves the XPolicyLab checkpoint convention and launches upstream WSA fine-tuning. `action_type` remains `joint`; `WSA_ACTION_MODE` selects whether the policy predicts `delta` or `abs` joint actions.

```bash
cd XPolicyLab/policy/WSA
bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>

# WSA-Base example on GPUs 0-7
WSA_VARIANT=base WSA_ACTION_MODE=delta \
  bash train.sh RoboTwin cotrain aloha_agilex joint 0 0,1,2,3,4,5,6,7

# WSA-Large example
WSA_VARIANT=large WSA_ACTION_MODE=delta LOAD_TEXT_ENCODER=true \
  bash train.sh RoboTwin cotrain-large aloha_agilex joint 0 0,1,2,3,4,5,6,7
```

Checkpoints land under `checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>/`. Useful overrides are `WSA_ROBOTWIN_ROOT`, `WSA_STATS_ROOT`, `WSA_POLICY_INIT_PATH`, `WSA_PROC_PER_NODE`, and the upstream launch variables documented in `WSA/README.md`. The released training recipe used 8 NVIDIA H200 GPUs.

## Evaluation

The commands in this section evaluate the released WSA-Base checkpoint through XPolicyLab's policy server and RoboTwin's official scheduler. Run them from the RoboTwin root so that the simulator can resolve its `assets/` directory. XPolicyLab does not invoke upstream `WSA/evaluation/RoboTwin/eval_wsa_base.sh`; the adapter preserves the same WSA inference settings while using XPolicyLab's server/client protocol.

The scheduler invokes the standard same-machine adapter entry point below for each task; use the scheduler commands later in this section when selecting C2C/C2R and the episode count:

```bash
bash XPolicyLab/policy/WSA/eval.sh \
  <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_env> <eval_env>
```

### Download the checkpoint and model assets

The reported checkpoint is [`zaleni/wsa-b-clean2rand-delta-causal`](https://huggingface.co/zaleni/wsa-b-clean2rand-delta-causal). WSA-Base also requires a local Qwen3-VL model/processor and Cosmos tokenizer. Download all three before starting a benchmark job; do not rely on network downloads during evaluation.

```bash
export ROBOTWIN_ROOT=/absolute/path/to/RoboTwin
export WSA_ASSET_ROOT=/absolute/path/to/wsa-assets
mkdir -p "${WSA_ASSET_ROOT}"

cd "${ROBOTWIN_ROOT}/XPolicyLab/policy/WSA"
bash download_checkpoints.sh base \
  "${WSA_ASSET_ROOT}/wsa-b-clean2rand-delta-causal"

hf download Qwen/Qwen3-VL-2B-Instruct \
  --local-dir "${WSA_ASSET_ROOT}/Qwen3-VL-2B-Instruct"
hf download nvidia/Cosmos-Tokenizer-CI8x8 \
  --local-dir "${WSA_ASSET_ROOT}/Cosmos-Tokenizer-CI8x8"
```

Set the runtime paths in the shell that launches the scheduler:

```bash
export WSA_CHECKPOINT_PATH="${WSA_ASSET_ROOT}/wsa-b-clean2rand-delta-causal"
export QWEN3_VL_PRETRAINED_PATH="${WSA_ASSET_ROOT}/Qwen3-VL-2B-Instruct"
export QWEN3_VL_PROCESSOR_PATH="${WSA_ASSET_ROOT}/Qwen3-VL-2B-Instruct"
export COSMOS_TOKENIZER_PATH_OR_NAME="${WSA_ASSET_ROOT}/Cosmos-Tokenizer-CI8x8"

export WSA_ACTION_MODE=delta
export WSA_DTYPE=bfloat16
export WSA_LOAD_CHECKPOINT_ON_CPU=true
export EVAL_ENV_TYPE=sim

# Keep the official per-task settings from robotwin_eval_config.yaml.
unset WSA_INFER_HORIZON
unset WSA_BINARIZE_GRIPPER
unset WSA_ALLOW_DUMMY_POLICY
```

The local checkpoint directory must contain `config.json`, `stats.json`, and model safetensors. The policy environment must also provide the XPolicyLab websocket dependencies:

```bash
test -f "${WSA_CHECKPOINT_PATH}/config.json"
test -f "${WSA_CHECKPOINT_PATH}/stats.json"
find "${WSA_CHECKPOINT_PATH}" -maxdepth 1 -name '*.safetensors' | head

conda run -n wsa python -c \
  "import torch, yaml, msgpack_numpy, websockets; print('WSA evaluation dependencies OK')"
```

Released-checkpoint evaluation disables the DA3 teacher, so it does not require `DA3_CODE_ROOT` or a DA3 teacher checkpoint.

### Evaluation reliability and model memory

The WSA simulator launcher keeps RoboTwin's files unchanged. At runtime it wraps the official `scripts/eval_policy_xpolicylab.py` entry point and makes `task_env.test_num` -- the counter printed and saved by RoboTwin -- the authoritative episode budget. Consequently, an environment/rendering failure before the first policy action does not consume one of the requested 100 valid episodes. The known transient `RuntimeError: cannot create buffer` is retried three times by default before RoboTwin performs its normal environment cleanup and advances to another seed. Optional client-side controls are:

```bash
export WSA_RENDER_BUFFER_RETRIES=3
export WSA_RENDER_BUFFER_RETRY_DELAY_S=0.25
```

Five consecutive failures after those retries still abort the task, because that normally means the renderer is wedged rather than a single bad episode. Fix the simulator/Vulkan resource problem before resuming that job.

WSA-Base checkpoint restoration defaults to `WSA_LOAD_CHECKPOINT_ON_CPU=true`. The upstream LeRobot loader otherwise sends the complete safetensors state dict to `config.device`; with `cuda`, that temporarily coexists with the already constructed model and can make the loading peak or CUDA reserved memory look like two copies of the 3B checkpoint. The adapter now restores on CPU, transfers the finished policy to CUDA once in the requested dtype, releases the temporary cache, and prints:

```text
[WSA] CUDA memory after load: parameters+buffers=... GiB, allocated=... GiB, reserved=... GiB
```

Use this line, rather than checkpoint file size alone, to verify resident model memory. The first forward pass will additionally allocate activations and attention/cache workspaces. Set `WSA_LOAD_CHECKPOINT_ON_CPU=false` only when host RAM is constrained and the GPU has enough headroom for the temporary CUDA checkpoint copy.

### WSA evaluation settings

The adapter reads shared runtime defaults from `deploy.yml`, checkpoint-owned settings from the checkpoint's `config.json`, and per-task overrides from `WSA/configs/robotwin_eval_config.yaml`:

| Setting | Value/source |
| --- | --- |
| action interface | `joint`; WSA arm predictions use `delta` mode |
| model action chunk | 50 steps, from checkpoint `config.json` |
| executed steps before replanning | task-specific 16, 24, or 30 |
| gripper binarization | task-specific |
| image history | current frame and the frame 15 control steps earlier |
| cameras | head, left wrist, right wrist RGB |
| image size | 224 x 224 |
| inference dtype | `bfloat16` |
| flow/diffusion inference steps | 10, from checkpoint `config.json` |


### Single-task simulator check

The official multitask scheduler takes its task list from a YAML file. Create a one-task config such as `env_cfg/eval/wsa_single_task.yml` in the RoboTwin checkout:

```yaml
gpu_ids: [0]
jobs_per_gpu: 1
num_workers: 1
tasks:
  - adjust_bottle
```

Then run one clean episode. Replace the environment names if policy and simulator use different environments; using `wsa` for both is supported when that environment contains both installations.

```bash
cd "${ROBOTWIN_ROOT}"

bash scripts/eval_policy.sh multitask \
  --config env_cfg/eval/wsa_single_task.yml \
  --policy-name WSA \
  --ckpt-name wsa-b-clean2rand-delta-causal \
  --env-cfg-type aloha_agilex \
  --policy-conda-env wsa \
  --eval-env-conda-env wsa \
  --action-type joint \
  --task-config demo_clean \
  --test-num 1 \
  --seed 42 \
  --jobs-per-gpu 1 \
```


### Clean-to-clean (C2C)

Run all 50 tasks with 100 valid episodes per task:

```bash
cd "${ROBOTWIN_ROOT}"

bash scripts/eval_policy.sh multitask \
  --config env_cfg/eval/all_tasks.yml \
  --policy-name WSA \
  --ckpt-name wsa-b-clean2rand-delta-causal \
  --env-cfg-type aloha_agilex \
  --policy-conda-env wsa \
  --eval-env-conda-env wsa \
  --action-type joint \
  --task-config demo_clean \
  --test-num 100 \
  --seed 42 \
  --jobs-per-gpu 1 \
  --output-dir eval_result/wsa_c2c
```

### Clean-to-random (C2R)

Use the same checkpoint and WSA settings, changing only the RoboTwin scene configuration:

```bash
cd "${ROBOTWIN_ROOT}"

bash scripts/eval_policy.sh multitask \
  --config env_cfg/eval/all_tasks.yml \
  --policy-name WSA \
  --ckpt-name wsa-b-clean2rand-delta-causal \
  --env-cfg-type aloha_agilex \
  --policy-conda-env wsa \
  --eval-env-conda-env wsa \
  --action-type joint \
  --task-config demo_randomized \
  --test-num 100 \
  --seed 42 \
  --jobs-per-gpu 1 \
  --output-dir eval_result/wsa_c2r
```

`env_cfg/eval/all_tasks.yml` defines the GPU pool and the 50-task list. `--jobs-per-gpu 1` above makes the concurrency explicit; increase it only after verifying that two WSA policy servers plus two simulators fit on one GPU. Use `--dry-run` first if you want to validate task, GPU, and command expansion without launching evaluation.

### Results and logs

Each scheduler invocation creates a timestamped directory under the selected `--output-dir`:

```text
eval_result/wsa_c2c/<run_id>/
|-- summary.json
|-- jobs/
`-- logs/
```

`summary.json` reports whether all jobs completed; the success rates are printed in the per-task logs and saved by RoboTwin under each task's `_result.txt`. For the latest C2C run:

```bash
RUN_DIR=$(ls -1dt eval_result/wsa_c2c/* | head -1)

jq '{jobs_total, jobs_succeeded, jobs_failed, jobs_skipped}' \
  "${RUN_DIR}/summary.json"
grep -H -E 'Final (batch )?success rate:' "${RUN_DIR}"/logs/*.log | sort
```

All 50 jobs must finish, and each formal task log must report `/100`. Run the same commands under `eval_result/wsa_c2r/` for C2R. Scheduler job success means the process exited normally; it is not the policy success rate.

## RoboTwin 2.0 Results

The released WSA-Base checkpoint was trained on clean demonstrations and evaluated over the 50-task RoboTwin 2.0 clean-to-random protocol. C2C uses `demo_clean`, C2R uses `demo_randomized`, and the reported average is `(C2C + C2R) / 2`.

| Checkpoint | C2C | C2R | Average |
| --- | ---: | ---: | ---: |
| `zaleni/wsa-b-clean2rand-delta-causal` | 83.8 | 77.8 | 80.8 |

## Configuration

Important `deploy.yml` settings are `action_mode`, `infer_horizon`, `binarize_gripper`, `dtype`, `checkpoint_path`, `wsa_root`, and `wsa_large_stats_path`. The server launcher maps these environment variables to overrides: `WSA_CHECKPOINT_PATH`, `WSA_ACTION_MODE`, `WSA_DTYPE`, `WSA_INFER_HORIZON`, `WSA_BINARIZE_GRIPPER`, `WSA_LARGE_STATS_PATH`, the Qwen/Cosmos/DA3 variables above, and WSA-Large's `WAN_MODEL_ID`, `WAN_TOKENIZER_MODEL_ID`, `ACTION_DIT_PRETRAINED_PATH`, `FUTURE_3D_PRETRAINED_PATH`, and `WSA_LARGE_LOAD_TEXT_ENCODER`.

## Notes

- Supported combination: `bench_name=RoboTwin`, `env_cfg_type=aloha_agilex`, `action_type=joint`; released checkpoints use `action_mode=delta`.
- The adapter infers Base versus Large from `config.json`; no evaluation-side variant flag is required.
- `eval_batch` is disabled. `get_action_batch` remains contract-complete and evaluates requested environments sequentially because one WSA forward is already GPU-intensive.
- A standalone XPolicyLab checkout has no outer `env_cfg/`; `model.py` first uses `get_robot_action_dim_info()` and only falls back to `utils/robot/_robot_info.json` for the debug-only standalone case.
- `allow_dummy_policy` returns hold-position chunks and is only for protocol checks, never for benchmark results.
