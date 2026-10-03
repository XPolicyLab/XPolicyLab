# EchoPolicy

**Contributor:** EchoPolicy contributors | **Paper:** Not released | **arXiv:** Not available | **Original code:** Inference runtime included in `echopolicy/`

Evaluation-only RoboDojo adapter combining the official fine-tuned Pi05 policy with VLM subgoal planning, progress assessment, and candidate selection. Supports `bench_name=RoboDojo`, `env_cfg_type=arx_x5`, and `action_type=joint`.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

Use the existing Pi_05 adapter's pinned openpi environment (Python 3.11 and a compatible CUDA/JAX installation). Install `uv` first, then:

```bash
cd XPolicyLab/policy/EchoPolicy
bash install.sh
```

This runs `../Pi_05/install.sh`, then installs the included orchestration package in editable mode in the same environment. Keep `echopolicy/subtask_template/` with the source: the supported installation is editable. The standard server wrapper accepts `uv` (the default Pi_05 environment), an absolute virtual-environment path, a project containing `.venv/`, or a conda environment name.

## Data Processing

Not included; this is an eval-only adapter using the official RoboDojo Pi05 checkpoint. It does not introduce a training dataset or a conversion pipeline.

## Training

Not included. Training release ETA: not scheduled. See the [Pi_05 adapter](../Pi_05/README.md) for the underlying policy's training recipe.

## Model Assets

Use the **official RoboDojo fine-tuned Pi05 weights**, distributed in the [RoboDojo checkpoint collection on Hugging Face](https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo/tree/main/ckpt/RoboDojo/Pi_05) or its [ModelScope mirror](https://modelscope.cn/datasets/RoboDojo-Benchmark/RoboDojo). The [official checkpoint downloader](https://github.com/RoboDojo-Benchmark/RoboDojo/blob/main/scripts/RoboDojo/download_ckpt.sh) resolves the Pi05 directory:

```bash
# From a RoboDojo workspace, with this XPolicyLab checkout inside it:
bash scripts/RoboDojo/download_ckpt.sh huggingface Pi_05
# Or: bash scripts/RoboDojo/download_ckpt.sh modelscope Pi_05
```

Pass the downloaded checkpoint directory as an absolute `ckpt_name` path. The evaluated checkpoint was step `59999`, containing `params/` and `assets/arx_x5_sim/norm_stats.json`. No weights or credentials are redistributed here. An explicit `checkpoint_path` takes precedence; otherwise XPolicyLab's shared resolver also supports `checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>/` under this adapter.

## Evaluation

The VLM provider must expose the Gemini `generateContent` API and the configured model. Set credentials in your shell or secret manager; do not put them in code or command arguments:

```bash
read -rsp 'VLM API key: ' VLM_API_KEY; echo
export VLM_API_KEY
export VLM_MODEL=gemini-3.8-flash
export VLM_THINKING_LEVEL=medium
# Set VLM_BASE_URL to your compatible provider's API base if needed.
# Default API base: https://generativelanguage.googleapis.com/v1beta
```

Model availability depends on your provider; the default model name is configurable and is not a claim of availability at the default API base.

```bash
cd XPolicyLab/policy/EchoPolicy
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_env_or_uv_path> <eval_env_conda_env>

# One policy and the standard simulator client:
bash eval.sh RoboDojo cover_blocks /path/to/checkpoint/59999 arx_x5 joint 0 0 0 uv RoboDojo

# Official debug client, no simulator (still loads the real policy and calls the VLM):
EVAL_ENV_TYPE=debug bash eval.sh RoboDojo cover_blocks /path/to/checkpoint/59999 arx_x5 joint 0 0 0 uv base
DEBUG_OBS_ENCODED=1 EVAL_ENV_TYPE=debug bash eval.sh RoboDojo cover_blocks /path/to/checkpoint/59999 arx_x5 joint 0 0 0 uv base
```

`POLICY_PORT` defaults to `17001` for `eval.sh`. For split-machine evaluation, use the standard `setup_eval_policy_server.sh` and `setup_eval_env_client.sh` positional interfaces described in the shared README. Each frontend accepts batches of 10 environments; VLM requests for those environments run concurrently. The orchestration runtime imposes no VLM concurrency semaphore or output-token cap; provider-side quotas still apply.

### Eight policy instances

```bash
# Preview GPU and port assignments without loading a model or needing a key:
bash serve_fleet.sh /path/to/checkpoint/59999 uv --dry-run

# Default: GPUs 0–7, public ports 17001–17008, private backend ports 27001–27008.
bash serve_fleet.sh /path/to/checkpoint/59999 uv

# An explicit alternative GPU/port assignment:
GPU_IDS=0,1 BASE_PORT=17001 BACKEND_BASE_PORT=27001 \
  bash serve_fleet.sh /path/to/checkpoint/59999 uv
```

The fleet runs in the foreground and cleans up its children on exit or when an instance stops. Every GPU hosts one independent Pi05 backend and frontend. Pi05 binds to loopback; frontends bind to `0.0.0.0` unless `ORCHESTRATOR_HOST` is set. Allow the chosen frontend ports through your deployment firewall. The protocol has no built-in authentication; expose it only in your intended evaluation network.

## Configuration

- `batch_size: 160`: fixed JAX inference batch, padded when necessary; larger requests split into fixed-size chunks. Default 16 candidates per environment, so 10 environments fill one batch.
- JAX preallocation is enabled with memory fraction `0.9` in `serve_one.sh`.
- `train_config_name: pi05_base_aloha_full_sim_arx-x5_seed_0`, `repo_id: arx_x5_sim`, `checkpoint_num: 59999`: match the official evaluated checkpoint.
- `checkpoint_path: null`: optional explicit checkpoint directory.
- `warmup: true`: compile one full batch before serving.
- `ECHOPOLICY_RUNTIME_ROOT`: cache, temporary-file and log root; defaults to `runtime/` under this adapter. Keep runtime artifacts out of version control.
- `VLM_MODEL` (default `gemini-3.8-flash`), `VLM_THINKING_LEVEL` (default `medium`), `VLM_BASE_URL`, `VLM_API_KEY`: provider configuration. API keys stay in the process environment.
- `BENCH_NAME`, `TASK_NAME`, `ENV_CFG_TYPE`, `ACTION_TYPE`, `POLICY_SEED`: fleet overrides; defaults are RoboDojo, cover_blocks, arx_x5, joint, and 0.

## Validation and limitations

The default `gemini-3.8-flash` / `medium` configuration evaluated `cover_blocks` episodes 0–9 with seed 0 and 10 parallel environments: **9/10 successful (90%), mean score 93.0/100**, with no unstable episodes. Episodes 1–9 succeeded; episode 0 failed with score 30/100. This used one freshly started policy instance on GPU 0 / port 17001, the official Pi05 checkpoint, fixed batch 160, and memory fraction 0.9.

For comparison, one run each with `gemini-3.8-flash` / `low` and `gemini-3.7-flash` / `low` scored 7/10 (71.5 points) and 6/10 (68.0 points), respectively. These are self-run ten-episode checks with sampling variability, not an official leaderboard result or an eight-instance throughput test.

Only the ARX X5 joint-action contract is supported. Pi05 inference uses its existing XPolicyLab transforms and normalization assets; the batched adapter performs one padded JAX call per candidate batch. CFG branches must use identical prompts. The runtime supports encoded observations through XPolicyLab's shared image decoder at the frontend boundary; `model.py` receives decoded RGB arrays.

PR packaging validation: 113 runtime tests and 4 CPU/JAX adapter contract tests passed (9 existing optional tests skipped). The official debug client completed both raw and encoded RGB loops with 10 environments, using the exported frontend, a real VLM, and an already loaded Pi05 backend. Fresh environment installation and eight-GPU throughput were not rerun. Real-checkpoint startup through the adapter launcher and complete simulator evaluation were validated on one GPU.
