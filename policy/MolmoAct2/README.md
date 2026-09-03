# MolmoAct2

**Contributor:** Haoquan Fang | **Paper:** MolmoAct2: Action Reasoning Models for Real-world Deployment | **arXiv:** [2605.02881](https://arxiv.org/abs/2605.02881) | **Original code:** [allenai/molmoact2](https://github.com/allenai/molmoact2)

This eval-only adapter runs [`hqfang/MolmoAct2-RoboDojo`](https://huggingface.co/hqfang/MolmoAct2-RoboDojo) directly through its public Transformers `predict_action` API. It supports dual ARX X5 RoboDojo observations, three RGB cameras, language instructions, and continuous absolute joint-pose actions. It does not clone or import LeRobot.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

Install the policy environment in `policy/MolmoAct2/.venv`:

```bash
cd XPolicyLab/policy/MolmoAct2
bash install.sh
source .venv/bin/activate
```

`MOLMOACT2_PYTHON` selects the Python version used by `uv` and defaults to `3.11`.

The model uses Hugging Face remote code. The default model and code revision are both pinned to `68964756dbfe5b455e6b4e4aa571199aa17d087c`; review that revision before running it in a trusted environment.

## Data Processing

Not included. This is an eval-only adapter and consumes runtime observations directly from XPolicyLab. `process_data.sh` is intentionally absent.

## Training

Not included. `train.sh` is intentionally absent because this adapter targets the converted public Transformers checkpoint rather than the former LeRobot training format. Use the [upstream MolmoAct2 repository](https://github.com/allenai/molmoact2) for training. An XPolicyLab-native training integration is not scheduled in this PR.

## Model Assets

No manual download is required. Passing the Hub repo id downloads the pinned snapshot into the normal Hugging Face cache:

```text
hqfang/MolmoAct2-RoboDojo
```

For an explicit immutable reference, pass:

```text
hf://hqfang/MolmoAct2-RoboDojo@68964756dbfe5b455e6b4e4aa571199aa17d087c
```

The full model page URL is also accepted. To prepare an offline local snapshot:

```bash
cd XPolicyLab/policy/MolmoAct2
.venv/bin/python download_checkpoint.py
```

Then use the resulting path as `ckpt_name`, for example `checkpoints/MolmoAct2-RoboDojo`. Local checkpoints resolve through the shared XPolicyLab checkpoint resolver and must contain `config.json`, `processor_config.json`, `norm_stats.json`, and either `model.safetensors` or `model.safetensors.index.json`.

## Evaluation

The policy side uses the adapter's `uv` environment. Only `env_cfg_type=arx_x5` with `action_type=joint` is supported by this checkpoint.

```bash
cd XPolicyLab/policy/MolmoAct2
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> uv <eval_env_conda_env>
```

Offline XPolicyLab wiring check:

```bash
EVAL_ENV_TYPE=debug bash eval.sh \
  RoboDojo stack_bowls \
  'hf://hqfang/MolmoAct2-RoboDojo@68964756dbfe5b455e6b4e4aa571199aa17d087c' \
  arx_x5 joint 0 0 0 uv base
```

Leave `EVAL_ENV_TYPE` unset, or set it to `sim`, for a RoboDojo simulator rollout.

## Configuration

Stable defaults live in `deploy.yml`:

| Key | Default | Meaning |
| --- | --- | --- |
| `hf_repo_id` | `hqfang/MolmoAct2-RoboDojo` | Only bare Hub repo id accepted without an `hf://` prefix. |
| `hf_revision` | pinned commit | Model, processor, and remote-code revision. |
| `norm_tag` | `robodojo` | Checkpoint normalization metadata tag. |
| `inference_action_mode` | `continuous` | This adapter supports continuous inference only. |
| `expected_action_representation` | `absolute` | Assertion against checkpoint metadata; it does not convert actions. |
| `num_steps` | `10` | Continuous flow-solver steps. |
| `actions_per_chunk` | `25` | Returned action horizon. |
| `dtype` | `bfloat16` | Inference dtype; `float32` is also supported but uses more memory. |
| `enable_cuda_graph` | `false` | Disabled by default for predictable variable-environment evaluation. |

Images remain RGB end to end and preserve camera order: high, left wrist, right wrist. The checkpoint returns de-normalized robot-scale absolute actions, so the adapter only validates and unpacks them into XPolicyLab action dictionaries.

Batch evaluation uses serial public-API calls with one deterministic generator per environment. Native model batching is intentionally deferred until it can match the public API action-for-action.
