# InternVLA_A1_5

**Contributor:** RoboDojo Team | **Paper:** InternVLA-A1.5 technical report | **arXiv:** TBD | **Original code:** See vendored `internvla_a1_5/`.

`InternVLA_A1_5` is the XPolicyLab/RoboDojo inference adapter for the InternVLA-A1.5 policy. It wraps `InternVLAA15Policy` (flow-matching action expert + Qwen3.5 VLM) into the `ModelTemplate` interface so it can be served by the XPolicyLab policy server and evaluated against RoboDojo simulation tasks; the InternVLA-A1.5/LeRobot runtime source is vendored in `internvla_a1_5/`, matching the self-contained layout of `InternVLA_A1`.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

The adapter no longer depends on a separate local `lerobot_lab` checkout. Upstream details are in `internvla_a1_5/tutorials/installation.md`; for development, `LEROBOT_SRC_PATH` can override the included source directory.

```bash
cd XPolicyLab/policy/InternVLA_A1_5
bash install.sh
conda activate internvla_a1_5
```

## Data Processing

No `process_data.sh` — the adapter only covers inference and does not include a data pipeline; prepare datasets with the upstream InternVLA-A1.5 recipe.

## Training

No `train.sh` — training scripts are not included; use the upstream InternVLA-A1.5 training recipe to produce checkpoints, then point `INTERNVLA_CKPT_PATH` (or `ckpt_name`, resolved under `checkpoints/` with `checkpoint_num` selecting the step) at the exported checkpoint for evaluation.

## Evaluation

```bash
cd XPolicyLab/policy/InternVLA_A1_5
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_conda_env> <eval_env_conda_env>

# Example: evaluate an exported local checkpoint on stack_bowls
INTERNVLA_CKPT_PATH=/path/to/pretrained_model \
  bash eval.sh RoboDojo stack_bowls internvla_a1_5 arx_x5 joint 0 0 0 <policy_conda_env> <eval_env_conda_env>

# Example: use the Hugging Face repo instead of a local export
INTERNVLA_CKPT_PATH=hxma/internvla_a15_robodojo_60k \
  bash eval.sh RoboDojo stack_bowls internvla_a1_5 arx_x5 joint 0 0 0 <policy_conda_env> <eval_env_conda_env>
```

`EVAL_ENV_TYPE=debug` runs the offline wiring check (no simulator); leave it unset or set `EVAL_ENV_TYPE=sim` for RoboDojo simulation. For split-machine deployment via `setup_eval_policy_server.sh` / `setup_eval_env_client.sh`, follow the [Deployment Flow](../../README.md#-deployment-flow).

## Configuration

Policy-specific `deploy.yml` keys worth checking before evaluation:

| Key | Notes |
|---|---|
| `stats_key` | Robot type used for both `get_schema(stats_key)` and `stats.json` lookup. Must match training (e.g. `aloha`). |
| `inference_backend` | `standard` or `optimized` (optimized requires `action_loss_only=True`). |
| `infer_horizon` | Number of action steps executed per inference call (default 20). |
| `action_mode` | `delta` (add to current state) or `abs` (absolute). |
| `tokenize_state` | Whether the chat processor tokenizes the state into the prompt. |
| `max_state_dim` / `max_action_dim` | Padding dims (default 32, matches A1.5 config). |
| `resize_size` | Image resize target (default 224). |
| `checkpoint_num` | Step number used to select a checkpoint under the run directory. |

Environment variables used by the adapter:

| Variable | Notes |
|---|---|
| `HF_HOME` | Local HuggingFace cache; needed for offline Qwen3.5 processor/VLM weights (`config.vlm_model_name_or_path`). |
| `INTERNVLA_CKPT_PATH` | Exported checkpoint directory or Hugging Face model repo id. |
| `LEROBOT_SRC_PATH` | Optional development override for the included `internvla_a1_5/src`. |
| `CUDA_VISIBLE_DEVICES` | GPU selection, set by the eval scripts. |

## Notes

- Differences from `InternVLA_A1`:

| Aspect | A1 | A1.5 |
|---|---|---|
| Image input | History frame pairs (current + past interval) | Single-frame 3 views (cam_high / cam_left_wrist / cam_right_wrist) |
| Chat processor | `Qwen3_VLProcessorTransformFn` | `InternVLAA15ChatProcessorTransformFn` (eval mode) |
| State/Action | Native dim, no pad/reorder | Padded to 32 dims + reordered via `get_schema(stats_key)` |
| Action recovery | Direct unnormalize | `invert_action_reorder` (uses `schema.action_reorder`) then unnormalize |
| Env vars | `COSMOS_PATH` / `QWEN3_2B_PATH` | `HF_HOME` for offline Qwen3.5 weights (`config.vlm_model_name_or_path`) |

- The inference flow mirrors `evaluation/RoboTwin/inference.py`: build a single-frame sample, run `ResizeImagesWithPadFn -> RemapImageKeyTransformFn -> NormalizeTransformFn -> InternVLAA15ChatProcessorTransformFn -> PadStateAndActionTransformFn -> ReorderStateActionTransform`, call `predict_action_chunk`, then invert the reorder back to the robot's native action layout, unnormalize, and (in delta mode) add the current state.
- The checkpoint's `config.json` must have `"type": "internvla_a1_5"` so `PreTrainedConfig.from_pretrained` resolves to `InternVLAA15Config`.
- The uploaded Hugging Face model repo must contain `model.safetensors`, `config.json`, and `stats.json` at its root.
- `stats_key` is shared between the dataset schema registry (`get_schema`) and the checkpoint's `stats.json`; they must refer to the same robot layout.
- The adapter supports `action_type='joint'` only, matching the A1 adapter.
