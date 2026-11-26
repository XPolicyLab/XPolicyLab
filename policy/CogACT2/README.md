# CogACT2

**Contributor:** MSRA | **Paper:** To Be Released | **arXiv:** To Be Released | **Original code:** To Be Released

This is an evaluation-only adapter for the CogACT2 Preview model.
It includes standalone network and preprocessing implementations and runs directly
in the XPolicyLab policy process.
It supports `bench_name=RoboDojo`, `env_cfg_type=arx_x5`, and `action_type=ee`.

## Installation
```bash
# Template: install into an existing policy environment
conda activate <policy_conda_env>
bash XPolicyLab/policy/CogACT2/install.sh

# Example: create and install a dedicated policy environment
conda create -n cogact2_policy python=3.12 -y
conda activate cogact2_policy
bash XPolicyLab/policy/CogACT2/install.sh
```

The install script installs XPolicyLab in editable mode and the dependencies in
[requirements.txt](requirements.txt), then attempts to install the upstream
acceleration packages independently: `flash-attn==2.8.3.post1`,
`flash-attn-4[cu13]==4.0.0b32`, `flash-linear-attention==0.5.2`, and
`causal-conv1d==1.7.0`.

## Data Processing

Currently unsupported.

## Training

Currently unsupported.

## Evaluation

Run from the RoboDojo workspace. Download the checkpoint as described under
[Model Assets](#model-assets), then pass its absolute directory as `ckpt_name`.
The following example uses the downloader's default destination and assumes
`robodojo_sim` is an existing RoboDojo simulator conda environment.

```bash
# Template
bash XPolicyLab/policy/CogACT2/eval.sh \
  <bench_name> <task_name> <checkpoint_directory> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_conda_env> <eval_env_conda_env>

# Example
EVAL_ENV_TYPE=sim bash XPolicyLab/policy/CogACT2/eval.sh \
  RoboDojo stack_bowls "$PWD/XPolicyLab/policy/CogACT2/checkpoints/CogACT-v2-RoboDojo" \
  arx_x5 ee 0 0 0 cogact2_policy robodojo_sim
```

The adapter consumes the instruction, three RGB cameras, camera calibration, and
measured end-effector/gripper state. It predicts absolute camera-space 6D poses
and converts them to RoboDojo's environment-local poses with scalar-first
quaternions and continuous gripper commands.

The deploy loop enables the existing sensors' intrinsic/extrinsic observations
and supplies `env_origin`, `episode_identity`, and `control_step` from RoboDojo's
scene origin, task name, evaluation seed, native layout ID, and action counter.


## Model Assets

Download the released checkpoint from
[`microsoft/CogACT-v2-RoboDojo`](https://huggingface.co/microsoft/CogACT-v2-RoboDojo):

```bash
conda activate cogact2_policy
bash XPolicyLab/policy/CogACT2/download_checkpoint.sh

# Optional: choose a destination and pin a published commit
bash XPolicyLab/policy/CogACT2/download_checkpoint.sh \
  --output-dir /path/to/CogACT-v2-RoboDojo --revision <full_commit_hash>
```

The default destination is `policy/CogACT2/checkpoints/CogACT-v2-RoboDojo/`
inside XPolicyLab, independent of the working directory. A relative
`--output-dir` is relative to the working directory. The script downloads only
the four required files below (approximately 17 GB in this release) and checks
their presence and sizes. Repeat the same command to reuse completed downloads
and resume interrupted transfers through Hugging Face's download metadata.

The Python entry point is also available as `python download_checkpoint.py`
from this directory.

The checkpoint directory contains:

```text
checkpoints/CogACT-v2-RoboDojo/
  weights.pt                 # default: complete raw state dict
  config.yaml
  action_feature.json
  norm_config.json
```

## Configuration

Policy-specific options in [deploy.yml](deploy.yml):

| Key | Default | Meaning |
| --- | --- | --- |
| `num_inference_steps` | `10` | Number of uniform Euler sampling steps; must be positive. |
| `execute_steps` | `20` | Actions executed per chunk; must fit within the predicted robot-action horizon. |
| `auto_cast_bf16` | `true` | Enable BF16 autocast with conversion caching disabled. Parameters and floating-point buffers remain FP32; disabling autocast selects FP32 computation and SDPA. |
| `attn_backend` | `fa4` | VLM and MoT attention preference: `fa4` / `auto` tries FA4 → FA2 → SDPA; `fa2` tries FA2 → SDPA; `sdpa` forces SDPA. Overrides checkpoint attention-backend fields. FLA and causal-conv1d are selected independently when available. |
| `use_ema` | `false` | Load `weights_ema.pt` instead of `weights.pt`. |
| `policy_seed` | `42` | Policy sampling seed, independent of the benchmark evaluation seed. |
| `checkpoint_path` | Unset | Optional explicit checkpoint directory, handled by the shared resolver. |
| `backbone_path` | From checkpoint config | Backbone config, tokenizer, and image-processor identifier or local snapshot. |
| `unnorm_key` | Unset | Exact normalization block to use when the checkpoint contains more than one. |

## Notes

At initialization, the adapter seeds Python, NumPy, and PyTorch from `policy_seed`,
disables TF32 and cuDNN benchmarking, and requires deterministic PyTorch algorithms.
Each action chunk uses the `episode_control_step_v1` sampling protocol, creating
its own CUDA generator from the SHA256 digest of this compact, ASCII-escaped JSON:

```text
["episode_control_step_v1", policy_seed, task, eval_seed, layout_id, control_step]
```

The seed is the first eight digest bytes, big endian, masked to 63 bits.

Direct callers must supply `env_origin`, camera calibration, and episode metadata
with each observation, for eample:

```python
obs["episode_identity"] = {"task": "stack_bowls", "eval_seed": 0, "layout_id": 42}
obs["control_step"] = 0
```

History requires dense control-step observations. Reset before replaying an
episode; replaying a later step also requires its original first frame and history.
`prepare_case()` and `reset()` clear memory but do not supply missing simulator metadata.
