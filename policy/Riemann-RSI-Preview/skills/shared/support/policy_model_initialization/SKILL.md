---
name: policy_model_initialization
description: "Validates skyrobot_root, model/encoder paths, norm-stat hash, training normalization mode, and visible CUDA device indices, then constructs SkyrobotWAMPolicy with configured inference and T5 options. It optionally caches task warmup prompts and unloads T5 before simulator startup."
---

# Policy model initialization

## Function

Validates skyrobot_root, model/encoder paths, norm-stat hash, training normalization mode, and visible CUDA device indices, then constructs SkyrobotWAMPolicy with configured inference and T5 options. It optionally caches task warmup prompts and unloads T5 before simulator startup.

## Inputs

- model_cfg: mapping with skyrobot_root, norm_stats_sha256, norm_mode, encoder_path, and device/t5_device
- model_cfg optional paths/options: model_path/exp_dir, train_config_path, norm_stats_path, num_latents, num_inference_steps, cfg_scale, t5_cpu, t5_lazy, t5_unload_after_encode, verbose, task_name, t5_warmup_prompts_by_task

## Outputs

- No return value; sets self.model_cfg, self.data_cfg, self.policy, self._pending_reset=True, and self._latest=None
- Raises ValueError for invalid root/encoder/device/norm-mode/visible-GPU conditions or hash failure

## Callable interface

Implementation: [implementation.py](implementation.py).

- `__init__(self, model_cfg)`

## Constraints

- Requires skyrobot_root to be an existing directory and inserts it at the front of sys.path
- Encoder path must be an existing directory; the error notes Wan2.2_VAE.pth + T5 weights
- norm_mode must equal training TOML data.norm_mode; norm_stats is checked by _verify_norm_stats
- Bound into adapter __init__ and depends on module globals _resolve_model_path, _verify_norm_stats, _load_train_config, and self
