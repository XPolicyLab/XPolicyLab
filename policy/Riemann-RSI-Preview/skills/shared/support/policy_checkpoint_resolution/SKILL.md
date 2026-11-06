---
name: policy_checkpoint_resolution
description: "Resolves a checkpoint directory from deploy/eval configuration with priority explicit model_path, then ckpt_name as an existing directory, then exp_dir/hf_model/<ckpt_name>. For last or latest it selects the highest numeric step-* directory."
---

# Policy checkpoint resolution

## Function

Resolves a checkpoint directory from deploy/eval configuration with priority explicit model_path, then ckpt_name as an existing directory, then exp_dir/hf_model/<ckpt_name>. For last or latest it selects the highest numeric step-* directory.

## Inputs

- cfg: dict possibly containing model_path, ckpt_name, and exp_dir
- filesystem paths under exp_dir or hf_model

## Outputs

- str path to the selected checkpoint directory

## Callable interface

Implementation: [implementation.py](implementation.py).

- `_resolve_model_path(cfg: dict)`

## Constraints

- Raises ValueError when neither model_path nor exp_dir is provided
- Falls back from exp_dir/hf_model to exp_dir when hf_model is absent
- last/latest raises FileNotFoundError if no step-* directories exist; missing final checkpoint directory raises FileNotFoundError
- A non-step ckpt_name is prefixed with step-; adapter-bound os global and exception behavior are preserved
