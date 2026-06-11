# GigaWorldPolicy Core

This package contains the GigaWorldPolicy model, LeRobot v2.1 dataset loader, XPolicyLab training config, and training utilities used by `policy/GigaWorldPolicy/train.sh`.

## Active Entry Points

- Config: `configs.xpolicylab_gigaworld.config`
- Trainer wrapper: `scripts/train_xpolicylab.py`
- Dataset conversion: `scripts/convert_xpolicylab_hdf5_to_lerobot.py`
- T5 embedding generation: `scripts/generate_t5_embeddings.py`
- Norm statistics: `scripts/comupte_norm_stats.py` or `scripts/fast_norm_stats.py`
- XPolicyLab inference utilities: `experiment/xpolicylab/inference_server.py`

Task selection is controlled by the LeRobot directory passed through `GIGAWORLD_DATA_DIR`. A directory can contain one task or a joint subset with multiple tasks; `meta/tasks.jsonl`, `meta/t5_text_embeds.pt`, and `norm_stats_delta.json` define the training data semantics.

## Environment

```bash
conda activate <your-env>
pip install -e .
```

## Training

Use the policy-level launcher:

```bash
cd <XPolicyLab>/policy/GigaWorldPolicy
GIGAWORLD_DATA_DIR=/path/to/lerobot GIGAWORLD_NUM_FRAMES=28 WANDB_MODE=online GIGAWORLD_WANDB_PROJECT=gwp-xpolicylab bash train.sh xpolicylab_lerobot_v21_video stack_bowls arx_x5 100 joint 930 0,1,2,3,4,5,6,7
```

The model implementation still uses the internal MoT module names in code and checkpoints; those are architecture identifiers, not dataset names.
