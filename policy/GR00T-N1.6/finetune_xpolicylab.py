#!/usr/bin/env python3
"""XPolicyLab finetune launcher for GR00T-N1.6 with explicit seed support."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import sys

POLICY_DIR = Path(__file__).resolve().parent
GR00T_DIR = POLICY_DIR / "Isaac-GR00T"
if str(GR00T_DIR) not in sys.path:
    sys.path.insert(0, str(GR00T_DIR))

from gr00t.configs.base_config import get_default_config
from gr00t.experiment.experiment import run


def _load_modality_config(modality_config_path: str) -> None:
    path = Path(modality_config_path)
    if not path.exists() or path.suffix != ".py":
        raise FileNotFoundError(f"Modality config path does not exist or is not a .py file: {path}")
    sys.path.append(str(path.parent))
    importlib.import_module(path.stem)
    print(f"Loaded modality config: {path}")


def _parse_color_jitter(values: list[str] | None):
    if not values:
        return None
    if len(values) % 2 != 0:
        raise ValueError("--color-jitter-params expects key value pairs")
    return {values[i]: float(values[i + 1]) for i in range(0, len(values), 2)}


def _resolve_embodiment_tag(value: str):
    from gr00t.data.embodiment_tags import EmbodimentTag

    if isinstance(value, EmbodimentTag):
        return value
    if value in EmbodimentTag.__members__:
        return EmbodimentTag[value]
    for item in EmbodimentTag:
        if value == item.value:
            return item
    raise ValueError(
        f"Unknown embodiment tag: {value}. "
        f"Use one of {[item.name for item in EmbodimentTag]} or their values."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--embodiment-tag", default="NEW_EMBODIMENT")
    parser.add_argument("--modality-config-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--save-total-limit", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--global-batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
    parser.add_argument("--shard-size", type=int, default=1024)
    parser.add_argument("--num-shards-per-epoch", type=int, default=100000)
    parser.add_argument("--episode-sampling-rate", type=float, default=0.1)
    parser.add_argument("--state-dropout-prob", type=float, default=0.2)
    parser.add_argument("--wandb-project", default="finetune-gr00t-n1d6")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--tune-llm", action="store_true")
    parser.add_argument("--tune-visual", action="store_true")
    parser.add_argument("--no-tune-projector", action="store_true")
    parser.add_argument("--no-tune-diffusion-model", action="store_true")
    parser.add_argument("--random-rotation-angle", type=int, default=None)
    parser.add_argument("--extra-augmentation-config", default=None)
    parser.add_argument("--color-jitter-params", nargs="*", default=None)
    args = parser.parse_args()

    os.environ.setdefault("LOGURU_LEVEL", "INFO")
    _load_modality_config(args.modality_config_path)

    embodiment = _resolve_embodiment_tag(args.embodiment_tag)
    embodiment_tag = embodiment.value

    config = get_default_config().load_dict(
        {
            "data": {
                "download_cache": False,
                "datasets": [
                    {
                        "dataset_paths": [args.dataset_path],
                        "mix_ratio": 1.0,
                        "embodiment_tag": embodiment_tag,
                    }
                ],
            }
        }
    )
    config.load_config_path = None

    config.model.tune_llm = args.tune_llm
    config.model.tune_visual = args.tune_visual
    config.model.tune_projector = not args.no_tune_projector
    config.model.tune_diffusion_model = not args.no_tune_diffusion_model
    config.model.state_dropout_prob = args.state_dropout_prob
    config.model.random_rotation_angle = args.random_rotation_angle
    config.model.color_jitter_params = _parse_color_jitter(args.color_jitter_params)
    config.model.extra_augmentation_config = (
        json.loads(args.extra_augmentation_config) if args.extra_augmentation_config else None
    )
    config.model.load_bf16 = False
    config.model.reproject_vision = False
    config.model.eagle_collator = True
    config.model.model_name = "nvidia/Eagle-Block2A-2B-v2"
    config.model.backbone_trainable_params_fp32 = True
    config.model.use_relative_action = True

    config.data.seed = args.seed
    config.data.shard_size = args.shard_size
    config.data.episode_sampling_rate = args.episode_sampling_rate
    config.data.num_shards_per_epoch = args.num_shards_per_epoch

    config.training.experiment_name = args.experiment_name
    config.training.start_from_checkpoint = args.base_model_path
    config.training.optim = "adamw_torch"
    config.training.global_batch_size = args.global_batch_size
    config.training.gradient_accumulation_steps = args.gradient_accumulation_steps
    config.training.dataloader_num_workers = args.dataloader_num_workers
    config.training.learning_rate = args.learning_rate
    config.training.output_dir = args.output_dir
    config.training.save_steps = args.save_steps
    config.training.save_total_limit = args.save_total_limit
    config.training.num_gpus = args.num_gpus
    config.training.use_wandb = args.use_wandb
    config.training.max_steps = args.max_steps
    config.training.weight_decay = args.weight_decay
    config.training.warmup_ratio = args.warmup_ratio
    config.training.wandb_project = args.wandb_project

    run(config)


if __name__ == "__main__":
    main()
