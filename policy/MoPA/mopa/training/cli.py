"""Train the arm-only MoPA policy from a prepared RGB-array dataset."""
from __future__ import annotations

import argparse
from importlib.resources import files
import json
import os
from pathlib import Path
import random

import numpy as np
import torch

from ..common import FORMAT_VERSION
from ..data.dataset import EpisodeDataset
from .trainer import train_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--base-vlm", default=os.environ.get("MOPA_BASE_VLM"))
    parser.add_argument("--config", type=Path, help="JSON overrides for the packaged model configuration.")
    parser.add_argument("--device", default=os.environ.get("TRAIN_DEVICE", "cuda"))
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--backbone-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-8)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cache-episodes", type=int, default=2)
    parser.add_argument("--save-interval", type=int, default=10000)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    return parser


def run_training(args: argparse.Namespace) -> float:
    """Resolve and save the complete model/training configuration before optimization."""
    device = torch.device(args.device)
    if device.type == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.random.default_generator.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    config = json.loads(files("mopa").joinpath("configs/model.json").read_text(encoding="utf-8"))
    if args.config:
        config.update(json.loads(args.config.read_text(encoding="utf-8")))
    base_vlm = args.base_vlm or config.get("base_vlm")
    if not base_vlm or not Path(base_vlm).expanduser().is_dir():
        raise ValueError("Supply --base-vlm or MOPA_BASE_VLM pointing to local Qwen3-VL weights.")
    if config["action_horizon"] != 4 or config["num_query_tokens"] != 8:
        raise ValueError("MoPA requires four actions and one set of eight arm queries.")
    if config.get("base_num_query_tokens", 0) != 0 or config.get("in_context_condition_dim", 0) != 0:
        raise ValueError("Arm-only MoPA requires zero base queries and zero base context dimensions.")
    if device.type == "cpu":
        config["backbone_dtype"] = "float32"
    dataset = EpisodeDataset(args.dataset, action_horizon=config["action_horizon"], cache_episodes=args.cache_episodes)
    for key in ("env_cfg_type", "action_type", "robot_action_dim_info", "cameras", "image_size"):
        if key in dataset.metadata:
            config[key] = dataset.metadata[key]
        else:
            config.pop(key, None)
    config.update(format_version=FORMAT_VERSION, action_dim=dataset.dim, state_dim=dataset.dim,
                  base_vlm=str(Path(base_vlm).expanduser().resolve()))
    config["training"] = {key: getattr(args, key) for key in (
        "seed", "device", "max_steps", "batch_size", "learning_rate", "backbone_learning_rate",
        "weight_decay", "max_grad_norm", "num_workers", "cache_episodes", "save_interval", "log_interval", "warmup_steps")}
    config["training"].update(dataset=str(Path(args.dataset).expanduser().resolve()),
                              output=str(Path(args.output).expanduser().resolve()))
    output = Path(args.output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Output path is not an empty directory: {output}; choose a new --output.")
    from ..models.policy import ArmQueryPolicy

    model = ArmQueryPolicy(config)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return train_model(model, dataset, config, output, device=str(device), max_steps=args.max_steps,
                       batch_size=args.batch_size, learning_rate=args.learning_rate,
                       backbone_learning_rate=args.backbone_learning_rate,
                       weight_decay=args.weight_decay, max_grad_norm=args.max_grad_norm,
                       num_workers=args.num_workers, save_interval=args.save_interval,
                       log_interval=args.log_interval, warmup_steps=args.warmup_steps, seed=args.seed)


def main(argv: list[str] | None = None) -> None:
    run_training(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
