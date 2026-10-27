# Copyright (c) 2026 Awomo-WAM Team. Licensed under the Apache License, Version 2.0.
"""Launch Awomo-0.5I RoboDojo SFT from a pretrain checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1]))
sys.path.insert(0, str(ROOT))


def resolve_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text())

    def expand(value):
        if isinstance(value, dict):
            return {key: expand(item) for key, item in value.items()}
        if isinstance(value, list):
            return [expand(item) for item in value]
        if isinstance(value, str):
            result = os.path.expandvars(value)
            if re.search(r"\$\{[A-Za-z_][A-Za-z_0-9]*\}", result):
                raise ValueError(f"Unset environment variable in config: {value}")
            return result
        return value

    config = expand(config)
    steps = config["step_scheduler"]
    assert (steps["max_steps"], steps["global_batch_size"], steps["local_batch_size"]) == (6755, 256, 8)
    assert config["fsdp"]["dp_size"] == 32 and config["fsdp"]["dp_replicate_size"] == 4
    assert config["gripper_experiment"] == {
        "loss_multiplier": 2.0, "sampling": "uniform", "stride": 1, "total_steps": 6755
    }
    dataset = config["data"]["dataloader"]["dataset"]
    assert dataset["window_stride"] == 1 and dataset["val_set_proportion"] == 0
    assert config["action_patch"]["bsmntw_enabled"] and not config["action_patch"]["self_flow_enabled"]
    return config


def source_digest() -> str:
    digest = hashlib.sha256()
    for path in sorted([*(ROOT / "training").rglob("*.py"), *(ROOT / "awomo").rglob("*.py")]):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "training/configs/sft.yaml")
    parser.add_argument("--check-config", action="store_true", help="Validate config and input paths; do not train.")
    parser.add_argument("--resume", type=Path, help="Restore all training state from a native SFT checkpoint.")
    parser.add_argument("--seed", type=int, help="Override both model and sampler seed.")
    args = parser.parse_args()
    config = resolve_config(args.config)
    if args.seed is not None:
        config["seed"] = config["data"]["dataloader"]["seed"] = args.seed
    dataset = config["data"]["dataloader"]["dataset"]
    required = [dataset["index_file"], dataset["pretrained_norm_stats"],
                config["checkpoint"]["base_checkpoint"],
                config["action_patch"]["dit"]["pretrained_path"],
                config["action_patch"]["encoders"]["text_encoder_path"],
                config["action_patch"]["encoders"]["vae_path"]]
    missing = [path for path in required if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"Missing explicit training inputs: {missing}")
    initialization = Path(config["checkpoint"]["base_checkpoint"])
    sys.path.insert(0, str(ROOT))
    from awomo.awomo.factory import read_checkpoint_config
    base_config = read_checkpoint_config(initialization, verify_checksums=False, allow_base_only=True)
    if base_config["artifact_kind"] != "base":
        raise ValueError("Fresh SFT requires a pretrain checkpoint; use --resume for training state.")
    if args.resume:
        if not (args.resume / ".complete").is_file():
            raise ValueError("Cannot resume from an incomplete checkpoint")
        config["checkpoint"]["restore_from"] = str(args.resume.resolve())
    if args.check_config:
        print(json.dumps({"config_valid": True, "input_paths_exist": True, "training_started": False,
                          "source_sha256": source_digest(), "steps": 6755, "global_batch_size": 256}, indent=2))
        return
    if int(os.environ.get("WORLD_SIZE", "1")) != 32:
        raise ValueError("This RoboDojo SFT configuration requires torchrun WORLD_SIZE=32 (4 nodes x 8 GPUs).")
    run_dir = Path(config["log"]["log_dir"]).parent
    if int(os.environ["RANK"]) == 0:
        if run_dir.exists() and not args.resume:
            raise FileExistsError(f"Refusing to reuse an existing run directory: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=bool(args.resume))
        (run_dir / "resolved_config.json").write_text(json.dumps(config, indent=2) + "\n")
    sys.path.insert(0, str(ROOT))
    from awomo.awomo.training.config.loader import ConfigNode
    from awomo.awomo.training.robodojo import (
        RoboDojoTrainer,
    )
    import torch.distributed as dist

    from awomo.awomo.training.builders import get_git_commit

    os.environ["SOURCE_DIGEST"] = source_digest()
    revision = get_git_commit(ROOT.parents[1])
    if revision:
        os.environ["GIT_COMMIT"] = revision
    else:
        os.environ.pop("GIT_COMMIT", None)
    recipe = RoboDojoTrainer(ConfigNode(config))
    try:
        recipe.setup()
        recipe.run_train_validation_loop()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
