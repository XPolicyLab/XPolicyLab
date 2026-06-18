#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))


def _jsonable(value: Any):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _set_if_not_none(mapping: dict, key: str, value: Any):
    if value is not None:
        mapping[key] = value


def _first_dataset_template(config: dict) -> dict:
    train = config.setdefault("dataloaders", {}).setdefault("train", {})
    current = train.get("data_or_config") or []
    if current:
        return copy.deepcopy(current[0])
    return {
        "_class_name": "LeRobotDataset",
        "data_path": "",
        "data_size": None,
        "delta_info": {"action": int(config.get("num_frames", 24))},
        "delta_frames": {},
        "video_backend": "pyav",
        "robotype": os.environ.get("GIGAWORLD_ROBOTYPE", "arx5"),
    }


def main():
    parser = argparse.ArgumentParser(description="Run GigaWorldPolicy training through XPolicyLab conventions")
    parser.add_argument("--config", required=True, help="Dotted config path, e.g. configs.xpolicylab_gigaworld.config")
    parser.add_argument("--project_dir", required=True)
    parser.add_argument("--record_config", required=True)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--force_data_dir", action="store_true")
    parser.add_argument("--norm_path", default=None)
    parser.add_argument("--pretrained_path", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--gpu_ids", default=None)
    parser.add_argument("--wandb_project", default=None)
    parser.add_argument("--wandb_name", default=None)
    parser.add_argument("--wandb_mode", default=None)
    parser.add_argument("--model_action_dim", type=int, default=None)
    parser.add_argument("--model_state_dim", type=int, default=None)
    parser.add_argument("--num_frames", type=int, default=None)
    parser.add_argument("--action_chunk", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    from world_action_model.runtime import load_config, resolve_runner

    config = copy.deepcopy(load_config(args.config))
    project_dir = Path(args.project_dir).resolve()
    project_dir.mkdir(parents=True, exist_ok=True)
    config["project_dir"] = str(project_dir)

    models = config.setdefault("models", {})
    models["view_dir"] = str(project_dir)
    _set_if_not_none(models, "pretrained", args.pretrained_path)
    _set_if_not_none(models, "action_dim", args.model_action_dim)
    _set_if_not_none(models, "state_dim", args.model_state_dim)

    train_cfg = config.setdefault("train", {})
    if args.seed is not None:
        train_cfg["seed"] = max(1, int(args.seed))

    wandb_cfg = config.setdefault("wandb", {})
    _set_if_not_none(wandb_cfg, "project", args.wandb_project)
    _set_if_not_none(wandb_cfg, "name", args.wandb_name)
    _set_if_not_none(wandb_cfg, "mode", args.wandb_mode)

    train_loader = config.setdefault("dataloaders", {}).setdefault("train", {})
    transform = train_loader.setdefault("transform", {})
    _set_if_not_none(transform, "model_action_dim", args.model_action_dim)
    _set_if_not_none(transform, "model_state_dim", args.model_state_dim)
    _set_if_not_none(transform, "num_frames", args.num_frames)
    if args.norm_path:
        transform["norm_path"] = args.norm_path

    if args.data_dir:
        data_dir = Path(args.data_dir).resolve()
        if data_dir.exists() or args.force_data_dir:
            ds = _first_dataset_template(config)
            ds["data_path"] = str(data_dir)
            if args.num_frames is not None:
                ds["delta_info"] = {"action": int(args.num_frames)}
                view_keys = transform.get("view_keys") or []
                offsets = [0, args.num_frames // 4, args.num_frames // 2, (3 * args.num_frames) // 4, args.num_frames]
                if view_keys:
                    ds["delta_frames"] = {k: offsets for k in view_keys}
            train_loader["data_or_config"] = [ds]
        else:
            print(f"[GigaWorldPolicy] data_dir does not exist, keeping config data list: {data_dir}")

    record_path = Path(args.record_config).resolve()
    record_path.parent.mkdir(parents=True, exist_ok=True)
    with open(record_path, "w", encoding="utf-8") as f:
        json.dump(_jsonable(config), f, indent=2, ensure_ascii=False)
    print(f"[GigaWorldPolicy] effective config written to {record_path}")
    print(f"[GigaWorldPolicy] project_dir={project_dir}")

    if args.dry_run:
        print("[GigaWorldPolicy] dry run enabled, skip training")
        return

    runners = config.get("runners", [])
    if not runners:
        raise ValueError("No runners specified in config")
    trainer = resolve_runner(runners[0])(config)
    trainer.run()


if __name__ == "__main__":
    main()
