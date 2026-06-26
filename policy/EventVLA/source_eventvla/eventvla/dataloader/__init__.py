import json
import os
from pathlib import Path

import numpy as np
import torch.distributed as dist
from accelerate.logging import get_logger
from torch.utils.data import DataLoader

from eventvla.dataloader.sequence_sampler import SequentialEpisodeBatchSampler
from eventvla.dataloader.vlm_datasets import make_vlm_dataloader

logger = get_logger(__name__)


def save_dataset_statistics(dataset_statistics, run_dir):
    """Saves a `dataset_statistics.json` file."""
    out_path = run_dir / "dataset_statistics.json"
    with open(out_path, "w") as f_json:
        for _, stats in dataset_statistics.items():
            for k in stats["action"].keys():
                if isinstance(stats["action"][k], np.ndarray):
                    stats["action"][k] = stats["action"][k].tolist()
            if "proprio" in stats:
                for k in stats["proprio"].keys():
                    if isinstance(stats["proprio"][k], np.ndarray):
                        stats["proprio"][k] = stats["proprio"][k].tolist()
            if "num_trajectories" in stats:
                if isinstance(stats["num_trajectories"], np.ndarray):
                    stats["num_trajectories"] = stats["num_trajectories"].item()
            if "num_transitions" in stats:
                if isinstance(stats["num_transitions"], np.ndarray):
                    stats["num_transitions"] = stats["num_transitions"].item()
        json.dump(dataset_statistics, f_json, indent=2)
    logger.info(f"Saved dataset statistics file at path {out_path}")



def _resolve_action_horizon(cfg) -> int:
    framework_cfg = getattr(cfg, "framework", None)
    action_cfg = getattr(framework_cfg, "action_model", None) if framework_cfg is not None else None
    if action_cfg is None:
        return 1

    explicit_horizon = getattr(action_cfg, "action_horizon", None)
    if explicit_horizon is not None:
        return max(1, int(explicit_horizon))

    future_window = getattr(action_cfg, "future_action_window_size", None)
    if future_window is not None:
        return max(1, int(future_window) + 1)

    return 1


def _cfg_bool(value, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "off", "none", "null"}
    return bool(value)



def build_dataloader(cfg, dataset_py="lerobot_datasets_oxe"):  # TODO now here only is get dataset, we need mv dataloader to here

    if dataset_py == "lerobot_datasets":
        from eventvla.dataloader.lerobot_datasets import get_vla_dataset, collate_fn

        vla_dataset_cfg = cfg.datasets.vla_data

        vla_dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)

        num_workers = int(vla_dataset_cfg.get("num_workers", 4))
        use_sequential_episode_sampler = bool(vla_dataset_cfg.get("use_sequential_episode_sampler", False))
        sampling_interval = int(vla_dataset_cfg.get("sampling_interval", 1))
        balance_task_step_counts = _cfg_bool(
            vla_dataset_cfg.get(
                "balance_task_step_counts",
                vla_dataset_cfg.get("balance_dataset_step_counts", False),
            )
        )
        action_horizon = _resolve_action_horizon(cfg)
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        if use_sequential_episode_sampler:
            batch_sampler = SequentialEpisodeBatchSampler(
                dataset=vla_dataset,
                batch_size=int(cfg.datasets.vla_data.per_device_batch_size),
                shuffle_trajectories=bool(vla_dataset_cfg.get("shuffle_trajectories", False)),
                seed=int(getattr(cfg, "seed", 42)),
                sampling_interval=sampling_interval,
                action_horizon=action_horizon,
                balance_dataset_step_counts=balance_task_step_counts,
                rank=rank,
                num_replicas=world_size,
            )
            vla_train_dataloader = DataLoader(
                vla_dataset,
                batch_sampler=batch_sampler,
                collate_fn=collate_fn,
                num_workers=num_workers,
            )
        else:
            vla_train_dataloader = DataLoader(
                vla_dataset,
                batch_size=cfg.datasets.vla_data.per_device_batch_size,
                collate_fn=collate_fn,
                num_workers=num_workers,
                # shuffle=True
            )

        if rank == 0:

            output_dir = Path(cfg.output_dir)
            vla_dataset.save_dataset_statistics(output_dir / "dataset_statistics.json")
        return vla_train_dataloader
    elif dataset_py == "vlm_datasets":
        vlm_data_module = make_vlm_dataloader(cfg)
        vlm_train_dataloader = vlm_data_module["train_dataloader"]

        return vlm_train_dataloader
