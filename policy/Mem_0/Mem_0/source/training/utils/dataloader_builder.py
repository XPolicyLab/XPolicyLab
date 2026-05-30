from pathlib import Path
from typing import List, Tuple

import torch
from termcolor import cprint
from torch.utils.data import DataLoader, Subset
from omegaconf import OmegaConf

from source.utils import BOLD, RESET_BOLD, RESET

# Import for random episode dataloader
try:
    from source.dataloader.random_episode_dataloader import RandomEpisodeIterableDataset
    from source.dataloader.dataset_min_max import LeRobot_Dataset as LeRobot_Dataset_MinMax
    RANDOM_EPISODE_AVAILABLE = True
except ImportError:
    RANDOM_EPISODE_AVAILABLE = False

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _resolve_num_workers(cfg) -> int:
    """Prefer dataloader.num_workers, fallback to trainer.num_workers, default 0."""
    dl_cfg = cfg.get("dataloader", {}) if hasattr(cfg, "get") else {}
    tr_cfg = cfg.get("trainer", {}) if hasattr(cfg, "get") else {}
    if getattr(dl_cfg, "get", None) is not None and dl_cfg.get("num_workers") is not None:
        return int(dl_cfg.get("num_workers"))
    return int(tr_cfg.get("num_workers", 0))

def build_rmbench_random_episode_dataloader(
    cfg,
    rank: int,
    world_size: int,
    norm_stats_path: str = None,
) -> Tuple[torch.utils.data.Dataset, DataLoader, str, List[Tuple[str, str, int]]]:
    """
    Construct dataset + dataloader for RMBench LeRobot data using RandomEpisodeIterableDataset.
    Each rank is assigned specific episodes (episode % world_size == rank), and episodes are
    randomly shuffled per rank while frames are read sequentially within each episode.
    
    This ensures:
    - Episodes are never split across ranks
    - Random episode selection per rank
    - Sequential frame reading within episodes
    - Compatible with MemoryBank's episode transition handling
    """
    if not RANDOM_EPISODE_AVAILABLE:
        raise ImportError(
            "RandomEpisodeIterableDataset is not available. "
            "Please ensure source.dataloader.random_episode_dataloader is importable."
        )
    
    trainer_cfg = cfg.get("trainer", {})
    batch_size = trainer_cfg.get("batch_size", 2)
    num_workers = _resolve_num_workers(cfg)
    
    if rank == 0:
        cprint(f"[config] num_workers: {num_workers} (RandomEpisodeIterableDataset)", "cyan")

    dataset_cfg_raw = cfg.get("vla_dataset", {})
    dataset_cfg = (
        OmegaConf.to_container(dataset_cfg_raw, resolve=True)
        if not isinstance(dataset_cfg_raw, dict)
        else dataset_cfg_raw
    )

    repo_id = ""
    if isinstance(dataset_cfg, dict):
        if "repo_id" in dataset_cfg:
            repo_id = dataset_cfg.get("repo_id", "")
        elif len(dataset_cfg) == 1:
            sole_key, nested = next(iter(dataset_cfg.items()))
            if isinstance(nested, dict):
                repo_id = nested.get("repo_id", "")
                dataset_cfg = nested

    if not repo_id:
        raise ValueError("vla_dataset.repo_id must be set for RMBench dataloader.")

    repo_id_resolved = str(Path(str(repo_id)).expanduser())

    features_to_load = dataset_cfg.get(
        "features_to_load",
        [
            "observation.image.head_camera",
            "observation.state",
            "action",
            "subtask",
            "subtask_end",
            "episode_id",
        ],
    )
    action_horizon = cfg.execution_module.action_model.get("action_horizon", 16)
    
    # Get dataloader-specific config
    dataloader_cfg = cfg.get("dataloader", {}) if hasattr(cfg, "get") else {}
    shuffle_episodes = dataloader_cfg.get("shuffle_episodes", True)
    seed = dataloader_cfg.get("seed", cfg.get("seed", 42))
    infinite = dataloader_cfg.get("infinite", True)

    # Create base dataset using LeRobot_Dataset from dataset_min_max
    if norm_stats_path is not None:
        cprint (f"[dataloader] Using normalization stats from: {norm_stats_path}", "cyan")
    base_dataset = LeRobot_Dataset_MinMax(
        repo_id=repo_id_resolved,
        features_to_load=features_to_load,
        action_horizon=action_horizon,
        norm_stats_path=norm_stats_path,
    )

    # Create RandomEpisodeIterableDataset
    iterable_dataset = RandomEpisodeIterableDataset(
        base_dataset=base_dataset,
        rank=rank,
        world_size=world_size,
        shuffle=shuffle_episodes,
        seed=seed,
        infinite=infinite,
    )

    # Create DataLoader
    dataloader = DataLoader(
        iterable_dataset,
        batch_size=batch_size,
        shuffle=False,  # IterableDataset doesn't need shuffle
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=lambda batch: batch,
        drop_last=True,
    )

    # Calculate dataset breakdown
    total_samples = len(base_dataset) if hasattr(base_dataset, "__len__") else "?"
    dataset_breakdown: List[Tuple[str, str, int]] = [
        (Path(repo_id).name, "rmbench", total_samples if isinstance(total_samples, int) else 0)
    ]

    if rank == 0:
        iterable_size = len(iterable_dataset) if hasattr(iterable_dataset, "__len__") else "?"
        cprint(
            f"[data] RandomEpisodeIterableDataset: {BOLD}{iterable_size}{RESET_BOLD} frames assigned to rank {rank}",
            "green"
        )

    return iterable_dataset, dataloader, repo_id, dataset_breakdown
