#!/usr/bin/env python

import logging
import tqdm
import argparse
from pathlib import Path

import torch
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.transforms.constants import get_feature_mapping, get_mask_mapping
from lerobot.utils.constants import OBS_STATE, ACTION, HF_LEROBOT_HOME
from lerobot.datasets.utils import write_json


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute normalization statistics for LeRobot datasets"
    )

    parser.add_argument(
        "--action_mode",
        type=str,
        choices=["abs", "delta"],
        required=True,
        help="Action mode used to compute statistics (abs or delta).",
    )

    parser.add_argument(
        "--chunk_size",
        type=int,
        required=True,
        help="Chunk size used for delta action computation.",
    )

    parser.add_argument(
        "--repo_id",
        type=str,
        default=None,
        help="LeRobotDataset repo id, or an absolute path to a local dataset directory.",
    )

    parser.add_argument(
        "--repo_id_file",
        type=str,
        default=None,
        help="Optional text file with one dataset repo_id/path per line. When set, stats are aggregated across all listed datasets.",
    )

    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Optional dataset root used when --repo_id is relative.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional output directory. If not set, use default HF_LEROBOT_HOME path.",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Optional exact output path for stats.json. Overrides --output_dir.",
    )

    args = parser.parse_args()
    if not args.repo_id and not args.repo_id_file:
        parser.error("One of --repo_id or --repo_id_file must be provided.")
    return args


def resolve_dataset_entry(repo_id: str, root: str | None) -> tuple[LeRobotDataset, str]:
    repo_path = Path(repo_id)
    if repo_path.is_absolute():
        dataset = LeRobotDataset(str(repo_path))
        dataset_name = repo_path.name
        return dataset, dataset_name

    dataset = LeRobotDataset(repo_id, root=root)
    dataset_name = Path(dataset.root).name
    return dataset, dataset_name


def resolve_datasets(cfg) -> tuple[list[LeRobotDataset], str]:
    if cfg.repo_id_file:
        path = Path(cfg.repo_id_file)
        repo_ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not repo_ids:
            raise ValueError(f"repo_id_file is empty: {path}")

        datasets = []
        for repo_id in repo_ids:
            dataset, _ = resolve_dataset_entry(repo_id, cfg.root)
            datasets.append(dataset)
        return datasets, path.stem

    dataset, dataset_name = resolve_dataset_entry(cfg.repo_id, cfg.root)
    return [dataset], dataset_name


def resolve_dataset(cfg) -> tuple[LeRobotDataset, str]:
    repo_path = Path(cfg.repo_id)
    if repo_path.is_absolute():
        dataset = LeRobotDataset(str(repo_path))
        dataset_name = repo_path.name
        return dataset, dataset_name

    dataset = LeRobotDataset(cfg.repo_id, root=cfg.root)
    dataset_name = Path(dataset.root).name
    return dataset, dataset_name


def resolve_output_path(cfg, dataset_name: str, robot_type: str) -> Path:
    if cfg.output_path:
        return Path(cfg.output_path)
    if cfg.output_dir:
        return Path(cfg.output_dir) / robot_type / cfg.action_mode / "stats.json"
    return HF_LEROBOT_HOME / "stats" / robot_type / cfg.action_mode / "stats.json"


class RunningStats:
    """Compute running statistics of a batch of vectors."""

    def __init__(self):
        self._count = 0
        self._mean = None
        self._mean_of_squares = None
        self._min = None
        self._max = None

    def update(self, batch: torch.Tensor) -> None:
        """
        Update the running statistics with a batch of vectors.

        Args:
            batch (torch.Tensor): shape [..., d]
        """
        batch = batch.to(torch.float32)

        if batch.ndim == 1:
            batch = batch[:, None]

        if batch.ndim > 1:
            batch = batch.reshape(-1, batch.shape[-1])  
        count = batch.shape[0]
        mean = batch.mean(dim=0)
        mean_sq = (batch ** 2).mean(dim=0)
        min_ = batch.min(dim=0).values
        max_ = batch.max(dim=0).values

        if self._count == 0:
            self._count = count
            self._mean = mean
            self._mean_of_squares = mean_sq
            self._min = min_
            self._max = max_
        else:
            total = self._count + count
            w_old = self._count / total
            w_new = count / total

            self._mean = w_old * self._mean + w_new * mean
            self._mean_of_squares = w_old * self._mean_of_squares + w_new * mean_sq
            self._min = torch.minimum(self._min, min_)
            self._max = torch.maximum(self._max, max_)
            self._count = total

    def get_statistics(self) -> dict:
        """Return mean, std, min, max."""
        if self._count == 0:
            raise ValueError("No data has been added yet.")
        var = self._mean_of_squares - self._mean ** 2
        std = torch.sqrt(torch.clamp(var, min=0.0))
        return {
            "min": self._min.tolist(),
            "max": self._max.tolist(),
            "mean": self._mean.tolist(),
            "std": std.tolist(),
            "count": [self._count],
        }


def compute_norm_stats(cfg):
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    dataset_tag = cfg.repo_id_file if cfg.repo_id_file else cfg.repo_id
    print(f"---------- compute statistics for dataset(s): {dataset_tag} ----------")
    datasets, dataset_name = resolve_datasets(cfg)
    dataset = datasets[0]

    action_mode = cfg.action_mode
    chunk_size = cfg.chunk_size
    robot_type = dataset.meta.robot_type

    mask = get_mask_mapping(robot_type, dataset.meta.features)
    mapping = get_feature_mapping(robot_type, dataset.meta.features)

    keys = list(dataset.meta.features.keys())
    [keys.remove(video_key) for video_key in dataset.meta.video_keys]
    [keys.remove(image_key) for image_key in dataset.meta.image_keys]
    stats = {key: RunningStats() for key in keys}

    total_frame = 0

    for dataset_idx, dataset in enumerate(datasets):
        if dataset.meta.robot_type != robot_type:
            raise ValueError(
                f"All datasets must share the same robot_type for aggregated stats. "
                f"Expected {robot_type}, got {dataset.meta.robot_type} for dataset #{dataset_idx}: {dataset.root}"
            )

        from_ids = np.asarray(dataset.meta.episodes['dataset_from_index'])
        to_ids = np.asarray(dataset.meta.episodes['dataset_to_index'])
        total_episodes = dataset.num_episodes

        for from_idx, to_idx in tqdm.tqdm(
            zip(from_ids, to_ids),
            total=total_episodes,
            desc=f"Computing stats [{dataset_idx + 1}/{len(datasets)}]",
        ):
            ep_len = to_idx - from_idx
            total_frame += ep_len
            if ep_len < chunk_size:
                continue
            curr_episode = dataset.hf_dataset.select(np.arange(from_idx, to_idx))
            for key in keys:
                if action_mode == 'abs' or key not in mapping[ACTION]:
                    val = torch.stack(curr_episode[key][:])
                    stats[key].update(val)
            if action_mode == 'delta':
                action = [torch.stack(curr_episode[key][:]) for key in mapping[ACTION]]
                action = [a if a.ndim > 1 else a[:, None] for a in action]
                action = torch.cat(action, dim=-1)
                state = [torch.stack(curr_episode[key][:]) for key in mapping[OBS_STATE]]
                state = [s if s.ndim > 1 else s[:, None] for s in state]
                state = torch.cat(state, dim=-1)
                truncated_state = state[0:(ep_len - chunk_size + 1)]
                action_chunk = action.unfold(dimension=0, size=chunk_size, step=1).permute(0, 2, 1)
                delta_action = action_chunk - torch.where(mask, truncated_state, 0)[:, None]
                sid, eid = 0, 0
                for action_key in mapping[ACTION]:
                    eid += dataset.meta.features[action_key]['shape'][0]
                    stats[action_key].update(delta_action[..., sid:eid])
                    sid = eid
        
    output_dict = {key: stats[key].get_statistics() for key in keys}
    for key in datasets[0].meta.video_keys + datasets[0].meta.image_keys:
        dataset.meta.stats[key]
        visual_stats = datasets[0].meta.stats[key]
        for stat_key in visual_stats:
            if isinstance(visual_stats[stat_key], np.ndarray):
                visual_stats[stat_key] = visual_stats[stat_key].tolist()
            elif isinstance(visual_stats[stat_key], torch.Tensor):
                visual_stats[stat_key] = visual_stats[stat_key].cpu().numpy().tolist()
        output_dict[key] = visual_stats
    output_path = resolve_output_path(cfg, dataset_name, robot_type)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_dict, output_path)

    print(f"total_frame: {total_frame}")
    print(f"stats_path: {output_path}")


if __name__ == "__main__":
    compute_norm_stats(parse_args())
