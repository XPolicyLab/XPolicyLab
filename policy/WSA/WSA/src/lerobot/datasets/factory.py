#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import re
import os
import logging
from pprint import pformat
from pathlib import Path
from collections import defaultdict
from omegaconf import OmegaConf, DictConfig

import math
import torch
import torch.distributed as dist

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
)
from lerobot.datasets.transforms import ImageTransforms
from lerobot.datasets.utils import load_json, cast_stats_to_numpy
from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset
from lerobot.datasets.transformed_dataset import (
    TransformedLeRobotDataset, 
    TransformedStreamingLeRobotDataset, 
    MultiLeRobotDataset, 
    MultiStreamingLeRobotDataset, 
)
from lerobot.policies.fastwam.configuration_fastwam import FastWAMDatasetConfig
from lerobot.policies.WSA_Large.configuration_wsa_large import WSALargeDatasetConfig
from lerobot.policies.WSA_Base.configuration_wsa_base import WSABaseDatasetConfig, RoboChallengeRawW1DatasetConfig
from lerobot.policies.names import WSA_LARGE, WSA_LARGE_LEGACY_ALIASES, is_wsa_base
from lerobot.transforms.constants import get_feature_mapping, get_image_mapping, infer_embodiment_variant
from lerobot.utils.constants import ACTION, OBS_PREFIX, REWARD, OBS_STATE
from lerobot.utils.constants import HF_LEROBOT_HOME

IMAGENET_STATS = {
    "mean": [[[0.485]], [[0.456]], [[0.406]]],  # (c,1,1)
    "std": [[[0.229]], [[0.224]], [[0.225]]],  # (c,1,1)
}


def get_rank_and_world_size() -> tuple[int, int]:
    """Get the global rank and world_size.

    If torch.distributed is not initialized, fall back to (0, 1).
    """
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def assign_repo_ids_for_rank(
    repo_ids: list[str],
    rank: int,
    world_size: int,
) -> list[str]:
    """
    Given the global list of repo_ids and a global rank, return the subset of
    repo_ids that this rank should load.

    Rules:
    - If the number of repo_ids >= world_size:
        Perform a disjoint contiguous split so that each rank gets a unique,
        non-overlapping chunk.
    - If the number of repo_ids < world_size:
        Allow repetition and assign repo_ids in a round-robin (modulo) manner,
        ensuring every rank receives at least one repo_id and no rank is empty.
    """
    n = len(repo_ids)
    if n == 0:
        raise ValueError("assign_repo_ids_for_rank: repo_ids is empty.")

    # Case 1: Enough repo_ids to distribute without overlap.
    if n >= world_size:
        return [rid for i, rid in enumerate(repo_ids) if i % world_size == rank]

    # Case 2: Fewer repo_ids than ranks → use round-robin assignment.
    idx = rank % n
    return [repo_ids[idx]]


def find_info_json_path_for_repo(cfg: TrainPipelineConfig, repo_id: str) -> Path | None:
    if cfg.dataset.root is not None: 
        root = Path(cfg.dataset.root)
        return root / repo_id / "meta" / "info.json"
    else:
        return HF_LEROBOT_HOME / repo_id / "meta" / "info.json"


def load_info_for_repos(
    cfg: TrainPipelineConfig,
    repo_ids: list[str],
) -> dict[str, int]:
    frames_map: dict[str, int] = {}
    episodes_map: dict[str, int] = {}

    for rid in repo_ids:
        info_path = find_info_json_path_for_repo(cfg, rid)
        info = load_json(info_path)
        frames_map[rid] = int(info["total_frames"])
        episodes_map[rid] = int(info["total_episodes"])

    return frames_map, episodes_map


def resolve_repo_ids(cfg: TrainPipelineConfig) -> list[str]:
    repo_id_file = getattr(cfg.dataset, "repo_id_file", None)
    if repo_id_file:
        path = Path(repo_id_file)
        if not path.is_file():
            raise FileNotFoundError(f"dataset.repo_id_file does not exist: {path}")

        repo_ids = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                repo_id = line.rstrip("\n")
                if repo_id.strip():
                    repo_ids.append(repo_id)

        if not repo_ids:
            raise ValueError(f"dataset.repo_id_file is empty: {path}")

        return repo_ids

    return [rid for rid in cfg.dataset.repo_id.split(" ") if rid]


def compute_balanced_repo_assignment(
    repo_ids: list[str],
    frames_map: dict[str, int],
    world_size: int,
) -> list[list[str]]:
    """
    Compute a balanced assignment of repo_ids to ranks based on total_frames.

    Goals:
    - Every rank gets at least one repo_id.
    - The total_frames sum per rank is as close as possible across ranks.
    - For len(repo_ids) >= world_size:
        * Each repo_id is used at most once (no duplication).
    - For len(repo_ids) < world_size:
        * Allow duplication of repo_ids across ranks to avoid empty ranks.

    Strategy:
    - Use a greedy LPT-style algorithm:
        1) Sort repo_ids by descending total_frames (ties broken by repo_id string).
        2) Repeatedly assign the next repo_id to the rank with the smallest current load.
        3) If len(repo_ids) < world_size, keep cycling over repo_ids until we have
           assigned at least one repo_id to every rank.
    """
    if world_size <= 0:
        raise ValueError("world_size must be positive.")

    n = len(repo_ids)
    if n == 0:
        raise ValueError("compute_balanced_repo_assignment: repo_ids is empty.")

    # Initialize per-rank containers
    rank_to_repos: list[list[str]] = [[] for _ in range(world_size)]
    rank_loads: list[int] = [0 for _ in range(world_size)]

    # Sort repo_ids by descending total_frames, tie-breaker by repo_id for determinism
    def frames_key(rid: str) -> tuple[int, str]:
        # Use negative so that larger total_frames come first
        return (-frames_map.get(rid, 0), rid)

    sorted_repos = sorted(repo_ids, key=frames_key)

    if n >= world_size:
        # Case A: enough repos to avoid duplication.
        # Greedy: always assign the next repo to the rank with the smallest current load.
        for rid in sorted_repos:
            min_rank = min(range(world_size), key=lambda r: (rank_loads[r], r))
            rank_to_repos[min_rank].append(rid)
            rank_loads[min_rank] += frames_map.get(rid, 0)
    else:
        # Case B: fewer repos than ranks -> we must allow duplication
        #
        # Strategy:
        # - Still use LPT greedily, but we keep "expanding" sorted_repos in cycles
        #   until every rank has at least one repo.
        # - This keeps total_frames per rank roughly balanced, even with repetition.
        assignments_done = 0
        idx = 0
        while assignments_done < world_size:
            rid = sorted_repos[idx % n]
            min_rank = min(range(world_size), key=lambda r: (rank_loads[r], r))
            rank_to_repos[min_rank].append(rid)
            rank_loads[min_rank] += frames_map.get(rid, 0)

            assignments_done += 1
            idx += 1

    logging.info(
            f"total_frames={sum(rank_loads)}"
        )
    for r in range(world_size):
        logging.info(
            f"[dist_loading] rank {r}: "
            f"num_frames={rank_loads[r]}"
        )

    return rank_to_repos


def group_repo_ids_by_rules(
    repo_ids: list[str],
    groups_cfg: DictConfig,
) -> tuple[dict[str, str], dict[str, list[str]], list[str]]:
    repo_to_group: dict[str, str] = {}
    group_to_repos: dict[str, list[str]] = defaultdict(list)
    ordered_group_names = [str(g.name) for g in groups_cfg.groups]

    for rid in repo_ids:
        matched = False
        for g in groups_cfg.groups:
            if re.search(g.match, rid):
                group_name = str(g.name)
                repo_to_group[rid] = group_name
                group_to_repos[group_name].append(rid)
                matched = True
                break
        if not matched:
            repo_to_group[rid] = "__default__"
            group_to_repos["__default__"].append(rid)

    if "__default__" in group_to_repos:
        default_repos = group_to_repos["__default__"]
        preview_limit = _env_int("LEROBOT_WEIGHT_RULES_DEFAULT_GROUP_LIMIT", 20)
        preview = "\n".join(f"  {rid}" for rid in default_repos[:max(preview_limit, 0)])
        if len(default_repos) > max(preview_limit, 0):
            preview += f"\n  ... +{len(default_repos) - max(preview_limit, 0)} more"
        message = (
            f"{len(default_repos)} repo(s) did not match any explicit weight_rules group and fell into "
            f"`default`. This should be reviewed instead of silently using default sampling config."
        )
        if preview:
            message = f"{message}\nUnmatched repo examples:\n{preview}"

        config_default_mode = str(getattr(getattr(groups_cfg, "default", None), "on_match", "warn"))
        mode = os.environ.get("LEROBOT_WEIGHT_RULES_DEFAULT_GROUP_MODE", config_default_mode).strip().lower()
        if mode in {"error", "raise", "strict", "fail"}:
            raise ValueError(message)
        if mode not in {"off", "quiet", "none", "0", "false"}:
            logging.warning(message)
        ordered_group_names.append("__default__")

    return repo_to_group, group_to_repos, ordered_group_names


def compute_group_balanced_repo_assignment(
    repo_ids: list[str],
    frames_map: dict[str, int],
    world_size: int,
    groups_cfg: DictConfig,
) -> list[list[str]]:
    """
    Compute a source-aware repo assignment for dist_loading.

    Goals:
    - Spread repos from each source group across ranks as evenly as possible.
    - Keep total frame counts per rank as balanced as possible.
    - If a group has at least `world_size` repos, try to give every rank at least
      one repo from that group.

    The implementation is intentionally simple:
    - Match repos into groups using the same regex rules as weight_rules.
    - For each group, sort repos by descending total_frames.
    - Seed one repo per rank when the group is large enough.
    - Assign the remaining repos greedily using (group_load, total_load, count).
    """
    if world_size <= 0:
        raise ValueError("world_size must be positive.")

    if not repo_ids:
        raise ValueError("compute_group_balanced_repo_assignment: repo_ids is empty.")

    _, group_to_repos, ordered_group_names = group_repo_ids_by_rules(repo_ids, groups_cfg)

    rank_to_repos: list[list[str]] = [[] for _ in range(world_size)]
    rank_total_loads: list[int] = [0 for _ in range(world_size)]
    rank_group_loads: dict[str, list[int]] = {
        group_name: [0 for _ in range(world_size)] for group_name in ordered_group_names
    }
    rank_group_counts: dict[str, list[int]] = {
        group_name: [0 for _ in range(world_size)] for group_name in ordered_group_names
    }

    def frames_key(rid: str) -> tuple[int, str]:
        return (-frames_map.get(rid, 0), rid)

    def assign_repo_to_rank(rid: str, group_name: str, rank: int) -> None:
        rank_to_repos[rank].append(rid)
        frame_count = frames_map.get(rid, 0)
        rank_total_loads[rank] += frame_count
        rank_group_loads[group_name][rank] += frame_count
        rank_group_counts[group_name][rank] += 1

    for group_name in ordered_group_names:
        repos = group_to_repos.get(group_name, [])
        if not repos:
            continue

        sorted_repos = sorted(repos, key=frames_key)
        start_idx = 0

        if len(sorted_repos) >= world_size:
            for rid in sorted_repos[:world_size]:
                candidate_ranks = [r for r in range(world_size) if rank_group_counts[group_name][r] == 0]
                rank = min(
                    candidate_ranks,
                    key=lambda r: (rank_total_loads[r], rank_group_loads[group_name][r], r),
                )
                assign_repo_to_rank(rid, group_name, rank)
            start_idx = world_size
        elif len(sorted_repos) < world_size:
            logging.warning(
                f"[dist_loading] group '{group_name}' has only {len(sorted_repos)} repos for "
                f"{world_size} ranks; some ranks will not receive this source."
            )

        for rid in sorted_repos[start_idx:]:
            rank = min(
                range(world_size),
                key=lambda r: (
                    rank_group_loads[group_name][r],
                    rank_total_loads[r],
                    rank_group_counts[group_name][r],
                    r,
                ),
            )
            assign_repo_to_rank(rid, group_name, rank)

    logging.info(f"total_frames={sum(rank_total_loads)}")
    for r in range(world_size):
        group_summary_parts = []
        for group_name in ordered_group_names:
            group_count = rank_group_counts[group_name][r]
            if group_count > 0:
                group_frames = rank_group_loads[group_name][r]
                group_summary_parts.append(f"{group_name}: repos={group_count}, frames={group_frames}")
        group_summary = "; ".join(group_summary_parts) if group_summary_parts else "no groups"
        logging.info(
            f"[dist_loading] rank {r}: num_frames={rank_total_loads[r]} | {group_summary}"
        )

    return rank_to_repos


def compute_repo_weights(
    repo_ids: list[str],
    frames_map: dict[str, int],
    episodes_map: dict[str, int],
    groups_cfg: DictConfig,
) -> dict[str, float]:
    """
    Compute global repo-level sampling weights from YAML group config.

    Returns:
        dict[str, float]: repo_id -> normalized weight (sum to 1)
    """
    _, group_to_repos, _ = group_repo_ids_by_rules(repo_ids, groups_cfg)

    group_budget = {}
    for g in groups_cfg.groups:
        group_budget[g.name] = float(g.total_weight)

    if "__default__" in group_to_repos:
        group_budget["__default__"] = float(groups_cfg.default.total_weight)

    repo_weights = {}

    for group_name, repos in group_to_repos.items():
        budget = group_budget[group_name]

        if group_name == "__default__":
            inside = groups_cfg.default.inside
            gamma = float(getattr(groups_cfg.default, "gamma", 1.0))
        else:
            g = next(x for x in groups_cfg.groups if x.name == group_name)
            inside = g.inside
            gamma = float(getattr(g, "gamma", 1.0))

        # compute raw scores
        scores = []
        for rid in repos:
            if inside == "frames_pow":
                s = frames_map[rid] ** gamma
            elif inside == "episodes_pow":
                s = episodes_map[rid] ** gamma
            elif inside == "uniform":
                s = 1.0
            else:
                raise ValueError(f"Unknown inside mode: {inside}")
            scores.append(s)

        total = sum(scores)
        for rid, s in zip(repos, scores):
            repo_weights[rid] = budget * (s / total)

    Z = sum(repo_weights.values())
    for rid in repo_weights:
        repo_weights[rid] /= Z

    return repo_weights


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _short_repo_id_for_log(repo_id: str, max_len: int = 72) -> str:
    parts = Path(repo_id).parts
    if parts:
        dataset_idx = next((i for i, part in enumerate(parts) if part.lower() == "dataset"), None)
        if dataset_idx is not None and dataset_idx + 1 < len(parts):
            short = "/".join(parts[dataset_idx + 1 :])
        else:
            short = "/".join(parts[-3:])
    else:
        short = repo_id

    if len(short) <= max_len:
        return short
    return "..." + short[-(max_len - 3) :]


def _format_repo_assignment_for_log(
    *,
    rank: int,
    world_size: int,
    repo_ids: list[str],
    frames_map: dict[str, int] | None,
    episodes_map: dict[str, int] | None,
    repo_weights_map: dict[str, float] | None,
    groups_cfg: DictConfig | None,
    label: str,
) -> str | None:
    mode = os.environ.get("LEROBOT_REPO_ASSIGNMENT_LOG_MODE", "summary").strip().lower()
    if mode in {"0", "false", "off", "none", "quiet"}:
        return None

    frames_map = frames_map or {}
    episodes_map = episodes_map or {}
    repo_weights_map = repo_weights_map or {}

    def format_repo(rid: str, *, short: bool = True) -> str:
        parts = [_short_repo_id_for_log(rid) if short else rid]
        if rid in repo_weights_map:
            parts.append(f"w={repo_weights_map[rid]:.6f}")
        if rid in frames_map:
            parts.append(f"f={frames_map[rid]:,}")
        if rid in episodes_map:
            parts.append(f"ep={episodes_map[rid]:,}")
        return parts[0] if len(parts) == 1 else f"{parts[0]} ({', '.join(parts[1:])})"

    if mode in {"full", "verbose", "all"}:
        return (
            f"[rank={rank:02d}/{world_size:02d}] {label} repo_ids_for_this_rank: repos={len(repo_ids)}\n"
            + "\n".join(f"[rank {rank}] repo_id = {format_repo(rid, short=False)}" for rid in repo_ids)
        )

    repo_to_group: dict[str, str] = {}
    ordered_group_names: list[str] = []
    if groups_cfg is not None:
        repo_to_group, _, ordered_group_names = group_repo_ids_by_rules(repo_ids, groups_cfg)

    group_counts: dict[str, int] = defaultdict(int)
    group_frames: dict[str, int] = defaultdict(int)
    group_weights: dict[str, float] = defaultdict(float)
    for rid in repo_ids:
        group_name = repo_to_group.get(rid, "all")
        group_counts[group_name] += 1
        group_frames[group_name] += frames_map.get(rid, 0)
        group_weights[group_name] += repo_weights_map.get(rid, 0.0)

    if ordered_group_names:
        group_order = [g for g in ordered_group_names if group_counts.get(g, 0) > 0]
    else:
        group_order = sorted(group_counts)

    group_summary = ", ".join(
        f"{group}: {group_counts[group]} repos/{group_frames[group]:,} frames"
        + (f"/w={group_weights[group]:.4f}" if repo_weights_map else "")
        for group in group_order
    )
    if not group_summary:
        group_summary = "no repos"

    total_frames = sum(frames_map.get(rid, 0) for rid in repo_ids)
    total_episodes = sum(episodes_map.get(rid, 0) for rid in repo_ids)
    total_weight = sum(repo_weights_map.get(rid, 0.0) for rid in repo_ids)

    preview_limit = max(_env_int("LEROBOT_REPO_ASSIGNMENT_LOG_LIMIT", 4), 0)
    sorted_repo_ids = sorted(
        repo_ids,
        key=lambda rid: (-repo_weights_map.get(rid, 0.0), -frames_map.get(rid, 0), rid),
    )
    preview_items = [format_repo(rid) for rid in sorted_repo_ids[:preview_limit]]
    if len(sorted_repo_ids) > preview_limit:
        preview_items.append(f"+{len(sorted_repo_ids) - preview_limit} more")
    preview = "; ".join(preview_items) if preview_items else "none"

    return (
        f"[rank={rank:02d}/{world_size:02d}] {label} assignment: repos={len(repo_ids)}, "
        f"frames={total_frames:,}, episodes={total_episodes:,}"
        + (f", weight_sum={total_weight:.6f}" if repo_weights_map else "")
        + f" | groups: {group_summary}\n"
        f"[rank={rank:02d}/{world_size:02d}] top repos: {preview}"
    )


def log_repo_assignment(
    *,
    rank: int,
    world_size: int,
    repo_ids: list[str],
    frames_map: dict[str, int] | None,
    episodes_map: dict[str, int] | None,
    repo_weights_map: dict[str, float] | None,
    groups_cfg: DictConfig | None,
    label: str,
) -> None:
    message = _format_repo_assignment_for_log(
        rank=rank,
        world_size=world_size,
        repo_ids=repo_ids,
        frames_map=frames_map,
        episodes_map=episodes_map,
        repo_weights_map=repo_weights_map,
        groups_cfg=groups_cfg,
        label=label,
    )
    if message:
        print(message, flush=True)


def resolve_delta_timestamps(
    cfg: PreTrainedConfig, ds_meta: LeRobotDatasetMetadata
) -> dict[str, list] | None:
    """Resolves delta_timestamps by reading from the 'delta_indices' properties of the PreTrainedConfig.

    Args:
        cfg (PreTrainedConfig): The PreTrainedConfig to read delta_indices from.
        ds_meta (LeRobotDatasetMetadata): The dataset from which features and fps are used to build
            delta_timestamps against.

    Returns:
        dict[str, list] | None: A dictionary of delta_timestamps, e.g.:
            {
                "observation.state": [-0.04, -0.02, 0]
                "observation.action": [-0.02, 0, 0.02]
            }
            returns `None` if the resulting dict is empty.
    """
    delta_timestamps = {}
    feature_mapping = get_feature_mapping(ds_meta.robot_type, ds_meta.features)
    image_mapping = get_image_mapping(ds_meta.robot_type, ds_meta.features)
    for key in ds_meta.features:
        if key == REWARD and cfg.reward_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.reward_delta_indices]
        elif key == ACTION and cfg.action_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.action_delta_indices]
        elif key.startswith(OBS_PREFIX) and cfg.observation_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.observation_delta_indices]
        elif key in feature_mapping[ACTION] and cfg.action_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.action_delta_indices]

        if key in image_mapping.keys() and hasattr(cfg, "image_delta_indices") and cfg.image_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.image_delta_indices]

    if len(delta_timestamps) == 0:
        delta_timestamps = None

    return delta_timestamps


def _build_single_dataset(
    cfg: TrainPipelineConfig,
    repo_id: str,
    image_transforms,
    seed_offset: int, 
):
    """
    Build one dataset (single robot) including:
    - metadata
    - delta timestamps
    - LeRobotDataset (or streaming version)
    - ImageNet stats substitution
    - external stats loading (if enabled)
    - TransformedLeRobotDataset wrapping

    Returns:
        transformed_dataset,
        stats_copy,
        robot_type
    """

    # Load metadata + determine delta timestamps
    ds_meta = LeRobotDatasetMetadata(
        repo_id,
        root=cfg.dataset.root,
        revision=cfg.dataset.revision,
    )
    delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta)

    if cfg.dataset.streaming:
        root = cfg.dataset.root if cfg.dataset.root is not None else HF_LEROBOT_HOME / repo_id
        base_ds = StreamingLeRobotDataset(
            repo_id=repo_id,
            root=root,          
            episodes=cfg.dataset.episodes,
            image_transforms=image_transforms,
            delta_timestamps=delta_timestamps,
            revision=cfg.dataset.revision,
            force_cache_sync=False,         
            streaming=True,                 
            buffer_size=cfg.dataset.buffer_size,               
            max_num_shards=cfg.num_workers,
            seed=cfg.seed + seed_offset,
            rng=None,
            shuffle=True,
        )
        transformed_ds = TransformedStreamingLeRobotDataset.from_base(
            base_ds,
            cfg.dataset.data_transforms.inputs,
        )
    else:

        # Create the actual LeRobot dataset (non-streaming recommended for multi-robot)
        base_ds = LeRobotDataset(
            repo_id,
            root=cfg.dataset.root,
            episodes=cfg.dataset.episodes,
            delta_timestamps=delta_timestamps,
            # tolerance_s=0.2, 
            image_transforms=image_transforms,
            revision=cfg.dataset.revision,
            video_backend=cfg.dataset.video_backend,
        )
        transformed_ds = TransformedLeRobotDataset.from_base(
            base_ds,
            cfg.dataset.data_transforms.inputs,
        )

    # Optional: override stats using ImageNet norm
    if cfg.dataset.use_imagenet_stats:
        for key in base_ds.meta.camera_keys:
            for stats_type, stats in IMAGENET_STATS.items():
                base_ds.meta.stats[key][stats_type] = torch.tensor(
                    stats, dtype=torch.float32
                )

    robot_type = base_ds.meta.robot_type
    resolved_robot_type = infer_embodiment_variant(robot_type, base_ds.meta.features)

    # Optional: load aggregated external stats
    if cfg.dataset.use_external_stats:
        stat_candidates: list[Path] = []
        if cfg.dataset.external_stats_path is not None:
            stat_candidates.append(Path(cfg.dataset.external_stats_path))
        elif getattr(cfg.dataset, "external_stats_root", None) is not None:
            action_mode = cfg.dataset.action_mode
            stat_root = Path(cfg.dataset.external_stats_root)
            stat_candidates.append(stat_root / resolved_robot_type / action_mode / "stats.json")
            if resolved_robot_type != robot_type:
                stat_candidates.append(stat_root / robot_type / action_mode / "stats.json")
        else:
            action_mode = cfg.dataset.action_mode
            stat_candidates.append(HF_LEROBOT_HOME / f"stats/{resolved_robot_type}/{action_mode}/stats.json")
            if resolved_robot_type != robot_type:
                stat_candidates.append(HF_LEROBOT_HOME / f"stats/{robot_type}/{action_mode}/stats.json")
        # stat_path = HF_LEROBOT_HOME / f"stats/{robot_type}/{action_mode}/{repo_id}/stats.json"

        stat_path = next((path for path in stat_candidates if path.exists()), stat_candidates[0])
        if stat_path.exists():
            ext_stats = cast_stats_to_numpy(load_json(stat_path))
            logging.info(f"Using external stats from {stat_path}")
            base_ds.meta.stats.update(ext_stats)
        else:
            candidates = "\n".join(f"  - {path}" for path in stat_candidates)
            raise FileNotFoundError(
                f"use_external_stats=True but no external stats file was found. Tried:\n{candidates}"
            )

    stats_copy = base_ds.meta.stats.copy()

    return transformed_ds, stats_copy, resolved_robot_type


def make_dataset(cfg: TrainPipelineConfig) -> LeRobotDataset | StreamingLeRobotDataset | MultiLeRobotDataset | MultiStreamingLeRobotDataset:
    """Handles the logic of setting up delta timestamps and image transforms before creating a dataset.

    Args:
        cfg (TrainPipelineConfig): A TrainPipelineConfig config which contains a DatasetConfig and a PreTrainedConfig.

    Raises:
        NotImplementedError: The MultiLeRobotDataset is currently deactivated.

    Returns:
        LeRobotDataset | MultiLeRobotDataset
    """
    if isinstance(cfg.dataset, (FastWAMDatasetConfig, WSALargeDatasetConfig)):
        if isinstance(cfg.dataset, WSALargeDatasetConfig):
            from lerobot.policies.WSA_Large.dataset_wsa_large import (
                build_wsa_large_dataset,
                resolve_wsa_large_dataset_dirs,
            )

            build_policy_dataset = build_wsa_large_dataset
            policy_label = "WSA_Large"
            stats_filename = "wsa_large_dataset_stats.json"
            stats_key = "wsa_large"
        else:
            from lerobot.policies.fastwam.dataset_fastwam import build_fastwam_dataset

            build_policy_dataset = build_fastwam_dataset
            policy_label = "FastWAM"
            stats_filename = "fastwam_dataset_stats.json"
            stats_key = "fastwam"

        if cfg.dataset.dist_loading and not isinstance(cfg.dataset, WSALargeDatasetConfig):
            raise ValueError(
                f"{policy_label} training in this framework does not support `dataset.dist_loading=true`. "
                "Leave it disabled so Accelerate can shard the dataloader correctly."
            )
        if cfg.dataset.text_embedding_cache_dir is None and not getattr(cfg.policy, "load_text_encoder", False):
            raise ValueError(
                f"{policy_label} training needs either `dataset.text_embedding_cache_dir` for cached text embeddings "
                "or `policy.load_text_encoder=true` for on-the-fly prompt encoding."
            )
        if isinstance(cfg.dataset, WSALargeDatasetConfig) and float(getattr(cfg.policy, "lambda_3d", 0.0)) > 0.0:
            from lerobot.policies.WSA_Large.dataset_wsa_large import resolve_wsa_large_concat_layout

            cfg.dataset.return_future_3d_images = True
            num_views = int(getattr(cfg.policy, "da3_num_views", 0))
            num_output_cameras = int(getattr(cfg.dataset, "processor_num_output_cameras", 0))
            if num_output_cameras != num_views:
                raise ValueError(
                    "Future3DExpert needs policy.da3_num_views to match "
                    "dataset.processor_num_output_cameras."
                )
            view_layout = getattr(cfg.policy, "future_3d_view_attention_layout", "auto")
            concat_layout = resolve_wsa_large_concat_layout(
                num_output_cameras,
                getattr(cfg.dataset, "concat_multi_camera", "auto"),
            )
            if view_layout == "auto":
                if num_views <= 1:
                    view_layout = "full"
                elif num_views == 3:
                    view_layout = "robotwin"
                else:
                    view_layout = "horizontal"
            if view_layout != "full" and view_layout != concat_layout:
                raise ValueError(
                    "Future3DExpert view-aware attention layout must match dataset.concat_multi_camera: "
                    f"effective policy.future_3d_view_attention_layout={view_layout!r}, "
                    f"dataset.concat_multi_camera={concat_layout!r}."
                )
        if isinstance(cfg.dataset, WSALargeDatasetConfig):
            original_dataset_dirs = list(cfg.dataset.dataset_dirs)
            original_repo_id_file = cfg.dataset.repo_id_file
            original_dataset_sampling_weights = list(cfg.dataset.dataset_sampling_weights or [])
            all_repo_ids = resolve_wsa_large_dataset_dirs(cfg.dataset)
            repo_ids = all_repo_ids
            repo_weights_map = None
            frames_map = None
            episodes_map = None
            weight_cfg = None
            rank, world_size = get_rank_and_world_size()

            if cfg.dataset.weight_rules_path is not None:
                frames_map, episodes_map = load_info_for_repos(cfg, all_repo_ids)
                weight_cfg = OmegaConf.load(cfg.dataset.weight_rules_path)
                repo_weights_map = compute_repo_weights(
                    all_repo_ids,
                    frames_map,
                    episodes_map,
                    weight_cfg,
                )

            if cfg.dataset.dist_loading:
                if world_size <= 1:
                    raise ValueError("dist_loading is not supported when num_processes is 1")
                if frames_map is None or episodes_map is None:
                    frames_map, episodes_map = load_info_for_repos(cfg, all_repo_ids)
                try:
                    if weight_cfg is not None:
                        rank_to_repos = compute_group_balanced_repo_assignment(
                            all_repo_ids,
                            frames_map,
                            world_size,
                            weight_cfg,
                        )
                        logging.info(
                            "[make_dataset] WSA_Large dist_loading=True, using source-aware "
                            "total_frames-balanced assignment."
                        )
                    else:
                        rank_to_repos = compute_balanced_repo_assignment(
                            all_repo_ids,
                            frames_map,
                            world_size,
                        )
                        logging.info(
                            "[make_dataset] WSA_Large dist_loading=True, using "
                            "total_frames-balanced assignment."
                        )
                    repo_ids = rank_to_repos[rank]
                except Exception as e:
                    logging.warning(
                        "[make_dataset] WSA_Large total_frames-based balancing failed with error: %s. "
                        "Falling back to simple rank-based assignment.",
                        e,
                    )
                    repo_ids = assign_repo_ids_for_rank(all_repo_ids, rank, world_size)

                cfg.dataset.dataset_dirs = list(repo_ids)
                cfg.dataset.repo_id_file = None

            if repo_weights_map is not None:
                cfg.dataset.dataset_sampling_weights = [repo_weights_map[rid] for rid in repo_ids]
                logging.info(
                    "[make_dataset] WSA_Large weighted sampling enabled from %s",
                    cfg.dataset.weight_rules_path,
                )
            else:
                cfg.dataset.dataset_sampling_weights = []
            log_repo_assignment(
                rank=rank,
                world_size=world_size,
                repo_ids=repo_ids,
                frames_map=frames_map,
                episodes_map=episodes_map,
                repo_weights_map=repo_weights_map,
                groups_cfg=weight_cfg,
                label="WSA_Large",
            )

        stats_cache_path = cfg.dataset.normalization_stats_path
        if stats_cache_path is None and cfg.output_dir is not None:
            stats_cache_path = str(Path(cfg.output_dir) / stats_filename)
        dataset = build_policy_dataset(cfg.dataset, stats_cache_path=stats_cache_path)
        if isinstance(cfg.dataset, WSALargeDatasetConfig):
            cfg.dataset.dataset_dirs = original_dataset_dirs
            cfg.dataset.repo_id_file = original_repo_id_file
            cfg.dataset.dataset_sampling_weights = original_dataset_sampling_weights
        checkpoint_stats = getattr(dataset, "checkpoint_stats", None)
        if isinstance(cfg.dataset, WSALargeDatasetConfig) and checkpoint_stats is not None:
            data_stats = checkpoint_stats
            for stats_alias in ("wsa_large", *WSA_LARGE_LEGACY_ALIASES):
                if stats_alias in data_stats and WSA_LARGE not in data_stats:
                    data_stats[WSA_LARGE] = data_stats[stats_alias]
                    break
            if WSA_LARGE in data_stats and "wsa_large" not in data_stats:
                data_stats["wsa_large"] = data_stats[WSA_LARGE]
        else:
            data_stats = {stats_key: dataset.dataset_stats} if dataset.dataset_stats is not None else {}
            if isinstance(cfg.dataset, WSALargeDatasetConfig) and dataset.dataset_stats is not None:
                data_stats[WSA_LARGE] = dataset.dataset_stats
        return dataset, data_stats

    wsa_base_pipeline_image_aug = (
        isinstance(cfg.dataset, WSABaseDatasetConfig)
        and cfg.dataset.image_transforms.enable
        and cfg.dataset.image_transforms.preset in {"pi05", "pi0.5", "pi05_style"}
    )
    if wsa_base_pipeline_image_aug:
        image_transforms = None
    elif cfg.dataset.image_transforms.enable:
        image_transforms = ImageTransforms(cfg.dataset.image_transforms)
    else:
        image_transforms = None

    if isinstance(cfg.dataset, RoboChallengeRawW1DatasetConfig):
        if not is_wsa_base(cfg.policy.type):
            raise ValueError("dataset.type=robochallenge_raw_* is only supported with policy.type=WSA_Base.")
        if not cfg.dataset.raw_root:
            raise ValueError("dataset.raw_root must point to a RoboChallenge raw root for raw training.")
        if not cfg.dataset.use_external_stats or cfg.dataset.external_stats_path is None:
            raise ValueError(
                "RoboChallenge raw training requires dataset.use_external_stats=true and "
                "dataset.external_stats_path. Compute stats before training."
            )

        from lerobot.policies.WSA_Base.robochallenge_raw_dataset import (
            RoboChallengeRawAlohaDataset,
            RoboChallengeRawW1Dataset,
            get_robochallenge_raw_spec,
            resolve_robochallenge_raw_task_names,
            resolve_robochallenge_raw_task_weights,
        )

        raw_spec = get_robochallenge_raw_spec(cfg.dataset.embodiment, cfg.dataset.task_preset)
        task_names = resolve_robochallenge_raw_task_names(cfg.dataset.task_preset)
        if (
            cfg.dataset.weighted_task_sampling
            and cfg.dataset.task_sampling_mode == "group_frames_pow"
            and task_names is None
        ):
            raise ValueError(
                f"RoboChallenge {raw_spec.robot_type} group_frames_pow sampling requires a table30v2 task_preset."
            )
        task_sampling_weights = (
            resolve_robochallenge_raw_task_weights(
                cfg.dataset.task_preset,
                regular_task_weight=cfg.dataset.regular_task_weight,
                extra_task_weight=cfg.dataset.extra_task_weight,
            )
            if cfg.dataset.weighted_task_sampling and cfg.dataset.task_sampling_mode == "per_task"
            else None
        )
        task_sampling_mode = cfg.dataset.task_sampling_mode if cfg.dataset.weighted_task_sampling else "none"
        raw_dataset_cls = RoboChallengeRawAlohaDataset if raw_spec.key == "aloha" else RoboChallengeRawW1Dataset

        dataset = raw_dataset_cls(
            raw_root=cfg.dataset.raw_root,
            external_stats_path=cfg.dataset.external_stats_path,
            transforms=cfg.dataset.data_transforms.inputs,
            chunk_size=int(cfg.policy.chunk_size),
            image_delta_indices=getattr(cfg.policy, "image_delta_indices", None),
            image_transforms=image_transforms,
            embodiment=raw_spec.robot_type,
            frame_interval=cfg.dataset.frame_interval,
            task_regex=cfg.dataset.task_regex,
            task_names=task_names,
            task_sampling_weights=task_sampling_weights,
            task_sampling_mode=task_sampling_mode,
            task_sampling_gamma=cfg.dataset.task_sampling_gamma,
            regular_task_total_weight=cfg.dataset.regular_task_total_weight,
            extra_task_total_weight=cfg.dataset.extra_task_total_weight,
            state_cache_dir=cfg.dataset.state_cache_dir,
            state_cache_size=cfg.dataset.state_cache_size,
            validate_videos=cfg.dataset.validate_videos,
        )
        data_stats = {dataset.meta.robot_type: dataset.meta.stats}
        return dataset, data_stats

    all_data_stats = {}
    all_repo_ids = resolve_repo_ids(cfg)
    logging.info(
        f"[make_dataset] all_repo_ids={all_repo_ids}"
    )

    frames_map, episodes_map = load_info_for_repos(cfg, all_repo_ids)
    if cfg.dataset.weight_rules_path is not None:
        weight_cfg = OmegaConf.load(cfg.dataset.weight_rules_path)
        repo_weights_map = compute_repo_weights(
            all_repo_ids,
            frames_map,
            episodes_map,
            weight_cfg,
        )
    else:
        repo_weights_map = None
    
    rank, world_size = get_rank_and_world_size()
    if cfg.dataset.dist_loading:
        # Try to balance by total_frames first.
        try:
            if cfg.dataset.weight_rules_path is not None:
                rank_to_repos = compute_group_balanced_repo_assignment(
                    all_repo_ids,
                    frames_map,
                    world_size,
                    weight_cfg,
                )
                logging.info(
                    "[make_dataset] dist_loading=True, using source-aware total_frames-balanced assignment."
                )
            else:
                rank_to_repos = compute_balanced_repo_assignment(
                    all_repo_ids,
                    frames_map,
                    world_size,
                )
                logging.info(
                    f"[make_dataset] dist_loading=True, using total_frames-balanced "
                    f"assignment."
                )
            repo_ids = rank_to_repos[rank]
        except Exception as e:
            # Fallback to the simple deterministic assignment
            logging.warning(
                f"[make_dataset] total_frames-based balancing failed with error: {e}. "
                "Falling back to simple rank-based assignment."
            )
            repo_ids = assign_repo_ids_for_rank(all_repo_ids, rank, world_size)
    else:
        repo_ids = all_repo_ids

    if repo_weights_map is not None:
        print(
            f"[rank={rank:02d}/{world_size:02d}], repo_ids_for_this_rank:\n"
            + "\n".join(
                f"[rank {rank}] repo_id = {rid}, weight = {repo_weights_map[rid]:.6f}"
                for rid in repo_ids
            )
        )
    else:
        print(
            f"[rank={rank:02d}/{world_size:02d}], repo_ids_for_this_rank:\n"
            + "\n".join(f"[rank {rank}] repo_id = {rid}" for rid in repo_ids)
        )

    if len(repo_ids) == 1:
        repo_id = repo_ids[0]

        transformed_ds, stats_copy, robot_type = _build_single_dataset(
            cfg,
            repo_id,
            image_transforms,
            rank,
        )
        all_data_stats[robot_type] = stats_copy

        return transformed_ds, all_data_stats
    
    transformed_datasets = []

    for rid, repo_id in enumerate(repo_ids):
        transformed_ds, stats_copy, robot_type = _build_single_dataset(
            cfg,
            repo_id,
            image_transforms,
            rank * 128 + rid,
        )
        transformed_datasets.append(transformed_ds)
        all_data_stats[robot_type] = stats_copy  # TODO: If multiple repos share robot_type, last one overwrites.

    dataset_weights = [
            repo_weights_map[ds.repo_id]
            for ds in transformed_datasets
        ] if repo_weights_map is not None else None
    
    if not cfg.dataset.streaming:
        multi_ds = MultiLeRobotDataset(transformed_datasets, dataset_weights=dataset_weights)
    else:
        multi_ds = MultiStreamingLeRobotDataset(transformed_datasets, dataset_weights=dataset_weights, seed=cfg.seed)

    return multi_ds, all_data_stats
