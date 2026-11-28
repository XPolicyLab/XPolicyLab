#!/usr/bin/env python

import argparse
import hashlib
import json
import multiprocessing as mp
from pathlib import Path

import tqdm
import torch
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.transforms.constants import get_feature_mapping, get_mask_mapping, infer_embodiment_variant
from lerobot.utils.constants import OBS_STATE, ACTION, HF_LEROBOT_HOME
from lerobot.datasets.utils import write_json


def parse_args():
    p = argparse.ArgumentParser(
        description="Compute (and aggregate) normalization statistics for LeRobot datasets"
    )

    p.add_argument(
        "--action_mode",
        type=str,
        choices=["abs", "delta"],
        required=True,
        help="Action mode used to compute statistics (abs or delta).",
    )
    p.add_argument(
        "--chunk_size",
        type=int,
        required=True,
        help="Chunk size used for delta action computation (episodes shorter than chunk_size are skipped).",
    )
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--repo_ids",
        type=str,
        nargs="+",
        help="One or more LeRobotDataset repo ids (must share the same resolved robot_type and feature schema).",
    )
    source.add_argument(
        "--repo_id_file",
        type=str,
        default=None,
        help="Text file with one LeRobotDataset repo id or local path per line.",
    )
    p.add_argument(
        "--root",
        type=str,
        default=None,
        help="Optional LeRobot dataset root. Useful when repo_ids are relative dataset names.",
    )
    p.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of worker processes (repo-level parallelism).",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional output root directory. If not set, uses HF_LEROBOT_HOME/stats/...",
    )
    p.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Optional exact output path for stats.json. Overrides --output_dir.",
    )
    p.add_argument(
        "--max_chunks_per_episode",
        type=int,
        default=None,
        help=(
            "Optional approximate mode: sample at most this many delta chunks per episode. "
            "All repos are still visited; only per-episode windows are subsampled."
        ),
    )
    p.add_argument(
        "--max_chunks_per_repo",
        type=int,
        default=None,
        help=(
            "Optional approximate mode: after per-episode sampling, cap total sampled delta chunks per repo. "
            "Use with --max_chunks_per_episode to bound very large repos."
        ),
    )
    p.add_argument(
        "--sample_seed",
        type=int,
        default=42,
        help="Seed for deterministic sampled stats mode.",
    )
    p.add_argument(
        "--skip_action_robot_types",
        nargs="*",
        default=[],
        help=(
            "Resolved robot types for which action values should not be scanned. "
            "Zero action stats are still written so action-conditioned datasets can load safely."
        ),
    )
    p.add_argument(
        "--zero_stats_robot_types",
        nargs="*",
        default=[],
        help=(
            "Resolved robot types whose vector stats are known to be all zeros. "
            "These groups are handled from metadata only, without initializing LeRobotDataset."
        ),
    )

    return p.parse_args()


def _repo_root_path(repo_id: str, root: str | None) -> Path:
    repo_path = Path(repo_id)
    if repo_path.is_absolute():
        return repo_path
    return Path(root) / repo_id if root else HF_LEROBOT_HOME / repo_id


def _load_repo_info(repo_id: str, root: str | None) -> dict:
    info_path = _repo_root_path(repo_id, root) / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing info.json for metadata-only stats: {info_path}")
    return json.loads(info_path.read_text(encoding="utf-8"))


def _load_repo_meta_stats(repo_id: str, root: str | None) -> dict:
    stats_path = _repo_root_path(repo_id, root) / "meta" / "stats.json"
    if not stats_path.is_file():
        return {}
    return json.loads(stats_path.read_text(encoding="utf-8"))


def resolve_dataset_entry(repo_id: str, root: str | None) -> tuple[LeRobotDataset, str]:
    repo_path = Path(repo_id)
    if repo_path.is_absolute():
        dataset = LeRobotDataset(str(repo_path))
        dataset_name = repo_path.name
        return dataset, dataset_name

    dataset = LeRobotDataset(repo_id, root=root)
    dataset_name = Path(dataset.root).name
    return dataset, dataset_name


class RunningStats:
    """Running stats for vectors: keeps count, mean, mean_sq, min, max."""

    def __init__(self):
        self._count = 0
        self._mean = None
        self._mean_of_squares = None
        self._min = None
        self._max = None

    def update(self, batch: torch.Tensor) -> None:
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
            return

        total = self._count + count
        w_old = self._count / total
        w_new = count / total

        self._mean = w_old * self._mean + w_new * mean
        self._mean_of_squares = w_old * self._mean_of_squares + w_new * mean_sq
        self._min = torch.minimum(self._min, min_)
        self._max = torch.maximum(self._max, max_)
        self._count = total

    def merge(self, other: "RunningStats") -> None:
        """Merge another RunningStats (exact for mean/mean_sq/min/max)."""
        if other._count == 0:
            return
        if self._count == 0:
            self._count = other._count
            self._mean = other._mean.clone()
            self._mean_of_squares = other._mean_of_squares.clone()
            self._min = other._min.clone()
            self._max = other._max.clone()
            return

        total = self._count + other._count
        w_self = self._count / total
        w_other = other._count / total

        self._mean = w_self * self._mean + w_other * other._mean
        self._mean_of_squares = w_self * self._mean_of_squares + w_other * other._mean_of_squares
        self._min = torch.minimum(self._min, other._min)
        self._max = torch.maximum(self._max, other._max)
        self._count = total

    def to_payload(self) -> dict:
        """Serialize to a JSON-friendly dict."""
        if self._count == 0:
            # Keep empty stats explicit
            return {
                "count": 0,
                "mean": None,
                "mean_sq": None,
                "min": None,
                "max": None,
            }
        return {
            "count": int(self._count),
            "mean": self._mean.detach().cpu().tolist(),
            "mean_sq": self._mean_of_squares.detach().cpu().tolist(),
            "min": self._min.detach().cpu().tolist(),
            "max": self._max.detach().cpu().tolist(),
        }

    @staticmethod
    def from_payload(p: dict) -> "RunningStats":
        rs = RunningStats()
        if p["count"] == 0:
            return rs
        rs._count = int(p["count"])
        rs._mean = torch.tensor(p["mean"], dtype=torch.float32)
        rs._mean_of_squares = torch.tensor(p["mean_sq"], dtype=torch.float32)
        rs._min = torch.tensor(p["min"], dtype=torch.float32)
        rs._max = torch.tensor(p["max"], dtype=torch.float32)
        return rs

    def get_statistics(self) -> dict:
        """Return mean, std, min, max, count."""
        if self._count == 0:
            raise ValueError("No data has been added yet.")
        var = self._mean_of_squares - self._mean ** 2
        std = torch.sqrt(torch.clamp(var, min=0.0))
        return {
            "min": self._min.tolist(),
            "max": self._max.tolist(),
            "mean": self._mean.tolist(),
            "std": std.tolist(),
            "count": [int(self._count)],
        }


def _stable_seed(repo_id: str, sample_seed: int) -> int:
    digest = hashlib.sha1(f"{sample_seed}|{repo_id}".encode("utf-8")).hexdigest()[:8]
    return int(digest, 16)


def _stack_selected_column(dataset: LeRobotDataset, key: str, indices: np.ndarray) -> torch.Tensor:
    selected = dataset.hf_dataset.select(indices)
    return torch.stack(selected[key][:])


def _as_2d_sequence(tensor: torch.Tensor) -> torch.Tensor:
    return tensor if tensor.ndim > 1 else tensor[:, None]


def _feature_dim(feature: dict) -> int:
    shape = feature.get("shape", ())
    if isinstance(shape, int):
        return int(shape)
    if not shape:
        return 1
    return int(shape[0])


def _zero_stats_payload(dim: int, count: int) -> dict:
    zeros = [0.0] * int(dim)
    return {
        "count": int(max(count, 1)),
        "mean": zeros,
        "mean_sq": zeros,
        "min": zeros,
        "max": zeros,
    }


def _zero_statistics(dim: int, count: int) -> dict:
    zeros = [0.0] * int(dim)
    return {
        "min": zeros,
        "max": zeros,
        "mean": zeros,
        "std": zeros,
        "count": [int(max(count, 1))],
    }


def _make_group_name(repo_ids: list[str]) -> str:
    """Short stable name for a repo set."""
    joined = "|".join(repo_ids)
    h = hashlib.sha1(joined.encode("utf-8")).hexdigest()[:10]
    return f"agg_{len(repo_ids)}repos_{h}"


def _resolve_output_path(cfg, repo_ids: list[str], resolved_robot_type: str, action_mode: str) -> tuple[Path, Path, str]:
    group_name = _make_group_name(repo_ids)
    if cfg.output_path:
        output_path = Path(cfg.output_path)
        output_dir = output_path.parent
    elif cfg.output_dir:
        output_dir = Path(cfg.output_dir) / group_name
        output_path = output_dir / "stats.json"
    else:
        out_root = HF_LEROBOT_HOME / "stats"
        output_dir = out_root / resolved_robot_type / action_mode / group_name
        output_path = output_dir / "stats.json"
    return output_path, output_dir, group_name


def _compute_zero_group_stats(repo_ids: list[str], action_mode: str, chunk_size: int, root: str | None) -> dict:
    infos = [_load_repo_info(rid, root) for rid in repo_ids]
    robot_type = infos[0]["robot_type"]
    resolved_robot_type = infer_embodiment_variant(robot_type, infos[0].get("features", {}))
    features = infos[0].get("features", {})
    mapping = get_feature_mapping(robot_type, features)

    resolved_types = {
        infer_embodiment_variant(info["robot_type"], info.get("features", {}))
        for info in infos
    }
    if len(resolved_types) != 1:
        raise ValueError(f"repo_ids must share the same resolved robot_type, got: {sorted(resolved_types)}")

    stat_keys = [k for k in mapping[OBS_STATE] + mapping[ACTION] if k in features]
    total_frames = sum(int(info.get("total_frames", 0)) for info in infos)
    total_episodes = sum(int(info.get("total_episodes", 0)) for info in infos)
    count = total_frames
    output_dict = {
        key: _zero_statistics(_feature_dim(features[key]), count)
        for key in stat_keys
    }

    first_meta_stats = _load_repo_meta_stats(repo_ids[0], root)
    for key, feature in features.items():
        if feature.get("dtype") in {"image", "video"} and key in first_meta_stats:
            output_dict[key] = first_meta_stats[key]

    return {
        "robot_type": robot_type,
        "resolved_robot_type": resolved_robot_type,
        "output_dict": output_dict,
        "total_frames": total_frames,
        "total_episodes": total_episodes,
        "chunk_size": chunk_size,
        "action_mode": action_mode,
    }


def _sample_episode_starts(
    from_ids: np.ndarray,
    to_ids: np.ndarray,
    chunk_size: int,
    max_chunks_per_episode: int | None,
    max_chunks_per_repo: int | None,
    rng: np.random.Generator,
) -> dict[int, np.ndarray]:
    starts_by_episode: dict[int, np.ndarray] = {}
    sampled_pairs: list[tuple[int, int]] = []

    for ep_idx, (from_idx, to_idx) in enumerate(zip(from_ids, to_ids)):
        ep_len = int(to_idx - from_idx)
        num_starts = ep_len - chunk_size + 1
        if num_starts <= 0:
            continue

        if max_chunks_per_episode is None or num_starts <= max_chunks_per_episode:
            starts = np.arange(num_starts, dtype=np.int64)
        else:
            starts = np.sort(
                rng.choice(num_starts, size=int(max_chunks_per_episode), replace=False).astype(np.int64)
            )

        starts_by_episode[ep_idx] = starts
        if max_chunks_per_repo is not None:
            sampled_pairs.extend((ep_idx, int(start)) for start in starts)

    if max_chunks_per_repo is None or len(sampled_pairs) <= max_chunks_per_repo:
        return starts_by_episode

    chosen = rng.choice(len(sampled_pairs), size=int(max_chunks_per_repo), replace=False)
    capped: dict[int, list[int]] = {}
    for idx in chosen:
        ep_idx, start = sampled_pairs[int(idx)]
        capped.setdefault(ep_idx, []).append(start)
    return {
        ep_idx: np.asarray(sorted(starts), dtype=np.int64)
        for ep_idx, starts in capped.items()
    }


def _update_sampled_delta_episode(
    dataset: LeRobotDataset,
    stats: dict[str, RunningStats],
    keys: list[str],
    mapping: dict,
    mask: torch.Tensor,
    from_idx: int,
    starts: np.ndarray,
    chunk_size: int,
    action_mode: str,
    skip_action_stats: bool,
) -> int:
    abs_starts = from_idx + starts
    action_keys = set(mapping[ACTION])

    for key in keys:
        if skip_action_stats and key in action_keys:
            continue
        if action_mode == "abs" or key not in action_keys:
            val = _stack_selected_column(dataset, key, abs_starts)
            stats[key].update(val)

    if action_mode != "delta" or skip_action_stats:
        return int(len(starts))

    action_rel = np.unique((starts[:, None] + np.arange(chunk_size, dtype=np.int64)[None, :]).reshape(-1))
    action_abs = from_idx + action_rel
    rel_to_pos = {int(rel): pos for pos, rel in enumerate(action_rel.tolist())}
    gather = torch.as_tensor(
        [[rel_to_pos[int(start + offset)] for offset in range(chunk_size)] for start in starts.tolist()],
        dtype=torch.long,
    )

    action = [_as_2d_sequence(_stack_selected_column(dataset, k, action_abs)) for k in mapping[ACTION]]
    action = torch.cat(action, dim=-1)
    action_chunk = action[gather]

    state = [_as_2d_sequence(_stack_selected_column(dataset, k, abs_starts)) for k in mapping[OBS_STATE]]
    state = torch.cat(state, dim=-1)
    delta_action = action_chunk - torch.where(mask, state, 0)[:, None]

    sid, eid = 0, 0
    for action_key in mapping[ACTION]:
        eid += dataset.meta.features[action_key]["shape"][0]
        stats[action_key].update(delta_action[..., sid:eid])
        sid = eid

    return int(len(starts))


def _compute_one_repo(
    repo_id: str,
    action_mode: str,
    chunk_size: int,
    root: str | None,
    max_chunks_per_episode: int | None,
    max_chunks_per_repo: int | None,
    sample_seed: int,
    skip_action_robot_types: tuple[str, ...],
) -> dict:
    """Worker: compute stats for one repo, return serializable payload."""
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    dataset, dataset_name = resolve_dataset_entry(repo_id, root)
    robot_type = dataset.meta.robot_type
    resolved_robot_type = infer_embodiment_variant(robot_type, dataset.meta.features)
    skip_action_stats = resolved_robot_type in skip_action_robot_types or robot_type in skip_action_robot_types

    mask = get_mask_mapping(robot_type, dataset.meta.features)
    mapping = get_feature_mapping(robot_type, dataset.meta.features)
    action_keys = set(mapping[ACTION])

    keys = list(dataset.meta.features.keys())
    for k in dataset.meta.video_keys + dataset.meta.image_keys:
        if k in keys:
            keys.remove(k)

    # Capture schema for consistency checks
    shapes = {k: dataset.meta.features[k]["shape"] for k in keys}

    stats = {k: RunningStats() for k in keys}
    total_frames = 0
    skipped_episodes = 0
    sampled_chunks = 0

    from_ids = np.asarray(dataset.meta.episodes["dataset_from_index"])
    to_ids = np.asarray(dataset.meta.episodes["dataset_to_index"])
    total_episodes = dataset.num_episodes
    sampled_mode = max_chunks_per_episode is not None or max_chunks_per_repo is not None

    starts_by_episode = None
    if sampled_mode:
        rng = np.random.default_rng(_stable_seed(repo_id, sample_seed))
        starts_by_episode = _sample_episode_starts(
            from_ids=from_ids,
            to_ids=to_ids,
            chunk_size=chunk_size,
            max_chunks_per_episode=max_chunks_per_episode,
            max_chunks_per_repo=max_chunks_per_repo,
            rng=rng,
        )

    for ep_idx, (from_idx, to_idx) in enumerate(zip(from_ids, to_ids)):
        ep_len = int(to_idx - from_idx)
        total_frames += ep_len

        if ep_len < chunk_size:
            skipped_episodes += 1
            continue

        if sampled_mode:
            starts = starts_by_episode.get(ep_idx, np.empty(0, dtype=np.int64))
            if starts.size == 0:
                continue
            sampled_chunks += _update_sampled_delta_episode(
                dataset=dataset,
                stats=stats,
                keys=keys,
                mapping=mapping,
                mask=mask,
                from_idx=int(from_idx),
                starts=starts,
                chunk_size=chunk_size,
                action_mode=action_mode,
                skip_action_stats=skip_action_stats,
            )
            continue

        curr_episode = dataset.hf_dataset.select(np.arange(from_idx, to_idx))

        # Non-action stats always update; action stats depend on mode
        for key in keys:
            if skip_action_stats and key in action_keys:
                continue
            if action_mode == "abs" or key not in action_keys:
                val = torch.stack(curr_episode[key][:])
                stats[key].update(val)

        if action_mode == "delta" and not skip_action_stats:
            action = [torch.stack(curr_episode[k][:]) for k in mapping[ACTION]]
            action = [a if a.ndim > 1 else a[:, None] for a in action]
            action = torch.cat(action, dim=-1)

            state = [torch.stack(curr_episode[k][:]) for k in mapping[OBS_STATE]]
            state = [s if s.ndim > 1 else s[:, None] for s in state]
            state = torch.cat(state, dim=-1)

            truncated_state = state[0 : (ep_len - chunk_size + 1)]
            action_chunk = action.unfold(dimension=0, size=chunk_size, step=1).permute(0, 2, 1)
            delta_action = action_chunk - torch.where(mask, truncated_state, 0)[:, None]

            sid, eid = 0, 0
            for action_key in mapping[ACTION]:
                eid += dataset.meta.features[action_key]["shape"][0]
                stats[action_key].update(delta_action[..., sid:eid])
                sid = eid

    if sampled_mode:
        selected_chunks = sum(len(starts) for starts in starts_by_episode.values()) if starts_by_episode else 0
    else:
        selected_chunks = sum(max(0, int(to_idx - from_idx) - chunk_size + 1) for from_idx, to_idx in zip(from_ids, to_ids))
    skipped_action_count = selected_chunks * chunk_size if action_mode == "delta" else total_frames
    payload = {}
    for k in keys:
        if skip_action_stats and k in action_keys:
            payload[k] = _zero_stats_payload(_feature_dim(dataset.meta.features[k]), skipped_action_count)
        else:
            payload[k] = stats[k].to_payload()

    return {
        "repo_id": repo_id,
        "dataset_name": dataset_name,
        "robot_type": robot_type,
        "resolved_robot_type": resolved_robot_type,
        "keys": keys,
        "shapes": shapes,
        "payload": payload,
        "total_frames": int(total_frames),
        "skipped_episodes": int(skipped_episodes),
        "total_episodes": int(total_episodes),
        "sampled_chunks": int(sampled_chunks),
        "sampled_mode": bool(sampled_mode),
        "skip_action_stats": bool(skip_action_stats),
    }

def _normalize_visual_stats(visual_stats: dict) -> dict:
    """Convert numpy/torch entries to lists."""
    out = {}
    for k, v in visual_stats.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, torch.Tensor):
            out[k] = v.detach().cpu().numpy().tolist()
        else:
            out[k] = v
    return out


def compute_norm_stats_multi(cfg):
    if cfg.max_chunks_per_episode is not None and cfg.max_chunks_per_episode <= 0:
        raise ValueError("--max_chunks_per_episode must be positive when set.")
    if cfg.max_chunks_per_repo is not None and cfg.max_chunks_per_repo <= 0:
        raise ValueError("--max_chunks_per_repo must be positive when set.")
    sampled_mode = cfg.max_chunks_per_episode is not None or cfg.max_chunks_per_repo is not None
    if sampled_mode and cfg.action_mode == "abs":
        print(
            "Sampled stats mode enabled for abs actions: non-action and action stats will be "
            "estimated from sampled frames instead of full episodes."
        )

    if cfg.repo_id_file:
        repo_id_file = Path(cfg.repo_id_file)
        repo_ids = [line.strip() for line in repo_id_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not repo_ids:
            raise ValueError(f"repo_id_file is empty: {repo_id_file}")
    else:
        repo_ids = cfg.repo_ids
    action_mode = cfg.action_mode
    chunk_size = cfg.chunk_size

    print(f"---------- aggregate stats for {len(repo_ids)} datasets ----------")
    first_info = _load_repo_info(repo_ids[0], cfg.root)
    first_robot_type = first_info["robot_type"]
    first_resolved_robot_type = infer_embodiment_variant(first_robot_type, first_info.get("features", {}))
    zero_stats_types = set(cfg.zero_stats_robot_types or [])
    if first_resolved_robot_type in zero_stats_types or first_robot_type in zero_stats_types:
        print(
            "Metadata-only zero stats mode enabled for "
            f"robot_type={first_robot_type}, resolved={first_resolved_robot_type}"
        )
        zero_result = _compute_zero_group_stats(
            repo_ids=repo_ids,
            action_mode=action_mode,
            chunk_size=chunk_size,
            root=cfg.root,
        )
        output_path, output_dir, group_name = _resolve_output_path(
            cfg=cfg,
            repo_ids=repo_ids,
            resolved_robot_type=zero_result["resolved_robot_type"],
            action_mode=action_mode,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(zero_result["output_dict"], output_path)

        print("---------- done ----------")
        print(f"robot_type: {zero_result['robot_type']}")
        if zero_result["resolved_robot_type"] != zero_result["robot_type"]:
            print(f"resolved_robot_type: {zero_result['resolved_robot_type']}")
        print(f"action_mode: {action_mode}")
        print(f"chunk_size: {chunk_size}")
        print(f"group_name: {group_name}")
        print(f"output: {output_path}")
        print(f"total_frames (metadata): {zero_result['total_frames']}")
        print(f"total_episodes (metadata): {zero_result['total_episodes']}")
        return

    if sampled_mode:
        print(
            "Sampled stats mode: "
            f"max_chunks_per_episode={cfg.max_chunks_per_episode}, "
            f"max_chunks_per_repo={cfg.max_chunks_per_repo}, "
            f"sample_seed={cfg.sample_seed}"
        )
    for rid in repo_ids:
        print(f"  - {rid}")

    # Repo-level parallelism
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=cfg.num_workers) as pool:
        results = list(
            tqdm.tqdm(
                pool.starmap(
                    _compute_one_repo,
                    [
                        (
                            rid,
                            action_mode,
                            chunk_size,
                            cfg.root,
                            cfg.max_chunks_per_episode,
                            cfg.max_chunks_per_repo,
                            cfg.sample_seed,
                            tuple(cfg.skip_action_robot_types),
                        )
                        for rid in repo_ids
                    ],
                ),
                total=len(repo_ids),
                desc="Computing per-repo stats",
            )
        )

    # Consistency checks
    robot_types = {r["robot_type"] for r in results}
    resolved_robot_types = {r["resolved_robot_type"] for r in results}
    if len(resolved_robot_types) != 1:
        raise ValueError(f"repo_ids must share the same resolved robot_type, got: {sorted(resolved_robot_types)}")
    robot_type = results[0]["robot_type"]
    resolved_robot_type = results[0]["resolved_robot_type"]

    keys0 = results[0]["keys"]
    shapes0 = results[0]["shapes"]
    for r in results[1:]:
        if r["keys"] != keys0:
            raise ValueError(f"Feature keys mismatch between repos: {results[0]['repo_id']} vs {r['repo_id']}")
        if r["shapes"] != shapes0:
            raise ValueError(f"Feature shapes mismatch between repos: {results[0]['repo_id']} vs {r['repo_id']}")

    # Merge numeric stats
    global_stats = {k: RunningStats() for k in keys0}
    total_frames = 0
    total_episodes = 0
    skipped_episodes = 0
    sampled_chunks = 0

    for r in results:
        total_frames += r["total_frames"]
        total_episodes += r["total_episodes"]
        skipped_episodes += r["skipped_episodes"]
        sampled_chunks += int(r.get("sampled_chunks", 0))
        for k in keys0:
            tmp = RunningStats.from_payload(r["payload"][k])
            global_stats[k].merge(tmp)

    output_dict = {k: global_stats[k].get_statistics() for k in keys0}

    # Visual stats: take from the first repo for simplicity
    first_ds, _ = resolve_dataset_entry(repo_ids[0], cfg.root)
    for k in first_ds.meta.video_keys + first_ds.meta.image_keys:
        output_dict[k] = _normalize_visual_stats(first_ds.meta.stats[k])

    # Output path
    output_path, output_dir, group_name = _resolve_output_path(
        cfg=cfg,
        repo_ids=repo_ids,
        resolved_robot_type=resolved_robot_type,
        action_mode=action_mode,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dict, output_path)

    print("---------- done ----------")
    print(f"robot_type: {robot_type}")
    if len(robot_types) > 1 or resolved_robot_type != robot_type:
        print(f"resolved_robot_type: {resolved_robot_type}")
    print(f"action_mode: {action_mode}")
    print(f"chunk_size: {chunk_size}")
    print(f"group_name: {group_name}")
    print(f"output: {output_path}")
    print(f"total_frames (sum of episode lengths): {total_frames}")
    print(f"total_episodes: {total_episodes} (skipped: {skipped_episodes} episodes with len < chunk_size)")
    if sampled_mode:
        print(f"sampled_chunks: {sampled_chunks}")


if __name__ == "__main__":
    compute_norm_stats_multi(parse_args())
