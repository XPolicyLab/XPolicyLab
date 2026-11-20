from __future__ import annotations

import os
import time
import traceback
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torchvision.transforms as tv_transforms
import torchvision.transforms.functional as transforms_f
from torch.utils.data import Dataset

from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.transforms import ImageTransforms
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.transforms.constants import (
    get_feature_mapping,
    get_image_mapping,
    get_mask_mapping,
    infer_embodiment_variant,
)
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE, SAMPLE_ACTION_LOSS_MASK

from .configuration_wsa_large import WSALargeDatasetConfig
from .core.data.dataset_utils import Normalize
from .core.data.lerobot.processors.base_processor import BaseProcessor
from .core.data.lerobot.processors.wsa_large_processor import WSALargeProcessor
from .core.data.lerobot.transforms.action_state_merger import ConcatLeftAlign
from .core.data.lerobot.transforms.image import ToTensor
from .core.data.lerobot.utils.normalizer import (
    canonicalize_norm_mode,
    load_dataset_stats_from_json,
    save_dataset_stats_to_json,
)
from .core.utils.logging_config import get_logger
from .stats_adapter import ensure_wsa_large_stats_format
from .text_cache import DEFAULT_PROMPT, build_text_embedding_cache_path

logger = get_logger(__name__)

MAX_GETITEM_ATTEMPT = 5
STANDARD_VIDEO_SIZE_BY_NUM_CAMERAS = {
    1: (224, 224),
    2: (224, 448),
    3: (384, 320),
}
STANDARD_CONCAT_LAYOUT_BY_NUM_CAMERAS = {
    1: "single",
    2: "horizontal",
    3: "robotwin",
}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _short_path_for_log(path: str | Path, max_parts: int = 4) -> str:
    parts = Path(path).parts
    if not parts:
        return str(path)
    return "/".join(parts[-max_parts:])


def resolve_wsa_large_concat_layout(num_cameras: int, concat_multi_camera: str) -> str:
    if num_cameras <= 0:
        raise ValueError(f"num_cameras must be positive, got {num_cameras}.")
    if num_cameras == 1:
        return "single"
    if concat_multi_camera == "auto":
        return STANDARD_CONCAT_LAYOUT_BY_NUM_CAMERAS.get(num_cameras, "horizontal")
    return concat_multi_camera


def resolve_wsa_large_video_size(
    num_cameras: int,
    requested_video_size: tuple[int, int],
    standardize_video_size_by_cameras: bool = True,
) -> tuple[int, int]:
    if standardize_video_size_by_cameras and num_cameras in STANDARD_VIDEO_SIZE_BY_NUM_CAMERAS:
        return STANDARD_VIDEO_SIZE_BY_NUM_CAMERAS[num_cameras]
    if len(requested_video_size) != 2:
        raise ValueError(f"video_size must be (height, width), got {requested_video_size}.")
    height, width = (int(value) for value in requested_video_size)
    if height <= 0 or width <= 0:
        raise ValueError(f"video_size values must be positive, got {requested_video_size}.")
    return height, width


def _canonical_image_key(mapped_key: str) -> str:
    prefix = f"{OBS_IMAGES}."
    if mapped_key.startswith(prefix):
        return mapped_key[len(prefix) :]
    return mapped_key.rsplit(".", 1)[-1]


def _as_feature_dim(shape: Any) -> int:
    if isinstance(shape, (list, tuple)):
        if not shape:
            return 1
        return int(shape[0])
    return int(shape)


def _as_torch_stats_value(value: Any, like: torch.Tensor) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=like.device, dtype=like.dtype)
    return torch.as_tensor(value, device=like.device, dtype=like.dtype)


def _select_stat(stats: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in stats:
            return stats[name]
    raise KeyError(f"Missing any of stats keys {names}; available={list(stats.keys())}")


def _normalize_with_stats(x: torch.Tensor, stats: dict[str, Any], mode: str) -> torch.Tensor:
    mode = canonicalize_norm_mode(mode)
    eps = 1e-6
    if mode in {"z-score", "mean_std"}:
        mean = _as_torch_stats_value(_select_stat(stats, "mean", "global_mean"), x)
        std = _as_torch_stats_value(_select_stat(stats, "std", "global_std"), x)
        return (x - mean) / (std + eps)

    if mode == "min/max":
        low = _as_torch_stats_value(_select_stat(stats, "min", "global_min"), x)
        high = _as_torch_stats_value(_select_stat(stats, "max", "global_max"), x)
    elif mode == "q01/q99":
        low = _as_torch_stats_value(_select_stat(stats, "q01", "global_q01"), x)
        high = _as_torch_stats_value(_select_stat(stats, "q99", "global_q99"), x)
    else:
        low_value, high_value = map(float, mode.split("/"))
        low = torch.full_like(x[..., :1], low_value)
        high = torch.full_like(x[..., :1], high_value)

    y = (x - low) / (high - low + eps)
    y = y * 2.0 - 1.0
    return y


class MaskedDeltaActionTransform:
    """Convert absolute action targets to deltas from the first observed state."""

    def __init__(self, delta_action_dim_mask: dict[str, list[bool]] | None):
        self.delta_action_dim_mask = delta_action_dim_mask

    def _mask_for(self, key: str, action: torch.Tensor) -> torch.Tensor:
        if self.delta_action_dim_mask is None:
            return torch.ones(action.shape[-1], dtype=torch.bool, device=action.device)
        if key not in self.delta_action_dim_mask:
            raise KeyError(f"Missing delta action mask for key {key!r}.")
        dim_mask = torch.as_tensor(self.delta_action_dim_mask[key], dtype=torch.bool, device=action.device)
        if dim_mask.numel() != action.shape[-1]:
            raise ValueError(
                f"Delta action mask for key {key!r} has {dim_mask.numel()} dims, "
                f"expected {action.shape[-1]}."
            )
        return dim_mask

    def _apply(self, batch: dict[str, Any], sign: float) -> dict[str, Any]:
        if "action" not in batch:
            return batch
        for key, action in batch["action"].items():
            if key not in batch["state"]:
                raise KeyError(f"Delta action transform requires matching state key {key!r}.")
            state = batch["state"][key]
            if action.shape[-1] != state.shape[-1]:
                raise ValueError(
                    f"Delta action transform expects action/state dim match for key {key!r}, "
                    f"got action={action.shape[-1]} and state={state.shape[-1]}."
                )
            dim_mask = self._mask_for(key, action)
            base_state = state[..., :1, :].to(device=action.device, dtype=action.dtype)
            delta_action = action.clone()
            delta_action[..., dim_mask] = delta_action[..., dim_mask] + sign * base_state[..., dim_mask]
            batch["action"][key] = delta_action
        return batch

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        return self._apply(batch, sign=-1.0)

    def backward(self, batch: dict[str, Any]) -> dict[str, Any]:
        return self._apply(batch, sign=1.0)


def _to_plain_dict(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _to_plain_dict(child) for key, child in value.items()}
    return value


def resolve_wsa_large_dataset_dirs(cfg: WSALargeDatasetConfig) -> list[str]:
    if cfg.dataset_dirs:
        return list(cfg.dataset_dirs)

    if cfg.repo_id_file is None:
        raise ValueError(
            "WSA_Large dataset needs either `dataset.dataset_dirs` or `dataset.repo_id_file`."
        )

    repo_id_file = Path(cfg.repo_id_file)
    if not repo_id_file.is_file():
        raise FileNotFoundError(f"WSA_Large dataset repo_id_file does not exist: {repo_id_file}")

    dataset_dirs = []
    with repo_id_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            dataset_dir = line.strip()
            if dataset_dir:
                dataset_dirs.append(dataset_dir)

    if not dataset_dirs:
        raise ValueError(f"WSA_Large dataset repo_id_file is empty: {repo_id_file}")

    return dataset_dirs


def sliding_window_with_replication(x: torch.Tensor, window_size: int) -> torch.Tensor:
    if x.dim() != 2:
        raise ValueError(f"Expected 2D tensor, got shape {tuple(x.shape)}")
    if window_size <= 0:
        raise ValueError(f"`window_size` must be positive, got {window_size}")

    num_rows = x.shape[0]
    row_idx = torch.arange(num_rows).unsqueeze(1)
    win_idx = torch.arange(window_size).unsqueeze(0)
    indices = torch.clamp(row_idx + win_idx, min=0, max=num_rows - 1)
    return x[indices]


class WSALargeMultiLeRobotDatasetV3(Dataset):
    def __init__(
        self,
        dataset_dirs: list[str],
        episodes: dict[str, list[int]] | None = None,
        image_transforms=None,
        delta_timestamps: dict[str, list[float]] | None = None,
        delta_timestamps_by_dataset: dict[str, dict[str, list[float]]] | None = None,
        video_backend: str | None = None,
        drop_non_intersection_features: bool = True,
        aggregate_metadata_stats: bool = True,
        dataset_weights: list[float] | None = None,
    ) -> None:
        super().__init__()
        self.dataset_dirs = dataset_dirs
        self._datasets: list[LeRobotDataset] = []

        for dataset_dir in dataset_dirs:
            root = Path(dataset_dir)
            repo_id = str(root)
            cur_delta_timestamps = (
                delta_timestamps_by_dataset.get(str(root), delta_timestamps)
                if delta_timestamps_by_dataset is not None
                else delta_timestamps
            )
            dataset = LeRobotDataset(
                repo_id=repo_id,
                root=root,
                episodes=episodes.get(dataset_dir) if episodes else None,
                image_transforms=image_transforms,
                delta_timestamps=cur_delta_timestamps,
                video_backend=video_backend,
            )
            self._datasets.append(dataset)

        if len(self._datasets) == 0:
            raise ValueError("At least one dataset directory is required.")

        fps_list = [dataset.fps for dataset in self._datasets]
        if delta_timestamps_by_dataset is None and len(set(fps_list)) != 1:
            raise ValueError(
                "All dataset_dirs must have the same fps when using shared delta_timestamps, "
                f"got {fps_list}"
            )

        self._lengths = [dataset.num_frames for dataset in self._datasets]
        self._cum_lengths = []
        running = 0
        for length in self._lengths:
            running += int(length)
            self._cum_lengths.append(running)

        if dataset_weights is None:
            self.dataset_weights = None
        else:
            if len(dataset_weights) != len(self._datasets):
                raise ValueError(
                    f"dataset_weights must have length {len(self._datasets)}, got {len(dataset_weights)}."
                )
            weights = torch.as_tensor(dataset_weights, dtype=torch.float32)
            if (weights < 0).any():
                raise ValueError("dataset_weights must be non-negative.")
            if float(weights.sum().item()) <= 0.0:
                raise ValueError("At least one dataset weight must be positive.")
            self.dataset_weights = weights / weights.sum()

        self.disabled_features: set[str] = set()
        if drop_non_intersection_features:
            intersection_features = set(self._datasets[0].features)
            for dataset in self._datasets:
                intersection_features.intersection_update(dataset.features)
            for dataset in self._datasets:
                extra_keys = set(dataset.features).difference(intersection_features)
                self.disabled_features.update(extra_keys)

        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        self.stats = aggregate_stats([dataset.meta.stats for dataset in self._datasets]) if aggregate_metadata_stats else {}

    def set_during_training(self, during_training: bool) -> None:
        del during_training

    @property
    def num_frames(self) -> int:
        return sum(dataset.num_frames for dataset in self._datasets)

    @property
    def num_episodes(self) -> int:
        return sum(dataset.num_episodes for dataset in self._datasets)

    @property
    def dataset_lengths(self) -> list[int]:
        return list(self._lengths)

    @property
    def dataset_cum_lengths(self) -> list[int]:
        return list(self._cum_lengths)

    def __len__(self) -> int:
        return self.num_frames

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds.")
        start_idx = 0
        dataset_idx = 0
        for cum_length in self._cum_lengths:
            if idx < cum_length:
                break
            start_idx = cum_length
            dataset_idx += 1
        item = self._datasets[dataset_idx][idx - start_idx]
        item["dataset_index"] = torch.tensor(dataset_idx)
        for data_key in self.disabled_features:
            if data_key in item:
                del item[data_key]
        return item

    def get_episode_data(self, episode_idx: int) -> dict[str, Any]:
        local_episode_idx = episode_idx
        for dataset in self._datasets:
            if local_episode_idx < dataset.num_episodes:
                return self._get_dataset_episode_data(dataset, local_episode_idx)
            local_episode_idx -= dataset.num_episodes
        raise IndexError(f"Episode index {episode_idx} out of bounds.")

    @staticmethod
    def _get_dataset_episode_data(dataset: LeRobotDataset, local_episode_idx: int) -> dict[str, Any]:
        dataset._ensure_hf_dataset_loaded()
        global_episode_idx = dataset.episodes[local_episode_idx] if dataset.episodes is not None else local_episode_idx
        episode_meta = dataset.meta.episodes[global_episode_idx]
        start_idx = int(episode_meta["dataset_from_index"])
        end_idx = int(episode_meta["dataset_to_index"])
        absolute_indices = list(range(start_idx, end_idx))
        if dataset._absolute_to_relative_idx is None:
            relative_indices = absolute_indices
        else:
            relative_indices = [dataset._absolute_to_relative_idx[idx] for idx in absolute_indices]

        result: dict[str, Any] = {}
        for key in dataset.features:
            if key in dataset.meta.video_keys:
                continue
            if key in {"task_index"}:
                continue
            try:
                result[key] = torch.stack(dataset.hf_dataset[key][relative_indices])
            except (KeyError, TypeError, IndexError):
                result[key] = torch.stack(dataset.hf_dataset[relative_indices][key])
        return result


class WSALargeBaseLerobotDatasetV3(Dataset):
    def __init__(
        self,
        dataset_dirs: list[str],
        shape_meta: dict[str, Any],
        action_size: int = 1,
        past_action_size: int = 0,
        obs_size: int = 1,
        past_obs_size: int = 0,
        val_set_proportion: float = 0.05,
        is_training_set: bool = False,
        seed: int = 42,
        global_sample_stride: int = 1,
        video_backend: str | None = None,
        pretrain_multi_embodiment: bool = False,
        max_action_dim: int | None = None,
        max_state_dim: int | None = None,
        norm_default_mode: str = "z-score",
        external_stats_root: str | None = None,
        action_mode: str = "abs",
        image_resize_shape: tuple[int, int] | None = None,
        dataset_weights: list[float] | None = None,
        post_image_transforms=None,
    ) -> None:
        if len(dataset_dirs) == 0:
            raise ValueError("At least one dataset directory is required")
        if past_action_size != 0 or past_obs_size != 0:
            raise ValueError("Only zero past_action_size and past_obs_size are supported.")
        if action_size != obs_size - 1:
            raise ValueError("In this dataset, action_size should be obs_size - 1")

        self.dataset_dirs = dataset_dirs
        self.shape_meta = shape_meta
        self.action_size = action_size
        self.past_action_size = past_action_size
        self.obs_size = obs_size
        self.processor: BaseProcessor | None = None
        self.is_training_set = is_training_set
        self.pretrain_multi_embodiment = bool(pretrain_multi_embodiment)
        self.max_action_dim = int(max_action_dim) if max_action_dim is not None else None
        self.max_state_dim = int(max_state_dim) if max_state_dim is not None else None
        self.norm_default_mode = str(norm_default_mode)
        self.external_stats_root = external_stats_root
        self.action_mode = str(action_mode)
        self.image_resize_shape = image_resize_shape
        self.post_image_transforms = post_image_transforms
        self.checkpoint_stats: dict[str, Any] | None = None
        self._multi_embodiment_stats_cache: dict[tuple[str, str], tuple[Path, dict[str, Any]]] = {}

        metas = [LeRobotDatasetMetadata(repo_id=str(Path(ds_dir)), root=Path(ds_dir)) for ds_dir in dataset_dirs]
        fps_list = [meta.fps for meta in metas]
        if not self.pretrain_multi_embodiment and len(set(fps_list)) != 1:
            raise ValueError(f"All dataset_dirs must have the same fps for shared-schema loading, got {fps_list}")
        fps = fps_list[0]

        self.image_meta = shape_meta["images"]
        self.state_meta = shape_meta["state"]
        self.action_meta = shape_meta["action"]

        delta_timestamps: dict[str, list[float]] = {}
        delta_timestamps_by_dataset: dict[str, dict[str, list[float]]] | None = None
        self.embodiment_adapters: list[dict[str, Any]] = []
        if self.pretrain_multi_embodiment:
            if self.max_action_dim is None or self.max_state_dim is None:
                raise ValueError("pretrain_multi_embodiment requires max_action_dim and max_state_dim.")
            self.embodiment_adapters = self._build_multi_embodiment_adapters(metas)
            self.checkpoint_stats = self._build_multi_embodiment_checkpoint_stats()
            delta_timestamps_by_dataset = {}
            for dataset_dir, adapter, meta in zip(dataset_dirs, self.embodiment_adapters, metas, strict=True):
                delta_timestamps_by_dataset[str(Path(dataset_dir))] = self._build_multi_embodiment_delta_timestamps(
                    adapter=adapter,
                    fps=meta.fps,
                    global_sample_stride=global_sample_stride,
                    obs_size=obs_size,
                    action_size=action_size,
                )
        else:
            for meta in self.image_meta:
                key = meta["key"]
                meta["lerobot_key"] = f"observation.images.{key}" if key != "default" else "observation.images"
                delta_timestamps[meta["lerobot_key"]] = [
                    (t * global_sample_stride) / fps for t in range(-past_obs_size, -past_obs_size + obs_size)
                ]
            for meta in self.state_meta:
                key = meta["key"]
                meta["lerobot_key"] = f"observation.state.{key}" if key != "default" else "observation.state"
                delta_timestamps[meta["lerobot_key"]] = [
                    (t * global_sample_stride) / fps for t in range(-past_obs_size, -past_obs_size + obs_size)
                ]
            for meta in self.action_meta:
                key = meta["key"]
                meta["lerobot_key"] = f"action.{key}" if key != "default" else "action"
                delta_timestamps[meta["lerobot_key"]] = [
                    (t * global_sample_stride) / fps for t in range(-past_action_size, -past_action_size + action_size)
                ]

            self._infer_feature_raw_shapes_from_metadata(metas)

        episodes: dict[str, list[int]] = {}
        if val_set_proportion < 1e-6:
            for meta in metas:
                episodes[str(meta.root)] = list(range(meta.total_episodes))
        else:
            for meta in metas:
                split_idx = int(meta.total_episodes * (1 - val_set_proportion))
                episode_indices = list(range(meta.total_episodes))
                rng = np.random.default_rng(seed)
                rng.shuffle(episode_indices)
                if self.is_training_set:
                    episodes[str(meta.root)] = [episode_indices[i] for i in range(split_idx)]
                else:
                    episodes[str(meta.root)] = [episode_indices[i] for i in range(split_idx, meta.total_episodes)]

        self.multi_dataset = WSALargeMultiLeRobotDatasetV3(
            dataset_dirs=self.dataset_dirs,
            episodes=episodes,
            delta_timestamps=delta_timestamps,
            delta_timestamps_by_dataset=delta_timestamps_by_dataset,
            video_backend=video_backend,
            drop_non_intersection_features=not self.pretrain_multi_embodiment,
            aggregate_metadata_stats=not self.pretrain_multi_embodiment,
            dataset_weights=dataset_weights,
        )

        episode_from = []
        episode_to = []
        end_index = 0
        for dataset in self.multi_dataset._datasets:
            current_from = torch.as_tensor(dataset.meta.episodes["dataset_from_index"], dtype=torch.long) + end_index
            current_to = torch.as_tensor(dataset.meta.episodes["dataset_to_index"], dtype=torch.long) + end_index
            episode_from.append(current_from)
            episode_to.append(current_to)
            end_index = int(current_to[-1].item())

        self.episode_data_index = {
            "from": torch.cat(episode_from),
            "to": torch.cat(episode_to),
        }

    def _build_multi_embodiment_adapters(self, metas: list[LeRobotDatasetMetadata]) -> list[dict[str, Any]]:
        adapters = []
        log_records: list[dict[str, Any]] = []
        for dataset_meta in metas:
            robot_type = dataset_meta.robot_type
            resolved_robot_type = infer_embodiment_variant(robot_type, dataset_meta.features)
            feature_mapping = get_feature_mapping(robot_type, dataset_meta.features)
            image_mapping = get_image_mapping(robot_type, dataset_meta.features)
            features = dataset_meta.features

            state_keys = [key for key in feature_mapping[OBS_STATE] if key in features]
            action_keys = [key for key in feature_mapping[ACTION] if key in features]
            has_action = len(action_keys) == len(feature_mapping[ACTION]) and len(action_keys) > 0
            if not state_keys:
                raise KeyError(
                    f"No state keys from FEATURE_MAPPING found for dataset {dataset_meta.root} "
                    f"(robot_type={robot_type}, resolved={resolved_robot_type})."
                )

            canonical_to_source = {}
            for source_key, mapped_key in image_mapping.items():
                if source_key in features:
                    canonical_to_source[_canonical_image_key(mapped_key)] = source_key
            if not canonical_to_source:
                raise KeyError(
                    f"No image keys from IMAGE_MAPPING found for dataset {dataset_meta.root} "
                    f"(robot_type={robot_type}, resolved={resolved_robot_type})."
                )

            stats_path, stats_payload = self._load_multi_embodiment_stats(robot_type, resolved_robot_type)
            adapters.append(
                {
                    "robot_type": robot_type,
                    "resolved_robot_type": resolved_robot_type,
                    "features": features,
                    "state_keys": state_keys,
                    "action_keys": action_keys,
                    "has_action": has_action,
                    "canonical_to_source": canonical_to_source,
                    "delta_mask": torch.as_tensor(get_mask_mapping(robot_type, dataset_meta.features), dtype=torch.bool),
                    "stats": stats_payload,
                    "stats_path": stats_path,
                }
            )
            log_records.append(
                {
                    "root": dataset_meta.root,
                    "robot_type": robot_type,
                    "resolved_robot_type": resolved_robot_type,
                    "state_keys": tuple(state_keys),
                    "action_keys": tuple(action_keys),
                    "has_action": has_action,
                    "canonical_to_source": tuple(sorted(canonical_to_source.items())),
                    "stats_path": stats_path,
                }
            )
        self._log_multi_embodiment_adapter_summary(log_records)
        return adapters

    def _build_multi_embodiment_checkpoint_stats(self) -> dict[str, Any]:
        stats_by_key: dict[str, Any] = {}
        stats_source_by_key: dict[str, str] = {}
        for adapter in self.embodiment_adapters:
            stats_payload = _to_plain_dict(adapter["stats"])
            stats_source = str(adapter["stats_path"])
            for key in (adapter["resolved_robot_type"], adapter["robot_type"]):
                key = str(key)
                if not key:
                    continue
                if key in stats_by_key:
                    if stats_source_by_key[key] != stats_source:
                        logger.warning(
                            "WSA_Large checkpoint stats key %s already came from %s; "
                            "keeping it and ignoring duplicate source %s.",
                            key,
                            stats_source_by_key[key],
                            stats_source,
                        )
                    continue
                stats_by_key[key] = stats_payload
                stats_source_by_key[key] = stats_source
        return stats_by_key

    def _log_multi_embodiment_adapter_summary(self, records: list[dict[str, Any]]) -> None:
        mode = os.environ.get("LEROBOT_WSA_LARGE_ADAPTER_LOG_MODE", "summary").strip().lower()
        if mode in {"0", "false", "off", "none", "quiet"}:
            return

        preview_limit = max(_env_int("LEROBOT_WSA_LARGE_ADAPTER_LOG_LIMIT", 3), 0)

        if mode in {"full", "verbose", "all"}:
            for record in records:
                logger.info(
                    "WSA_Large pretrain adapter: root=%s robot_type=%s resolved=%s state_keys=%s action_keys=%s has_action=%s images=%s stats=%s",
                    record["root"],
                    record["robot_type"],
                    record["resolved_robot_type"],
                    list(record["state_keys"]),
                    list(record["action_keys"]),
                    record["has_action"],
                    dict(record["canonical_to_source"]),
                    record["stats_path"],
                )
            return

        groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            key = (
                record["robot_type"],
                record["resolved_robot_type"],
                record["state_keys"],
                record["action_keys"],
                record["has_action"],
                record["canonical_to_source"],
                record["stats_path"],
            )
            groups[key].append(record)

        logger.info(
            "WSA_Large pretrain adapter summary: repos=%d groups=%d stats_files=%d",
            len(records),
            len(groups),
            len({record["stats_path"] for record in records}),
        )
        for key, group_records in sorted(groups.items(), key=lambda item: (-len(item[1]), str(item[0]))):
            robot_type, resolved_robot_type, state_keys, action_keys, has_action, canonical_to_source, stats_path = key
            examples = [_short_path_for_log(record["root"]) for record in group_records[:preview_limit]]
            if len(group_records) > preview_limit:
                examples.append(f"+{len(group_records) - preview_limit} more")
            logger.info(
                "WSA_Large adapter group: count=%d robot_type=%s resolved=%s state_keys=%s action_keys=%s has_action=%s images=%s stats=%s examples=%s",
                len(group_records),
                robot_type,
                resolved_robot_type,
                list(state_keys),
                list(action_keys),
                has_action,
                dict(canonical_to_source),
                stats_path,
                examples,
            )

    def _load_multi_embodiment_stats(self, robot_type: str, resolved_robot_type: str) -> tuple[Path, dict[str, Any]]:
        cache_key = (str(robot_type), str(resolved_robot_type))
        if cache_key in self._multi_embodiment_stats_cache:
            return self._multi_embodiment_stats_cache[cache_key]
        if not self.external_stats_root:
            raise ValueError(
                "pretrain_multi_embodiment requires dataset.external_stats_root so each robot_type can use "
                "its own normalization stats."
            )
        root = Path(self.external_stats_root)
        candidates = [
            root / resolved_robot_type / self.action_mode / "stats.json",
            root / robot_type / self.action_mode / "stats.json",
        ]
        for path in candidates:
            if path.is_file():
                payload = load_dataset_stats_from_json(str(path))
                self._multi_embodiment_stats_cache[cache_key] = (path, payload)
                return path, payload
        raise FileNotFoundError(
            "Missing external stats for WSA_Large pretrain. Tried: "
            + ", ".join(str(path) for path in candidates)
        )

    @staticmethod
    def _build_multi_embodiment_delta_timestamps(
        adapter: dict[str, Any],
        fps: int,
        global_sample_stride: int,
        obs_size: int,
        action_size: int,
    ) -> dict[str, list[float]]:
        obs_times = [(t * global_sample_stride) / fps for t in range(obs_size)]
        action_times = [(t * global_sample_stride) / fps for t in range(action_size)]
        delta_timestamps = {}
        for key in adapter["canonical_to_source"].values():
            delta_timestamps[key] = obs_times
        for key in adapter["state_keys"]:
            delta_timestamps[key] = obs_times
        if adapter["has_action"]:
            for key in adapter["action_keys"]:
                delta_timestamps[key] = action_times
        return delta_timestamps

    def _infer_feature_raw_shapes_from_metadata(self, metas: list[LeRobotDatasetMetadata]) -> None:
        def _collect_feature_specs(feature_key: str) -> list[dict[str, Any]]:
            specs = []
            for dataset_meta in metas:
                if feature_key not in dataset_meta.features:
                    raise KeyError(
                        f"Feature '{feature_key}' not found in dataset metadata for {dataset_meta.root}."
                    )
                specs.append(dataset_meta.features[feature_key])
            return specs

        def _canonicalize_visual_shape(shape: Any, names: Any) -> list[int]:
            shape_list = list(shape)
            if len(shape_list) != 3:
                return shape_list
            if names is not None and len(names) == 3 and names[2] in ["channel", "channels"]:
                return [shape_list[2], shape_list[0], shape_list[1]]
            if shape_list[-1] in [1, 3, 4] and shape_list[0] not in [1, 3, 4]:
                return [shape_list[2], shape_list[0], shape_list[1]]
            return shape_list

        def _all_same(shapes: list[Any]) -> bool:
            if not shapes:
                return True
            first = tuple(shapes[0]) if isinstance(shapes[0], (list, tuple)) else shapes[0]
            for shape in shapes[1:]:
                current = tuple(shape) if isinstance(shape, (list, tuple)) else shape
                if current != first:
                    return False
            return True

        for meta in self.image_meta:
            specs = _collect_feature_specs(meta["lerobot_key"])
            shapes = [_canonicalize_visual_shape(spec["shape"], spec.get("names")) for spec in specs]
            if not _all_same(shapes):
                raise ValueError(
                    f"Inconsistent image shapes for '{meta['lerobot_key']}' across datasets: {shapes}"
                )
            actual_shape = list(shapes[0])
            if list(meta["raw_shape"]) != actual_shape:
                logger.info(
                    "Overriding WSA_Large image raw_shape for %s from %s to dataset metadata shape %s.",
                    meta["lerobot_key"],
                    meta["raw_shape"],
                    actual_shape,
                )
                meta["raw_shape"] = actual_shape

        for meta in self.state_meta:
            specs = _collect_feature_specs(meta["lerobot_key"])
            shapes = [spec["shape"] for spec in specs]
            if not _all_same(shapes):
                raise ValueError(
                    f"Inconsistent state shapes for '{meta['lerobot_key']}' across datasets: {shapes}"
                )
            actual_shape = shapes[0]
            actual_dim = int(actual_shape[0]) if isinstance(actual_shape, (list, tuple)) else int(actual_shape)
            if int(meta["raw_shape"]) != actual_dim:
                logger.info(
                    "Overriding WSA_Large state raw_shape for %s from %s to dataset metadata dim %s.",
                    meta["lerobot_key"],
                    meta["raw_shape"],
                    actual_dim,
                )
                meta["raw_shape"] = actual_dim

        for meta in self.action_meta:
            specs = _collect_feature_specs(meta["lerobot_key"])
            shapes = [spec["shape"] for spec in specs]
            if not _all_same(shapes):
                raise ValueError(
                    f"Inconsistent action shapes for '{meta['lerobot_key']}' across datasets: {shapes}"
                )
            actual_shape = shapes[0]
            actual_dim = int(actual_shape[0]) if isinstance(actual_shape, (list, tuple)) else int(actual_shape)
            if int(meta["raw_shape"]) != actual_dim:
                logger.info(
                    "Overriding WSA_Large action raw_shape for %s from %s to dataset metadata dim %s.",
                    meta["lerobot_key"],
                    meta["raw_shape"],
                    actual_dim,
                )
                meta["raw_shape"] = actual_dim

    def _get_action(self, meta: dict[str, Any], lerobot_sample: dict[str, Any]) -> torch.Tensor:
        key, lerobot_key, raw_shape = meta["key"], meta["lerobot_key"], meta["raw_shape"]
        action = lerobot_sample[lerobot_key]
        if action.ndim == 1:
            action = action.unsqueeze(-1)
        if action.shape[-1] != raw_shape:
            raise ValueError(f"Action '{key}' shape {action.shape[-1]} mismatch with meta {raw_shape}.")
        return action

    def _get_state(self, meta: dict[str, Any], lerobot_sample: dict[str, Any]) -> torch.Tensor:
        key, lerobot_key, raw_shape = meta["key"], meta["lerobot_key"], meta["raw_shape"]
        state = lerobot_sample[lerobot_key]
        if state.ndim == 1:
            state = state.unsqueeze(-1)
        if state.shape[-1] != raw_shape:
            raise ValueError(f"State '{key}' shape {state.shape[-1]} mismatch with meta {raw_shape}.")
        return state

    def _get_image(self, meta: dict[str, Any], lerobot_sample: dict[str, Any]) -> torch.Tensor:
        key, lerobot_key, raw_shape = meta["key"], meta["lerobot_key"], meta["raw_shape"]
        image = lerobot_sample[lerobot_key]
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4:
            raise ValueError(f"Expected image tensor with 4 dims for key '{key}', got {tuple(image.shape)}")
        if image.shape[-1] in [1, 3, 4] and image.shape[1] not in [1, 3, 4]:
            image = image.permute(0, 3, 1, 2).contiguous()
        if image.dtype != torch.uint8:
            if image.is_floating_point():
                image = (image.clamp(0.0, 1.0) * 255).to(torch.uint8)
            else:
                image = image.to(torch.uint8)
        expected_shape = list(raw_shape)
        if len(expected_shape) == 3 and expected_shape[-1] in [1, 3, 4] and expected_shape[0] not in [1, 3, 4]:
            expected_shape = [expected_shape[2], expected_shape[0], expected_shape[1]]
        if list(image.shape[1:]) != expected_shape:
            raise ValueError(f"Image '{key}' shape {tuple(image.shape[1:])} mismatch with meta {raw_shape}.")
        return image

    @staticmethod
    def _as_sequence_vector(value: torch.Tensor) -> torch.Tensor:
        if value.ndim == 1:
            return value.unsqueeze(-1)
        if value.ndim != 2:
            raise ValueError(f"Expected sequence vector [T,D] or [T], got shape {tuple(value.shape)}")
        return value.float()

    @staticmethod
    def _as_sequence_image(value: torch.Tensor) -> torch.Tensor:
        image = value
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4:
            raise ValueError(f"Expected image sequence [T,C,H,W] or [T,H,W,C], got {tuple(image.shape)}")
        if image.shape[-1] in [1, 3, 4] and image.shape[1] not in [1, 3, 4]:
            image = image.permute(0, 3, 1, 2).contiguous()
        if image.dtype != torch.uint8:
            if image.is_floating_point():
                image = (image.clamp(0.0, 1.0) * 255).to(torch.uint8)
            else:
                image = image.to(torch.uint8)
        return image

    @staticmethod
    def _align_for_cat(tensors: list[torch.Tensor]) -> tuple[list[torch.Tensor], list[int]]:
        max_ndim = max(tensor.ndim for tensor in tensors)
        out = []
        sizes = []
        for tensor in tensors:
            tensor = tensor if tensor.ndim == max_ndim else tensor.unsqueeze(-1)
            out.append(tensor)
            sizes.append(tensor.shape[-1])
        return out, sizes

    @staticmethod
    def _split_by_sizes(tensor: torch.Tensor, sizes: list[int]) -> list[torch.Tensor]:
        chunks = []
        start = 0
        for size in sizes:
            chunks.append(tensor[..., start : start + size])
            start += size
        return chunks

    @staticmethod
    def _pad_vector_with_dim_mask(tensor: torch.Tensor, target_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
        if tensor.shape[-1] > target_dim:
            raise ValueError(f"Cannot pad vector dim {tensor.shape[-1]} to smaller target_dim={target_dim}.")
        pad_dim = target_dim - tensor.shape[-1]
        if pad_dim > 0:
            tensor = torch.nn.functional.pad(tensor, (0, pad_dim))
        dim_mask = torch.zeros(target_dim, dtype=torch.bool, device=tensor.device)
        if pad_dim > 0:
            dim_mask[-pad_dim:] = True
        return tensor, dim_mask

    @staticmethod
    def _combined_padding_mask(lerobot_sample: dict[str, Any], keys: list[str], length: int) -> torch.Tensor:
        masks = []
        for key in keys:
            pad_key = f"{key}_is_pad"
            if pad_key in lerobot_sample:
                masks.append(torch.as_tensor(lerobot_sample[pad_key], dtype=torch.bool))
        if not masks:
            return torch.zeros(length, dtype=torch.bool)
        out = masks[0].clone()
        for mask in masks[1:]:
            out = out | mask.to(dtype=torch.bool)
        return out

    @staticmethod
    def _stats_for_key(stats_payload: dict[str, Any], key: str) -> dict[str, Any]:
        if key in stats_payload:
            return stats_payload[key]
        for group in ("action", "state"):
            group_stats = stats_payload.get(group)
            if isinstance(group_stats, dict) and key in group_stats:
                return group_stats[key]
        raise KeyError(f"Missing normalization stats for source key {key!r}. Available: {list(stats_payload.keys())}")

    def _normalize_source_tensor(self, tensor: torch.Tensor, stats_payload: dict[str, Any], key: str) -> torch.Tensor:
        return _normalize_with_stats(
            tensor.float(),
            self._stats_for_key(stats_payload, key),
            self.norm_default_mode,
        )

    def _apply_post_image_transform(self, key: str, image: torch.Tensor) -> torch.Tensor:
        if self.post_image_transforms is None:
            return image
        if hasattr(self.post_image_transforms, "set_current_key"):
            self.post_image_transforms.set_current_key(key)
        return self.post_image_transforms(image)

    def _build_multi_embodiment_sample(self, lerobot_sample: dict[str, Any]) -> dict[str, Any]:
        dataset_index = int(torch.as_tensor(lerobot_sample["dataset_index"]).item())
        adapter = self.embodiment_adapters[dataset_index]
        stats_payload = adapter["stats"]

        image_tensors: dict[str, torch.Tensor] = {}
        camera_view_mask = []
        first_processed_image = None
        resize_shape = self.image_resize_shape
        if resize_shape is None:
            resize_shape = tuple(self.image_meta[0]["shape"][1:])
        to_tensor = ToTensor()
        resize = tv_transforms.Resize(size=list(resize_shape))

        for meta in self.image_meta:
            key = meta["key"]
            source_key = adapter["canonical_to_source"].get(key)
            if source_key is None:
                camera_view_mask.append(False)
                image_tensors[key] = None
                continue
            image = self._as_sequence_image(lerobot_sample[source_key])
            image = resize(to_tensor(image))
            image = self._apply_post_image_transform(key, image)
            first_processed_image = image if first_processed_image is None else first_processed_image
            image_tensors[key] = image
            camera_view_mask.append(True)

        if first_processed_image is None:
            raise ValueError(f"No valid images found for dataset_index={dataset_index}.")
        for key, value in list(image_tensors.items()):
            if value is None:
                image_tensors[key] = torch.zeros_like(first_processed_image)

        state_tensors = [self._as_sequence_vector(lerobot_sample[key]) for key in adapter["state_keys"]]
        state_tensors, state_sizes = self._align_for_cat(state_tensors)
        state_concat_raw = torch.cat(state_tensors, dim=-1)

        has_action = bool(adapter["has_action"])
        use_action_loss = has_action and adapter["resolved_robot_type"] != "egodex_v"
        use_action_input = has_action
        if use_action_input:
            action_tensors = [self._as_sequence_vector(lerobot_sample[key]) for key in adapter["action_keys"]]
            action_tensors, action_sizes = self._align_for_cat(action_tensors)
            action_concat_raw = torch.cat(action_tensors, dim=-1)
            if self.action_mode == "delta":
                dim = min(action_concat_raw.shape[-1], state_concat_raw.shape[-1], adapter["delta_mask"].numel())
                delta_mask = adapter["delta_mask"][:dim].to(device=action_concat_raw.device)
                base_state = state_concat_raw[:1, :dim].to(device=action_concat_raw.device, dtype=action_concat_raw.dtype)
                action_concat_raw = action_concat_raw.clone()
                delta_slice = action_concat_raw[:, :dim]
                delta_slice[:, delta_mask] = delta_slice[:, delta_mask] - base_state[:, delta_mask]
                action_concat_raw[:, :dim] = delta_slice
            action_tensors = self._split_by_sizes(action_concat_raw, action_sizes)
            action_tensors = [
                self._normalize_source_tensor(tensor, stats_payload, key)
                for tensor, key in zip(action_tensors, adapter["action_keys"], strict=True)
            ]
            action_concat = torch.cat(action_tensors, dim=-1)
            action, action_dim_is_pad = self._pad_vector_with_dim_mask(action_concat, self.max_action_dim)
            action_is_pad = self._combined_padding_mask(lerobot_sample, adapter["action_keys"], self.action_size)
            sample_action_loss_mask = torch.tensor([1.0 if use_action_loss else 0.0], dtype=torch.float32)
        else:
            action = torch.zeros(self.action_size, self.max_action_dim, dtype=torch.float32)
            action_dim_is_pad = torch.ones(self.max_action_dim, dtype=torch.bool)
            action_is_pad = torch.ones(self.action_size, dtype=torch.bool)
            sample_action_loss_mask = torch.tensor([0.0], dtype=torch.float32)

        state_tensors = [
            self._normalize_source_tensor(tensor, stats_payload, key)
            for tensor, key in zip(state_tensors, adapter["state_keys"], strict=True)
        ]
        state_concat = torch.cat(state_tensors, dim=-1)
        state, state_dim_is_pad = self._pad_vector_with_dim_mask(state_concat, self.max_state_dim)
        state_is_pad = self._combined_padding_mask(lerobot_sample, adapter["state_keys"], self.obs_size)
        image_is_pad = self._combined_padding_mask(
            lerobot_sample,
            list(adapter["canonical_to_source"].values()),
            self.obs_size,
        )

        return {
            "idx": int(torch.as_tensor(lerobot_sample.get("index", 0)).item()) if "index" in lerobot_sample else 0,
            "instruction": self.processor.augment_instruction(lerobot_sample) if self.processor is not None else str(lerobot_sample["task"]),
            "pixel_values": torch.stack([image_tensors[meta["key"]] for meta in self.image_meta], dim=0),
            "action": action,
            "action_is_pad": action_is_pad,
            "action_dim_is_pad": action_dim_is_pad,
            "proprio": state,
            "proprio_is_pad": state_is_pad,
            "proprio_dim_is_pad": state_dim_is_pad,
            "image_is_pad": image_is_pad,
            "camera_view_mask": torch.as_tensor(camera_view_mask, dtype=torch.bool),
            SAMPLE_ACTION_LOSS_MASK: sample_action_loss_mask,
            "robot_type": adapter["resolved_robot_type"],
            "has_action": torch.tensor(has_action, dtype=torch.bool),
        }

    def _split_lerobot_sample(self, lerobot_sample: dict[str, Any]) -> dict[str, Any]:
        return lerobot_sample

    def _get_episode_data(self, episode_idx: int) -> dict[str, Any]:
        lerobot_sample = self.multi_dataset.get_episode_data(episode_idx)
        lerobot_sample = self._split_lerobot_sample(lerobot_sample)
        state: dict[str, torch.Tensor] = {}
        action: dict[str, torch.Tensor] = {}
        for meta in self.state_meta:
            state_tensor = self._get_state(meta, lerobot_sample)
            state[meta["key"]] = state_tensor.unsqueeze(1).float()
        for meta in self.action_meta:
            action_tensor = self._get_action(meta, lerobot_sample)
            action[meta["key"]] = sliding_window_with_replication(action_tensor, self.action_size).float()
        return {"action": action, "state": state}

    def _set_return_images(self, flag: bool) -> None:
        self.return_images = flag
        self.multi_dataset.set_during_training(flag)

    def __len__(self) -> int:
        return self.multi_dataset.num_frames

    def _get_additional_data(self, sample: dict[str, Any], lerobot_sample: dict[str, Any]) -> dict[str, Any]:
        del lerobot_sample
        return sample

    def __getitem__(self, idx: int):
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds {len(self)}.")

        sample_idx = idx
        attempt = 0
        last_exception: Optional[Exception] = None
        while attempt < MAX_GETITEM_ATTEMPT:
            try:
                lerobot_sample = self.multi_dataset[sample_idx]
                lerobot_sample = self._split_lerobot_sample(lerobot_sample)
                if self.pretrain_multi_embodiment:
                    return self._build_multi_embodiment_sample(lerobot_sample)
                break
            except Exception as err:
                attempt += 1
                last_exception = err
                logger.warning(
                    "Error loading sample %s (attempt %s). Retrying with a random index. Error: %s",
                    sample_idx,
                    attempt,
                    err,
                )
                sample_idx = np.random.randint(len(self))
                print(traceback.format_exc())
        else:
            raise RuntimeError(
                f"Failed to load a valid sample after {MAX_GETITEM_ATTEMPT} attempts for index {idx}."
            ) from last_exception

        sample = {
            "idx": sample_idx,
            "task": lerobot_sample["task"],
            "action": {},
            "state": {},
            "images": {},
        }
        for meta in self.state_meta:
            sample["state"][meta["key"]] = self._get_state(meta, lerobot_sample)
        for meta in self.action_meta:
            sample["action"][meta["key"]] = self._get_action(meta, lerobot_sample)
        for meta in self.image_meta:
            sample["images"][meta["key"]] = self._get_image(meta, lerobot_sample)

        sample["action_is_pad"] = lerobot_sample[f"{self.action_meta[0]['lerobot_key']}_is_pad"]
        sample["state_is_pad"] = lerobot_sample[f"{self.state_meta[0]['lerobot_key']}_is_pad"]
        sample["image_is_pad"] = lerobot_sample[f"{self.image_meta[0]['lerobot_key']}_is_pad"]

        sample = self._get_additional_data(sample, lerobot_sample)

        for key, value in lerobot_sample.items():
            if key not in sample and "observation" not in key and "action" not in key:
                sample[key] = value

        if self.processor is not None:
            sample = self.processor.preprocess(sample)
        return sample

    def set_processor(self, processor: BaseProcessor):
        self.processor = processor
        if self.is_training_set:
            self.processor.train()
        else:
            self.processor.eval()
        return self

    def get_dataset_stats(self, preprocessor: BaseProcessor):
        state_min = defaultdict(list)
        state_max = defaultdict(list)
        state_mean = defaultdict(list)
        state_var = defaultdict(list)
        state_q01 = defaultdict(list)
        state_q99 = defaultdict(list)
        action_min = defaultdict(list)
        action_max = defaultdict(list)
        action_mean = defaultdict(list)
        action_var = defaultdict(list)
        action_q01 = defaultdict(list)
        action_q99 = defaultdict(list)

        episodes_num = self.multi_dataset.num_episodes

        def process_episode(episode_idx: int):
            batch = self._get_episode_data(episode_idx)
            return preprocessor.action_state_transform(batch)

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_episode, num) for num in range(episodes_num)]
            for future in as_completed(futures):
                batch = future.result()
                for meta in self.state_meta:
                    key = meta["key"]
                    cur_state = batch["state"][key]
                    state_min[key].append(cur_state.amin(0))
                    state_max[key].append(cur_state.amax(0))
                    state_mean[key].append(cur_state.mean(0))
                    state_var[key].append(cur_state.var(0))
                    state_q01[key].append(torch.quantile(cur_state, 0.01, dim=0, keepdim=False))
                    state_q99[key].append(torch.quantile(cur_state, 0.99, dim=0, keepdim=False))
                for meta in self.action_meta:
                    key = meta["key"]
                    cur_action = batch["action"][key]
                    action_min[key].append(cur_action.amin(0))
                    action_max[key].append(cur_action.amax(0))
                    action_mean[key].append(cur_action.mean(0))
                    action_var[key].append(cur_action.var(0))
                    action_q01[key].append(torch.quantile(cur_action, 0.01, dim=0, keepdim=False))
                    action_q99[key].append(torch.quantile(cur_action, 0.99, dim=0, keepdim=False))

        def get_mean_std(means: list[torch.Tensor], vars_: list[torch.Tensor]):
            means_tensor = torch.stack(means)
            vars_tensor = torch.stack(vars_)
            stepwise_mean = means_tensor.mean(0)
            stepwise_std = (vars_tensor + (means_tensor - stepwise_mean) ** 2).mean(0).sqrt()
            global_mean = means_tensor.mean((0, 1))
            global_std = (vars_tensor + (means_tensor - global_mean) ** 2).mean((0, 1)).sqrt()
            return stepwise_mean, stepwise_std, global_mean, global_std

        stats = {
            "state": defaultdict(dict),
            "action": defaultdict(dict),
            "num_episodes": episodes_num,
            "num_transition": self.multi_dataset.num_frames,
        }
        for meta in self.state_meta:
            key = meta["key"]
            stats["state"][key]["stepwise_min"] = torch.stack(state_min[key]).amin(0)
            stats["state"][key]["stepwise_max"] = torch.stack(state_max[key]).amax(0)
            stats["state"][key]["global_min"] = stats["state"][key]["stepwise_min"].amin(0)
            stats["state"][key]["global_max"] = stats["state"][key]["stepwise_max"].amax(0)
            stats["state"][key]["stepwise_q01"] = torch.stack(state_q01[key]).amin(0)
            stats["state"][key]["stepwise_q99"] = torch.stack(state_q99[key]).amax(0)
            stats["state"][key]["global_q01"] = stats["state"][key]["stepwise_q01"].amin(0)
            stats["state"][key]["global_q99"] = stats["state"][key]["stepwise_q99"].amax(0)
            (
                stats["state"][key]["stepwise_mean"],
                stats["state"][key]["stepwise_std"],
                stats["state"][key]["global_mean"],
                stats["state"][key]["global_std"],
            ) = get_mean_std(state_mean[key], state_var[key])
        for meta in self.action_meta:
            key = meta["key"]
            stats["action"][key]["stepwise_min"] = torch.stack(action_min[key]).amin(0)
            stats["action"][key]["stepwise_max"] = torch.stack(action_max[key]).amax(0)
            stats["action"][key]["global_min"] = stats["action"][key]["stepwise_min"].amin(0)
            stats["action"][key]["global_max"] = stats["action"][key]["stepwise_max"].amax(0)
            stats["action"][key]["stepwise_q01"] = torch.stack(action_q01[key]).amin(0)
            stats["action"][key]["stepwise_q99"] = torch.stack(action_q99[key]).amax(0)
            stats["action"][key]["global_q01"] = stats["action"][key]["stepwise_q01"].amin(0)
            stats["action"][key]["global_q99"] = stats["action"][key]["stepwise_q99"].amax(0)
            (
                stats["action"][key]["stepwise_mean"],
                stats["action"][key]["stepwise_std"],
                stats["action"][key]["global_mean"],
                stats["action"][key]["global_std"],
            ) = get_mean_std(action_mean[key], action_var[key])
        return stats


class WSALargeRobotVideoDatasetV3(Dataset):
    def __init__(
        self,
        dataset_dirs: list[str],
        shape_meta: dict[str, Any],
        num_frames: int = 33,
        video_size: tuple[int, int] = (224, 448),
        standardize_video_size_by_cameras: bool = True,
        camera_key: str | None = None,
        processor: BaseProcessor | None = None,
        text_embedding_cache_dir: str | None = None,
        text_embedding_cache_max_entries: int = 0,
        cache_in_memory: bool = False,
        context_len: int = 128,
        normalization_stats_path: str | None = None,
        use_lerobot_meta_stats: bool = False,
        val_set_proportion: float = 0.05,
        is_training_set: bool = False,
        global_sample_stride: int = 1,
        action_video_freq_ratio: int = 1,
        skip_padding_as_possible: bool = False,
        max_padding_retry: int = 3,
        concat_multi_camera: str = "horizontal",
        override_instruction: str | None = None,
        video_backend: str | None = None,
        return_future_3d_images: bool = False,
        future_3d_target_index: int = -1,
        pretrain_multi_embodiment: bool = False,
        max_action_dim: int | None = None,
        max_state_dim: int | None = None,
        norm_default_mode: str = "z-score",
        external_stats_root: str | None = None,
        action_mode: str = "abs",
        dataset_weights: list[float] | None = None,
        image_transforms=None,
    ) -> None:
        self.lerobot_dataset = WSALargeBaseLerobotDatasetV3(
            dataset_dirs=dataset_dirs,
            shape_meta=shape_meta,
            obs_size=num_frames,
            action_size=num_frames - 1,
            val_set_proportion=val_set_proportion,
            is_training_set=is_training_set,
            global_sample_stride=global_sample_stride,
            video_backend=video_backend,
            pretrain_multi_embodiment=pretrain_multi_embodiment,
            max_action_dim=max_action_dim,
            max_state_dim=max_state_dim,
            norm_default_mode=norm_default_mode,
            external_stats_root=external_stats_root,
            action_mode=action_mode,
            image_resize_shape=tuple(shape_meta["images"][0]["shape"][1:]),
            dataset_weights=dataset_weights,
            post_image_transforms=image_transforms,
        )
        self.clip_num_frames = num_frames
        self.action_video_freq_ratio = action_video_freq_ratio
        if (num_frames - 1) % self.action_video_freq_ratio != 0:
            raise ValueError(
                f"num_frames-1 must be divisible by action_video_freq_ratio, got {num_frames - 1} and {self.action_video_freq_ratio}"
            )
        if ((num_frames - 1) // self.action_video_freq_ratio) % 4 != 0:
            raise ValueError(
                "video frames must be divisible by 4 for tokenization, "
                f"got {(num_frames - 1) // self.action_video_freq_ratio}"
            )
        self.video_sample_indices = list(range(0, num_frames, self.action_video_freq_ratio))
        self.camera_key = camera_key
        self.lerobot_dataset._set_return_images(True)

        self.video_size = tuple(int(value) for value in video_size)
        self.standardize_video_size_by_cameras = standardize_video_size_by_cameras
        self.text_embedding_cache_dir = text_embedding_cache_dir
        self.text_embedding_cache_max_entries = int(text_embedding_cache_max_entries)
        if self.text_embedding_cache_max_entries < 0:
            raise ValueError("text_embedding_cache_max_entries must be non-negative.")
        self._cached_text_contexts: OrderedDict[str, tuple[torch.Tensor, torch.Tensor]] = OrderedDict()
        self._memory_cache: list[dict[str, Any]] | None = None
        self.context_len = context_len
        self.skip_padding_as_possible = skip_padding_as_possible
        self.max_padding_retry = max_padding_retry
        self.concat_multi_camera = concat_multi_camera
        self.override_instruction = override_instruction
        self.return_future_3d_images = return_future_3d_images
        self.future_3d_target_index = int(future_3d_target_index)

        self.normalize_transform = Normalize(args={"mean": 0.5, "std": 0.5})

        self.dataset_stats = None
        self.checkpoint_stats = getattr(self.lerobot_dataset, "checkpoint_stats", None)
        if processor is not None and not pretrain_multi_embodiment:
            stats_path = Path(normalization_stats_path) if normalization_stats_path is not None else None
            if stats_path is not None and stats_path.is_file():
                dataset_stats = load_dataset_stats_from_json(str(stats_path))
                logger.info("Using dataset stats from %s", stats_path)
            elif use_lerobot_meta_stats:
                if processor.action_state_transforms is not None:
                    raise ValueError(
                        "LeRobot metadata stats are raw absolute-action stats and cannot be used after "
                        "WSA_Large action/state transforms. Use ACTION_TYPE=abs for USE_EXTERNAL_STATS=false, "
                        "or provide delta stats with USE_EXTERNAL_STATS=true."
                    )
                logger.info("Using LeRobot metadata stats for WSA_Large normalization.")
                dataset_stats = self.lerobot_dataset.multi_dataset.stats
            else:
                if not is_training_set:
                    raise ValueError(
                        "normalization_stats_path must point to an existing stats file for validation/test datasets."
                    )
                logger.info("Calculating dataset stats for normalization...")
                dataset_stats = self.lerobot_dataset.get_dataset_stats(processor)
                if stats_path is not None:
                    stats_path.parent.mkdir(parents=True, exist_ok=True)
                    save_dataset_stats_to_json(dataset_stats, str(stats_path))
            dataset_stats = ensure_wsa_large_stats_format(
                dataset_stats,
                self.lerobot_dataset.shape_meta,
                require_state=True,
            )
            processor.set_normalizer_from_stats(dataset_stats)
            self.dataset_stats = _to_plain_dict(dataset_stats)
            self.lerobot_dataset.set_processor(processor)

        if cache_in_memory:
            if image_transforms is not None:
                logger.warning(
                    "WSA_Large cache_in_memory=True freezes image augmentation at preload time."
                )
            self.preload_in_memory()

    @property
    def num_episodes(self) -> int:
        return self.lerobot_dataset.multi_dataset.num_episodes

    @property
    def num_frames(self) -> int:
        return self.lerobot_dataset.multi_dataset.num_frames

    @property
    def num_frames_total(self) -> int:
        return self.num_frames

    @property
    def dataset_weights(self) -> torch.Tensor | None:
        return self.lerobot_dataset.multi_dataset.dataset_weights

    @property
    def dataset_lengths(self) -> list[int]:
        return self.lerobot_dataset.multi_dataset.dataset_lengths

    @property
    def dataset_cum_lengths(self) -> list[int]:
        return self.lerobot_dataset.multi_dataset.dataset_cum_lengths

    def __len__(self) -> int:
        return len(self.lerobot_dataset)

    @staticmethod
    def _resize_video_view(video: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        return transforms_f.resize(
            video,
            size=list(size),
            interpolation=transforms_f.InterpolationMode.BILINEAR,
            antialias=True,
        )

    def _concat_camera_views(
        self,
        video: torch.Tensor,
        num_cameras: int,
        target_video_size: tuple[int, int],
        concat_layout: str,
    ) -> torch.Tensor:
        target_h, target_w = target_video_size
        if concat_layout == "single":
            if num_cameras != 1:
                raise ValueError(f"`single` camera layout requires 1 camera, got {num_cameras}.")
            return self._resize_video_view(video[0], target_video_size)

        if concat_layout == "horizontal":
            if target_w % num_cameras != 0:
                raise ValueError(
                    "horizontal camera layout requires target width divisible by camera count: "
                    f"width={target_w}, num_cameras={num_cameras}."
                )
            tile_w = target_w // num_cameras
            views = [
                self._resize_video_view(video[view_idx], (target_h, tile_w))
                for view_idx in range(num_cameras)
            ]
            return torch.cat(views, dim=-1)

        if concat_layout == "vertical":
            if target_h % num_cameras != 0:
                raise ValueError(
                    "vertical camera layout requires target height divisible by camera count: "
                    f"height={target_h}, num_cameras={num_cameras}."
                )
            tile_h = target_h // num_cameras
            views = [
                self._resize_video_view(video[view_idx], (tile_h, target_w))
                for view_idx in range(num_cameras)
            ]
            return torch.cat(views, dim=-2)

        if concat_layout == "robotwin":
            if num_cameras != 3:
                raise ValueError(
                    f"`concat_multi_camera='robotwin'` requires exactly 3 cameras, got {num_cameras}."
                )
            if target_h % 3 != 0 or target_w % 2 != 0:
                raise ValueError(
                    "robotwin camera layout requires target height divisible by 3 and width divisible by 2: "
                    f"height={target_h}, width={target_w}."
                )
            bottom_h = target_h // 3
            top_h = target_h - bottom_h
            half_w = target_w // 2
            cam_top = self._resize_video_view(video[0], (top_h, target_w))
            cam_left = self._resize_video_view(video[1], (bottom_h, half_w))
            cam_right = self._resize_video_view(video[2], (bottom_h, half_w))
            bottom = torch.cat([cam_left, cam_right], dim=-1)
            return torch.cat([cam_top, bottom], dim=-2)

        raise ValueError(
            f"Invalid concat_multi_camera: {concat_layout}. "
            "Expected one of: auto, horizontal, vertical, robotwin."
        )

    def _get(self, idx: int) -> dict[str, Any]:
        sample_idx = idx
        sample = None
        for attempt in range(self.max_padding_retry + 1):
            sample = self.lerobot_dataset[sample_idx]
            if not self.skip_padding_as_possible:
                break
            action_is_pad = sample["action_is_pad"]
            image_is_pad = sample["image_is_pad"]
            proprio_is_pad = sample["proprio_is_pad"]
            has_pad = bool(action_is_pad.any().item()) or bool(image_is_pad.any().item()) or bool(proprio_is_pad.any().item())
            if not has_pad or attempt >= self.max_padding_retry:
                break
            sample_idx = np.random.randint(len(self.lerobot_dataset))

        if sample is None:
            raise RuntimeError(f"Failed to load sample at index {idx}")

        image_is_pad = sample["image_is_pad"]
        video = sample["pixel_values"]
        camera_view_mask = sample.get("camera_view_mask", None)
        num_cameras = 1
        if video.ndim == 5:
            video = video[:, self.video_sample_indices, :, :, :]
            num_cameras, t_video, channels, height, width = video.shape
        else:
            if video.ndim != 4:
                raise ValueError(f"Expected video to have 4 or 5 dims, got {tuple(video.shape)}")
            video = video[self.video_sample_indices, :, :, :]
            t_video, channels, height, width = video.shape
        image_is_pad = image_is_pad[self.video_sample_indices]

        video = video.view(num_cameras, t_video, channels, height, width)
        future_3d_images = None
        future_3d_img_masks = None
        if self.return_future_3d_images:
            target_index = self.future_3d_target_index
            if target_index < 0:
                target_index = t_video + target_index
            if target_index < 0 or target_index >= t_video:
                raise IndexError(
                    f"future_3d_target_index={self.future_3d_target_index} is out of bounds for {t_video} video frames."
                )
            future_3d_images = video[:, target_index].clone()
            target_valid = ~image_is_pad[target_index].to(dtype=torch.bool)
            if camera_view_mask is None:
                future_3d_img_masks = torch.full(
                    (num_cameras,),
                    bool(target_valid.item()),
                    dtype=torch.bool,
                    device=future_3d_images.device,
                )
            else:
                future_3d_img_masks = camera_view_mask.to(
                    device=future_3d_images.device,
                    dtype=torch.bool,
                ) & target_valid.to(device=future_3d_images.device)
        target_video_size = resolve_wsa_large_video_size(
            num_cameras,
            self.video_size,
            self.standardize_video_size_by_cameras,
        )
        concat_layout = resolve_wsa_large_concat_layout(num_cameras, self.concat_multi_camera)
        video = self._concat_camera_views(video, num_cameras, target_video_size, concat_layout)
        video = self.normalize_transform(video)
        video = video.permute(1, 0, 2, 3)

        action = sample["action"]
        proprio = sample["proprio"][:-1, :]
        if video.shape[1] <= 1:
            raise ValueError(f"`video` must have at least 2 frames, got shape {tuple(video.shape)}")
        if action.shape[0] % (video.shape[1] - 1) != 0:
            raise ValueError(
                f"`action` horizon must be divisible by `video` transitions, got {action.shape[0]} and {video.shape[1] - 1}"
            )

        task = self.override_instruction if self.override_instruction is not None else sample["instruction"]
        task = str(task).strip()
        instruction = DEFAULT_PROMPT.format(task=task)
        data = {
            "video": video,
            "action": action,
            "proprio": proprio,
            "prompt": instruction,
            "image_is_pad": image_is_pad,
            "action_is_pad": sample["action_is_pad"],
            "proprio_is_pad": sample["proprio_is_pad"],
        }
        if "action_dim_is_pad" in sample:
            data["action_dim_is_pad"] = sample["action_dim_is_pad"]
        if "proprio_dim_is_pad" in sample:
            data["proprio_dim_is_pad"] = sample["proprio_dim_is_pad"]
        if SAMPLE_ACTION_LOSS_MASK in sample:
            data[SAMPLE_ACTION_LOSS_MASK] = sample[SAMPLE_ACTION_LOSS_MASK]
        if future_3d_images is not None and future_3d_img_masks is not None:
            data["future_3d_images"] = future_3d_images
            data["future_3d_img_masks"] = future_3d_img_masks

        if self.text_embedding_cache_dir is not None:
            context, context_mask = self._get_cached_text_context(instruction)
            data["context"] = context
            data["context_mask"] = context_mask
        return data

    def _get_cached_text_context(self, prompt: str):
        if self.text_embedding_cache_dir is None:
            raise ValueError("text_embedding_cache_dir is not set.")
        if self.text_embedding_cache_max_entries > 0 and prompt in self._cached_text_contexts:
            self._cached_text_contexts.move_to_end(prompt)
            return self._cached_text_contexts[prompt]
        os.makedirs(self.text_embedding_cache_dir, exist_ok=True)
        cache_path = build_text_embedding_cache_path(
            self.text_embedding_cache_dir,
            prompt,
            self.context_len,
        )
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Missing text embedding cache: {cache_path}. "
                "Precompute text embeddings first, or set policy.load_text_encoder=true "
                "and leave dataset.text_embedding_cache_dir unset."
            )
        payload = torch.load(cache_path, map_location="cpu")
        context = payload["context"]
        context_mask = payload["mask"].bool()
        if context.ndim != 2:
            raise ValueError(f"Cached `context` must be 2D [L, D], got shape {tuple(context.shape)} in {cache_path}")
        if context_mask.ndim != 1:
            raise ValueError(f"Cached `mask` must be 1D [L], got shape {tuple(context_mask.shape)} in {cache_path}")
        if context.shape[0] != self.context_len or context_mask.shape[0] != self.context_len:
            raise ValueError(
                f"Cached context len mismatch: expected {self.context_len}, got {context.shape[0]} and {context_mask.shape[0]} in {cache_path}"
            )
        context = context.detach().cpu()
        context_mask = context_mask.detach().cpu()
        context[~context_mask] = 0.0
        if not context.is_contiguous():
            context = context.contiguous()
        if not context_mask.is_contiguous():
            context_mask = context_mask.contiguous()
        if self.text_embedding_cache_max_entries > 0:
            self._cached_text_contexts[prompt] = (context, context_mask)
            self._cached_text_contexts.move_to_end(prompt)
            while len(self._cached_text_contexts) > self.text_embedding_cache_max_entries:
                self._cached_text_contexts.popitem(last=False)
        return context, context_mask

    @staticmethod
    def _prepare_value_for_memory_cache(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu()
            return tensor if tensor.is_contiguous() else tensor.contiguous()
        if isinstance(value, dict):
            return {
                key: WSALargeRobotVideoDatasetV3._prepare_value_for_memory_cache(child)
                for key, child in value.items()
            }
        if isinstance(value, list):
            return [
                WSALargeRobotVideoDatasetV3._prepare_value_for_memory_cache(child)
                for child in value
            ]
        if isinstance(value, tuple):
            return tuple(
                WSALargeRobotVideoDatasetV3._prepare_value_for_memory_cache(child)
                for child in value
            )
        return value

    @staticmethod
    def _cached_value_nbytes(value: Any) -> int:
        if isinstance(value, torch.Tensor):
            return int(value.numel() * value.element_size())
        if isinstance(value, dict):
            return sum(WSALargeRobotVideoDatasetV3._cached_value_nbytes(child) for child in value.values())
        if isinstance(value, (list, tuple)):
            return sum(WSALargeRobotVideoDatasetV3._cached_value_nbytes(child) for child in value)
        if isinstance(value, str):
            return len(value.encode("utf-8"))
        return 0

    def preload_in_memory(self) -> None:
        if self._memory_cache is not None:
            return

        num_samples = len(self)
        progress_interval = _env_int("LEROBOT_WSA_LARGE_CACHE_LOG_INTERVAL", 100)
        start_time = time.perf_counter()
        cached_samples: list[dict[str, Any]] = []
        cached_bytes = 0

        logger.info("WSA_Large cache_in_memory=True: preloading %d samples into CPU memory.", num_samples)
        for idx in range(num_samples):
            sample = self._get(idx)
            sample = self._prepare_value_for_memory_cache(sample)
            cached_samples.append(sample)
            cached_bytes += self._cached_value_nbytes(sample)
            if progress_interval > 0 and ((idx + 1) % progress_interval == 0 or idx + 1 == num_samples):
                logger.info(
                    "WSA_Large memory cache progress: %d/%d samples, %.1f MiB cached.",
                    idx + 1,
                    num_samples,
                    cached_bytes / (1024**2),
                )

        self._memory_cache = cached_samples
        elapsed = time.perf_counter() - start_time
        logger.info(
            "WSA_Large memory cache ready: %d samples, %.1f MiB tensors, %.1f s.",
            len(cached_samples),
            cached_bytes / (1024**2),
            elapsed,
        )

    def _load_uncached(self, idx: int):
        try:
            data = self._get(idx)
        except Exception as exc:
            print(f"Error processing sample idx {idx}: {exc}. Returning a random sample instead.")
            print(traceback.format_exc())
            random_idx = np.random.randint(len(self))
            data = self._get(random_idx)
        return data

    def __getitem__(self, idx: int):
        if self._memory_cache is not None:
            return self._memory_cache[idx]
        return self._load_uncached(idx)


def build_wsa_large_processor(
    cfg: WSALargeDatasetConfig,
    image_transforms=None,
) -> WSALargeProcessor:
    resize_shape = cfg.processor_resize_shape
    if resize_shape is None:
        resize_shape = tuple(cfg.image_shapes[0][1:])

    num_output_cameras = int(cfg.processor_num_output_cameras)
    action_output_dim = int(cfg.processor_action_output_dim)
    proprio_output_dim = int(cfg.processor_proprio_output_dim)

    train_image_transforms = {
        key: [
            ToTensor(),
            tv_transforms.Resize(size=list(resize_shape)),
            *([image_transforms] if image_transforms is not None else []),
        ]
        for key in cfg.image_keys
    }
    val_image_transforms = {
        key: [
            ToTensor(),
            tv_transforms.Resize(size=list(resize_shape)),
        ]
        for key in cfg.image_keys
    }
    action_state_transforms = None
    if cfg.action_mode == "delta":
        action_state_transforms = [MaskedDeltaActionTransform(cfg.processor_delta_action_dim_mask)]

    return WSALargeProcessor(
        shape_meta=cfg.shape_meta,
        num_obs_steps=cfg.num_frames,
        num_output_cameras=num_output_cameras,
        action_output_dim=action_output_dim,
        proprio_output_dim=proprio_output_dim,
        action_state_transforms=action_state_transforms,
        use_stepwise_action_norm=cfg.processor_use_stepwise_action_norm,
        norm_default_mode=cfg.processor_norm_default_mode,
        norm_exception_mode=None,
        action_state_merger=ConcatLeftAlign(
            action_target_dim=action_output_dim,
            state_target_dim=proprio_output_dim,
        ),
        train_transforms=train_image_transforms,
        val_transforms=val_image_transforms,
        use_zh_instruction=cfg.processor_use_zh_instruction,
        delta_action_dim_mask=cfg.processor_delta_action_dim_mask,
    )


def build_wsa_large_dataset(
    cfg: WSALargeDatasetConfig,
    stats_cache_path: str | None = None,
    is_training_set: bool = True,
) -> WSALargeRobotVideoDatasetV3:
    image_transforms = ImageTransforms(cfg.image_transforms) if cfg.image_transforms.enable else None
    processor = None if cfg.pretrain_multi_embodiment else build_wsa_large_processor(
        cfg,
        image_transforms=image_transforms,
    )
    if cfg.pretrain_multi_embodiment and not cfg.use_external_stats:
        raise ValueError("WSA_Large multi-embodiment pretraining requires dataset.use_external_stats=true.")
    if cfg.pretrain_multi_embodiment and not cfg.external_stats_root:
        raise ValueError("WSA_Large multi-embodiment pretraining requires dataset.external_stats_root.")
    dataset_dirs = resolve_wsa_large_dataset_dirs(cfg)
    normalization_stats_path = stats_cache_path or cfg.normalization_stats_path
    use_lerobot_meta_stats = False
    if not cfg.use_external_stats and cfg.normalization_stats_path is None:
        normalization_stats_path = None
        use_lerobot_meta_stats = True
    return WSALargeRobotVideoDatasetV3(
        dataset_dirs=dataset_dirs,
        shape_meta=cfg.shape_meta,
        num_frames=cfg.num_frames,
        video_size=cfg.video_size,
        standardize_video_size_by_cameras=cfg.standardize_video_size_by_cameras,
        camera_key=cfg.camera_key,
        processor=processor,
        text_embedding_cache_dir=cfg.text_embedding_cache_dir,
        text_embedding_cache_max_entries=cfg.text_embedding_cache_max_entries,
        cache_in_memory=cfg.cache_in_memory,
        context_len=cfg.context_len,
        normalization_stats_path=normalization_stats_path,
        use_lerobot_meta_stats=use_lerobot_meta_stats,
        val_set_proportion=cfg.val_set_proportion,
        is_training_set=is_training_set,
        global_sample_stride=cfg.global_sample_stride,
        action_video_freq_ratio=cfg.action_video_freq_ratio,
        skip_padding_as_possible=cfg.skip_padding_as_possible,
        max_padding_retry=cfg.max_padding_retry,
        concat_multi_camera=cfg.concat_multi_camera,
        override_instruction=cfg.override_instruction,
        video_backend=cfg.video_backend,
        return_future_3d_images=cfg.return_future_3d_images,
        future_3d_target_index=cfg.future_3d_target_index,
        pretrain_multi_embodiment=cfg.pretrain_multi_embodiment,
        max_action_dim=cfg.processor_action_output_dim,
        max_state_dim=cfg.processor_proprio_output_dim,
        norm_default_mode=cfg.processor_norm_default_mode,
        external_stats_root=cfg.external_stats_root,
        action_mode=cfg.action_mode,
        dataset_weights=cfg.dataset_sampling_weights or None,
        image_transforms=image_transforms,
    )
