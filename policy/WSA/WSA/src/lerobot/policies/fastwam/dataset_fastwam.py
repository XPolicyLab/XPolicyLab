from __future__ import annotations

import os
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torchvision.transforms as tv_transforms
import torchvision.transforms.functional as transforms_f
from torch.utils.data import Dataset

from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

from .configuration_fastwam import FastWAMDatasetConfig
from .core.data.dataset_utils import CenterCrop, Normalize, ResizeSmallestSideAspectPreserving
from .core.data.lerobot.processors.base_processor import BaseProcessor
from .core.data.lerobot.processors.fastwam_processor import FastWAMProcessor
from .core.data.lerobot.transforms.action_state_merger import ConcatLeftAlign
from .core.data.lerobot.transforms.image import ToTensor
from .core.data.lerobot.utils.normalizer import (
    load_dataset_stats_from_json,
    save_dataset_stats_to_json,
)
from .core.utils.logging_config import get_logger
from .text_cache import DEFAULT_PROMPT, build_text_embedding_cache_path

logger = get_logger(__name__)

MAX_GETITEM_ATTEMPT = 5
def _to_plain_dict(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _to_plain_dict(child) for key, child in value.items()}
    return value


def resolve_fastwam_dataset_dirs(cfg: FastWAMDatasetConfig) -> list[str]:
    if cfg.dataset_dirs:
        return list(cfg.dataset_dirs)

    if cfg.repo_id_file is None:
        raise ValueError(
            "FastWAM dataset needs either `dataset.dataset_dirs` or `dataset.repo_id_file`."
        )

    repo_id_file = Path(cfg.repo_id_file)
    if not repo_id_file.is_file():
        raise FileNotFoundError(f"FastWAM dataset repo_id_file does not exist: {repo_id_file}")

    dataset_dirs = []
    with repo_id_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            dataset_dir = line.strip()
            if dataset_dir:
                dataset_dirs.append(dataset_dir)

    if not dataset_dirs:
        raise ValueError(f"FastWAM dataset repo_id_file is empty: {repo_id_file}")

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


class FastWAMMultiLeRobotDatasetV3(Dataset):
    def __init__(
        self,
        dataset_dirs: list[str],
        episodes: dict[str, list[int]] | None = None,
        image_transforms=None,
        delta_timestamps: dict[str, list[float]] | None = None,
        video_backend: str | None = None,
    ) -> None:
        super().__init__()
        self.dataset_dirs = dataset_dirs
        self._datasets: list[LeRobotDataset] = []

        for dataset_dir in dataset_dirs:
            root = Path(dataset_dir)
            repo_id = str(root)
            dataset = LeRobotDataset(
                repo_id=repo_id,
                root=root,
                episodes=episodes.get(dataset_dir) if episodes else None,
                image_transforms=image_transforms,
                delta_timestamps=delta_timestamps,
                video_backend=video_backend,
            )
            self._datasets.append(dataset)

        if len(self._datasets) == 0:
            raise ValueError("At least one dataset directory is required.")

        fps_list = [dataset.fps for dataset in self._datasets]
        if len(set(fps_list)) != 1:
            raise ValueError(f"All dataset_dirs must have the same fps, got {fps_list}")

        self.disabled_features: set[str] = set()
        intersection_features = set(self._datasets[0].features)
        for dataset in self._datasets:
            intersection_features.intersection_update(dataset.features)
        for dataset in self._datasets:
            extra_keys = set(dataset.features).difference(intersection_features)
            self.disabled_features.update(extra_keys)

        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        self.stats = aggregate_stats([dataset.meta.stats for dataset in self._datasets])

    def set_during_training(self, during_training: bool) -> None:
        del during_training

    @property
    def num_frames(self) -> int:
        return sum(dataset.num_frames for dataset in self._datasets)

    @property
    def num_episodes(self) -> int:
        return sum(dataset.num_episodes for dataset in self._datasets)

    def __len__(self) -> int:
        return self.num_frames

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds.")
        start_idx = 0
        dataset_idx = 0
        for dataset in self._datasets:
            if idx >= start_idx + dataset.num_frames:
                start_idx += dataset.num_frames
                dataset_idx += 1
                continue
            break
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


class FastWAMBaseLerobotDatasetV3(Dataset):
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

        metas = [LeRobotDatasetMetadata(repo_id=str(Path(ds_dir)), root=Path(ds_dir)) for ds_dir in dataset_dirs]
        fps_list = [meta.fps for meta in metas]
        if len(set(fps_list)) != 1:
            raise ValueError(f"All dataset_dirs must have the same fps, got {fps_list}")
        fps = fps_list[0]

        self.image_meta = shape_meta["images"]
        self.state_meta = shape_meta["state"]
        self.action_meta = shape_meta["action"]

        delta_timestamps: dict[str, list[float]] = {}
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

        self.multi_dataset = FastWAMMultiLeRobotDatasetV3(
            dataset_dirs=self.dataset_dirs,
            episodes=episodes,
            delta_timestamps=delta_timestamps,
            video_backend=video_backend,
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
                    "Overriding FastWAM image raw_shape for %s from %s to dataset metadata shape %s.",
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
                    "Overriding FastWAM state raw_shape for %s from %s to dataset metadata dim %s.",
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
                    "Overriding FastWAM action raw_shape for %s from %s to dataset metadata dim %s.",
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


class FastWAMRobotVideoDatasetV3(Dataset):
    def __init__(
        self,
        dataset_dirs: list[str],
        shape_meta: dict[str, Any],
        num_frames: int = 33,
        video_size: tuple[int, int] = (224, 448),
        camera_key: str | None = None,
        processor: BaseProcessor | None = None,
        text_embedding_cache_dir: str | None = None,
        context_len: int = 128,
        normalization_stats_path: str | None = None,
        val_set_proportion: float = 0.05,
        is_training_set: bool = False,
        global_sample_stride: int = 1,
        action_video_freq_ratio: int = 1,
        skip_padding_as_possible: bool = False,
        max_padding_retry: int = 3,
        concat_multi_camera: str = "horizontal",
        override_instruction: str | None = None,
        video_backend: str | None = None,
    ) -> None:
        self.lerobot_dataset = FastWAMBaseLerobotDatasetV3(
            dataset_dirs=dataset_dirs,
            shape_meta=shape_meta,
            obs_size=num_frames,
            action_size=num_frames - 1,
            val_set_proportion=val_set_proportion,
            is_training_set=is_training_set,
            global_sample_stride=global_sample_stride,
            video_backend=video_backend,
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

        self.video_size = video_size
        self.text_embedding_cache_dir = text_embedding_cache_dir
        self.context_len = context_len
        self.skip_padding_as_possible = skip_padding_as_possible
        self.max_padding_retry = max_padding_retry
        self.concat_multi_camera = concat_multi_camera
        self.override_instruction = override_instruction

        self.resize_transform = ResizeSmallestSideAspectPreserving(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.crop_transform = CenterCrop(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.normalize_transform = Normalize(args={"mean": 0.5, "std": 0.5})

        self.dataset_stats = None
        if processor is not None:
            stats_path = Path(normalization_stats_path) if normalization_stats_path is not None else None
            if stats_path is not None and stats_path.is_file():
                dataset_stats = load_dataset_stats_from_json(str(stats_path))
                logger.info("Using dataset stats from %s", stats_path)
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
            processor.set_normalizer_from_stats(dataset_stats)
            self.dataset_stats = _to_plain_dict(dataset_stats)
            self.lerobot_dataset.set_processor(processor)

    @property
    def num_episodes(self) -> int:
        return self.lerobot_dataset.multi_dataset.num_episodes

    @property
    def num_frames(self) -> int:
        return self.lerobot_dataset.multi_dataset.num_frames

    @property
    def num_frames_total(self) -> int:
        return self.num_frames

    def __len__(self) -> int:
        return len(self.lerobot_dataset)

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
        if self.concat_multi_camera == "robotwin":
            if num_cameras != 3:
                raise ValueError(
                    f"`concat_multi_camera='robotwin'` requires exactly 3 cameras, got {num_cameras}"
                )
            cam_top = transforms_f.resize(
                video[0], size=[256, 320], interpolation=transforms_f.InterpolationMode.BILINEAR, antialias=True
            )
            cam_left = transforms_f.resize(
                video[1], size=[128, 160], interpolation=transforms_f.InterpolationMode.BILINEAR, antialias=True
            )
            cam_right = transforms_f.resize(
                video[2], size=[128, 160], interpolation=transforms_f.InterpolationMode.BILINEAR, antialias=True
            )
            bottom = torch.cat([cam_left, cam_right], dim=-1)
            video = torch.cat([cam_top, bottom], dim=-2)
        elif num_cameras > 1:
            if self.concat_multi_camera == "horizontal":
                video = torch.cat([video[i] for i in range(num_cameras)], dim=-1)
            elif self.concat_multi_camera == "vertical":
                video = torch.cat([video[i] for i in range(num_cameras)], dim=-2)
            else:
                raise ValueError(
                    f"Invalid concat_multi_camera: {self.concat_multi_camera}. "
                    "Expected one of: horizontal, vertical, robotwin."
                )
        else:
            video = video.squeeze(0)

        video = self.resize_transform(video)
        video = self.crop_transform(video)
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

        if self.text_embedding_cache_dir is not None:
            context, context_mask = self._get_cached_text_context(instruction)
            context[~context_mask] = 0.0
            context_mask = torch.ones_like(context_mask)
            data["context"] = context
            data["context_mask"] = context_mask
        return data

    def _get_cached_text_context(self, prompt: str):
        if self.text_embedding_cache_dir is None:
            raise ValueError("text_embedding_cache_dir is not set.")
        os.makedirs(self.text_embedding_cache_dir, exist_ok=True)
        cache_path = build_text_embedding_cache_path(
            self.text_embedding_cache_dir,
            prompt,
            self.context_len,
        )
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Missing text embedding cache: {cache_path}. "
                "Run `python src/lerobot/scripts/fastwam_precompute_text_embeds.py ...` first, or set policy.load_text_encoder=true "
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
        return context, context_mask

    def __getitem__(self, idx: int):
        try:
            data = self._get(idx)
        except Exception as exc:
            print(f"Error processing sample idx {idx}: {exc}. Returning a random sample instead.")
            print(traceback.format_exc())
            random_idx = np.random.randint(len(self))
            data = self._get(random_idx)
        return data


def build_fastwam_processor(cfg: FastWAMDatasetConfig) -> FastWAMProcessor:
    resize_shape = cfg.processor_resize_shape
    if resize_shape is None:
        resize_shape = tuple(cfg.image_shapes[0][1:])

    num_output_cameras = int(cfg.processor_num_output_cameras)
    action_output_dim = int(cfg.processor_action_output_dim)
    proprio_output_dim = int(cfg.processor_proprio_output_dim)

    image_transforms = {
        key: [
            ToTensor(),
            tv_transforms.Resize(size=list(resize_shape)),
        ]
        for key in cfg.image_keys
    }
    return FastWAMProcessor(
        shape_meta=cfg.shape_meta,
        num_obs_steps=cfg.num_frames,
        num_output_cameras=num_output_cameras,
        action_output_dim=action_output_dim,
        proprio_output_dim=proprio_output_dim,
        action_state_transforms=None,
        use_stepwise_action_norm=cfg.processor_use_stepwise_action_norm,
        norm_default_mode=cfg.processor_norm_default_mode,
        norm_exception_mode=None,
        action_state_merger=ConcatLeftAlign(),
        train_transforms=image_transforms,
        val_transforms=image_transforms,
        use_zh_instruction=cfg.processor_use_zh_instruction,
        delta_action_dim_mask=cfg.processor_delta_action_dim_mask,
    )


def build_fastwam_dataset(cfg: FastWAMDatasetConfig, stats_cache_path: str | None = None) -> FastWAMRobotVideoDatasetV3:
    processor = build_fastwam_processor(cfg)
    dataset_dirs = resolve_fastwam_dataset_dirs(cfg)
    return FastWAMRobotVideoDatasetV3(
        dataset_dirs=dataset_dirs,
        shape_meta=cfg.shape_meta,
        num_frames=cfg.num_frames,
        video_size=cfg.video_size,
        camera_key=cfg.camera_key,
        processor=processor,
        text_embedding_cache_dir=cfg.text_embedding_cache_dir,
        context_len=cfg.context_len,
        normalization_stats_path=stats_cache_path or cfg.normalization_stats_path,
        val_set_proportion=cfg.val_set_proportion,
        is_training_set=True,
        global_sample_stride=cfg.global_sample_stride,
        action_video_freq_ratio=cfg.action_video_freq_ratio,
        skip_padding_as_possible=cfg.skip_padding_as_possible,
        max_padding_retry=cfg.max_padding_retry,
        concat_multi_camera=cfg.concat_multi_camera,
        override_instruction=cfg.override_instruction,
        video_backend=cfg.video_backend,
    )
