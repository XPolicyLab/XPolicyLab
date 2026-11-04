import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Any, DefaultDict
from tqdm import tqdm
from flexwam.datasets.lerobot.lerobot.lerobot_dataset import LeRobotDatasetMetadata, MultiLeRobotDataset

from concurrent.futures import ThreadPoolExecutor, as_completed
from heapq import heappop, heappush
import traceback
from flexwam.utils.logging_config import get_logger
from flexwam.datasets.lerobot.processors.base_processor import BaseProcessor
from flexwam.datasets.lerobot.utils.normalizer import RunningQuantileStats

logger = get_logger(__name__)

MAX_GETITEM_ATTEMPT = 5


def _resolve_lerobot_key(meta: Dict[str, Any], prefix: str) -> str:
    explicit_key = meta.get("lerobot_key")
    if explicit_key is not None:
        return explicit_key

    key = meta["key"]
    if key == "default":
        return prefix
    return f"{prefix}.{key}"


class BaseLerobotDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dirs: List[str],

        # shapes
        shape_meta: Dict[str, Any],
        action_size: int = 1,
        past_action_size: int = 0, # Excludes the current frame
        obs_size: int = 1, # should be
        past_obs_size: int = 0,

        # train vs val
        val_set_proportion: float = 0.05,
        is_training_set: bool = False,
        seed: int = 42,

        # sampling
        global_sample_stride: int = 1,
    ):
        assert len(dataset_dirs) > 0, "At least one dataset directory is required"
        assert past_action_size == 0
        assert past_obs_size == 0
        assert action_size == obs_size - 1, "In this dataset, action_size should be obs_size - 1"

        self.dataset_dirs = dataset_dirs
        self.shape_meta = shape_meta
        self.action_size = action_size
        self.past_action_size = past_action_size
        self.obs_size = obs_size
        self.processor = None  # Will be set externally
        metas = []
        for ds_dir in dataset_dirs:
            ds_root = Path(ds_dir)
            repo_id = ds_dir
            meta = LeRobotDatasetMetadata(repo_id=repo_id, root=ds_root)
            metas.append(meta)

        fps_list = [m.fps for m in metas]
        assert len(set(fps_list)) == 1, f"All dataset_dirs must have the same fps, got {fps_list}"
        fps = fps_list[0]

        self.global_sample_stride = global_sample_stride

        self.val_set_proportion = val_set_proportion
        self.is_training_set = is_training_set

        self.image_meta = shape_meta["images"]
        self.state_meta = shape_meta["state"]
        self.action_meta = shape_meta["action"]

        delta_timestamps = {}
        for meta in self.image_meta:
            meta["lerobot_key"] = _resolve_lerobot_key(meta, "observation.images")
            delta_timestamps[meta["lerobot_key"]] = [
                (t * global_sample_stride) / fps for t in range(-past_obs_size, -past_obs_size + obs_size)
            ]

        for meta in self.state_meta:
            meta["lerobot_key"] = _resolve_lerobot_key(meta, "observation.state")
            delta_timestamps[meta["lerobot_key"]] = [
                (t * global_sample_stride) / fps for t in range(-past_obs_size, -past_obs_size + obs_size)
            ]

        for meta in self.action_meta:
            meta["lerobot_key"] = _resolve_lerobot_key(meta, "action")
            delta_timestamps[meta["lerobot_key"]] = [(t * global_sample_stride) / fps for t in range(-past_action_size, -past_action_size + action_size)]

        episodes = {}
        if val_set_proportion < 1e-6:
            for meta in metas:
                episodes.update({meta.repo_id: list(range(meta.total_episodes))})
        else:
            for meta in metas:
                split_idx = int(meta.total_episodes * (1 - val_set_proportion))
                # random shuffle episode indices before splitting
                episode_indices = list(range(meta.total_episodes))
                rng = np.random.default_rng(seed)
                rng.shuffle(episode_indices)
                if self.is_training_set:
                    episodes.update({meta.repo_id: [episode_indices[i] for i in range(split_idx)]})
                else:
                    episodes.update({meta.repo_id: [episode_indices[i] for i in range(split_idx, meta.total_episodes)]})

        self.multi_dataset = MultiLeRobotDataset(
            dataset_dirs=self.dataset_dirs,
            episodes=episodes,
            delta_timestamps=delta_timestamps,
        )

        # HACK: lerobot 3.0 will fix this
        episode_data_index = []
        end_index = 0
        for dataset in self.multi_dataset._datasets:
            multi_episode_data_index = {
                "from": dataset.episode_data_index["from"] + end_index,
                "to": dataset.episode_data_index["to"] + end_index,
            }
            episode_data_index.append(multi_episode_data_index)
            end_index = multi_episode_data_index["to"][-1]

        self.episode_data_index = {
            "from": torch.cat([dataset["from"] for dataset in episode_data_index]),
            "to": torch.cat([dataset["to"] for dataset in episode_data_index]),
        }

    def _get_action(self, meta, lerobot_sample) -> torch.Tensor:
        key, lerobot_key, raw_shape = meta["key"], meta["lerobot_key"], meta["raw_shape"]
        action: torch.Tensor = lerobot_sample[lerobot_key] # [T, action_dim]
        if action.ndim == 1: # for shape of 1, like gripper
            action = action.unsqueeze(-1)
        assert action.shape[-1] == raw_shape, f"Action '{key}' shape {action.shape[-1]} mismatch with meta {raw_shape}."
        return action

    def _get_state(self, meta, lerobot_sample) -> torch.Tensor:
        key, lerobot_key, raw_shape = meta["key"], meta["lerobot_key"], meta["raw_shape"]
        state: torch.Tensor = lerobot_sample[lerobot_key]
        if state.ndim == 1: # for shape of 1, like gripper
            state = state.unsqueeze(-1)
        # state = state[..., :-1, :]  # use state_{t} as observation_t
        assert state.shape[-1] == raw_shape, f"State '{key}' shape {state.shape[-1]} mismatch with meta {raw_shape}."
        return state

    def _get_image(self, meta, lerobot_sample) -> torch.Tensor:
        key, lerobot_key, raw_shape = meta["key"], meta["lerobot_key"], meta["raw_shape"]
        image: torch.Tensor = lerobot_sample[lerobot_key]
        if image.ndim == 3: # time dim will lost when obs_size is 1
            image = image.unsqueeze(0)
        image = (image * 255).to(torch.uint8) # (1, 3, H, W)
        # For config simplication
        # assert image.shape[1:] == raw_shape, f"Image '{key}' shape {image.shape[1:]} mismatch with {raw_shape}."
        return image

    def _split_lerobot_sample(self, lerobot_sample) -> Dict[str, Any]:
        return lerobot_sample

    def _get_episode_data(self, episode_idx):
        """Rebuild every training sample window in an episode, matching `__getitem__`.

        LeRobot pads by clamping to the last in-episode frame, which is the same
        as `sliding_window_with_replication`. OpenPI pools mean/std/q01/q99 over
        those sample windows, not over raw episode trajectories.
        """
        lerobot_sample = self.multi_dataset.get_episode_data(episode_idx)
        lerobot_sample = self._split_lerobot_sample(lerobot_sample)
        state, action = {}, {}
        for meta in self.state_meta:
            s = self._get_state(meta, lerobot_sample)
            state[meta["key"]] = sliding_window_with_replication(s, self.obs_size).float()
        for meta in self.action_meta:
            a = self._get_action(meta, lerobot_sample)
            action[meta["key"]] = sliding_window_with_replication(a, self.action_size).float()
        sample = {"action": action, "state": state}
        return sample

    def _set_return_images(self, flag: bool):
        self.return_images = flag
        self.multi_dataset.set_during_training(flag)

    def __len__(self):
        return self.multi_dataset.num_frames

    def _get_additional_data(self, sample, lerobot_sample):
        return sample

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds {len(self)}.")

        # Retry with random indices until we successfully load a frame.
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
                    f"Error loading sample {sample_idx} (attempt {attempt}). "
                    "Retrying with a random index. "
                    f"Error: {err}"
                )
                sample_idx = np.random.randint(len(self))
                print(traceback.format_exc())
        else:
            raise RuntimeError(
                f"Failed to load a valid sample after {MAX_GETITEM_ATTEMPT} attempts "
                f"for index {idx}."
            ) from last_exception

        # Get data from lerobot, organized in nested dict
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

        for key in lerobot_sample:
            if key not in sample and "observation" not in key and "action" not in key:
                sample[key] = lerobot_sample[key]

        # Preprocess the sample using the processor
        # for quick data loading
        if self.processor is not None:
            sample = self.processor.preprocess(sample)

        return sample

    def set_processor(self, processor: BaseProcessor):
        """Set processor instance from external initialization."""
        self.processor = processor
        if self.is_training_set:
            self.processor.train()
        else:
            self.processor.eval()
        return self

    def get_dataset_stats(self, preprocessor: BaseProcessor):
        """Pool normalization stats over training sample windows, matching OpenPI.

        OpenPI `compute_norm_stats` iterates the dataloader after data transforms
        and updates one running histogram. We load by episode only for I/O, then
        expand each episode into the same clamped windows as `__getitem__` and
        update stats in episode order so q01/q99 are not per-episode quantiles.
        """
        state_min = DefaultDict(list)
        state_max = DefaultDict(list)
        state_stepwise_stats = {}
        state_global_stats = {}

        action_min = DefaultDict(list)
        action_max = DefaultDict(list)
        action_stepwise_stats = {}
        action_global_stats = {}

        episodes_num = self.multi_dataset.num_episodes

        def _update_running_stats(stepwise_store, global_store, key, x: torch.Tensor):
            # Match OpenPI: flatten all non-feature dims and weight every vector equally.
            if x.ndim != 3:
                raise ValueError(f"Expected running-stat batch shape [N, T, dim], got {tuple(x.shape)}.")
            if key not in global_store:
                global_store[key] = RunningQuantileStats()
            global_store[key].update(x)

            num_steps = x.shape[1]
            if key not in stepwise_store:
                stepwise_store[key] = [RunningQuantileStats() for _ in range(num_steps)]
            elif len(stepwise_store[key]) != num_steps:
                raise ValueError(
                    f"Running-stat time dimension mismatch for key {key!r}: "
                    f"expected {len(stepwise_store[key])}, got {num_steps}."
                )
            for t, stats_t in enumerate(stepwise_store[key]):
                stats_t.update(x[:, t, :])

        def _stacked_stepwise_quantiles(store):
            q01_list = []
            q99_list = []
            for stats_t in store:
                q01, q99 = stats_t.get_quantiles((0.01, 0.99))
                q01_list.append(q01)
                q99_list.append(q99)
            return torch.stack(q01_list, dim=0), torch.stack(q99_list, dim=0)

        def _stacked_stepwise_mean_std(store):
            mean_list = []
            std_list = []
            for stats_t in store:
                mean, std = stats_t.get_mean_std()
                mean_list.append(mean)
                std_list.append(std)
            return torch.stack(mean_list, dim=0), torch.stack(std_list, dim=0)

        def process_episode(episode_idx):
            batch = self._get_episode_data(episode_idx)
            batch = preprocessor.action_state_transform(batch)
            return batch

        def accumulate_batch(batch):
            for meta in self.state_meta:
                key = meta["key"]
                cur_state: torch.Tensor = batch["state"][key] # (B, T, dim)
                state_min[key].append(cur_state.amin(0))
                state_max[key].append(cur_state.amax(0))
                _update_running_stats(state_stepwise_stats, state_global_stats, key, cur_state)
            for meta in self.action_meta:
                key = meta["key"]
                cur_action: torch.Tensor = batch["action"][key] # (B, T, dim)
                action_min[key].append(cur_action.amin(0))
                action_max[key].append(cur_action.amax(0))
                _update_running_stats(action_stepwise_stats, action_global_stats, key, cur_action)

        next_episode_idx = 0
        pending_batches = []
        with ThreadPoolExecutor() as executor:
            future_to_idx = {
                executor.submit(process_episode, episode_idx): episode_idx
                for episode_idx in range(episodes_num)
            }
            for future in tqdm(
                as_completed(future_to_idx),
                total=episodes_num,
                desc="Computing OpenPI-style norm stats",
            ):
                episode_idx = future_to_idx[future]
                try:
                    heappush(pending_batches, (episode_idx, future.result()))
                except Exception as e:
                    logger.error(f"Error processing episode: {e}")
                    print(traceback.format_exc())
                    raise e
                while pending_batches and pending_batches[0][0] == next_episode_idx:
                    _, batch = heappop(pending_batches)
                    accumulate_batch(batch)
                    next_episode_idx += 1

        if next_episode_idx != episodes_num or pending_batches:
            raise RuntimeError(
                "Failed to accumulate norm stats in episode order: "
                f"next={next_episode_idx}, num_episodes={episodes_num}, pending={len(pending_batches)}."
            )

        stats = {"state": DefaultDict(dict), "action": DefaultDict(dict), "num_episodes": episodes_num, "num_transition": self.multi_dataset.num_frames}
        for meta in self.state_meta:
            key = meta["key"]
            stats["state"][key]["stepwise_min"] = torch.stack(state_min[key]).amin(0)
            stats["state"][key]["stepwise_max"] = torch.stack(state_max[key]).amax(0)
            stats["state"][key]["global_min"] = stats["state"][key]["stepwise_min"].amin(0)
            stats["state"][key]["global_max"] = stats["state"][key]["stepwise_max"].amax(0)
            stepwise_q01, stepwise_q99 = _stacked_stepwise_quantiles(state_stepwise_stats[key])
            stepwise_mean, stepwise_std = _stacked_stepwise_mean_std(state_stepwise_stats[key])
            global_q01, global_q99 = state_global_stats[key].get_quantiles((0.01, 0.99))
            global_mean, global_std = state_global_stats[key].get_mean_std()
            stats["state"][key]["stepwise_q01"] = stepwise_q01
            stats["state"][key]["stepwise_q99"] = stepwise_q99
            stats["state"][key]["global_q01"] = global_q01
            stats["state"][key]["global_q99"] = global_q99
            stats["state"][key]["stepwise_mean"] = stepwise_mean
            stats["state"][key]["stepwise_std"] = stepwise_std
            stats["state"][key]["global_mean"] = global_mean
            stats["state"][key]["global_std"] = global_std

        for meta in self.action_meta:
            key = meta["key"]
            stats["action"][key]["stepwise_min"] = torch.stack(action_min[key]).amin(0)
            stats["action"][key]["stepwise_max"] = torch.stack(action_max[key]).amax(0)
            stats["action"][key]["global_min"] = stats["action"][key]["stepwise_min"].amin(0)
            stats["action"][key]["global_max"] = stats["action"][key]["stepwise_max"].amax(0)
            stepwise_q01, stepwise_q99 = _stacked_stepwise_quantiles(action_stepwise_stats[key])
            stepwise_mean, stepwise_std = _stacked_stepwise_mean_std(action_stepwise_stats[key])
            global_q01, global_q99 = action_global_stats[key].get_quantiles((0.01, 0.99))
            global_mean, global_std = action_global_stats[key].get_mean_std()
            stats["action"][key]["stepwise_q01"] = stepwise_q01
            stats["action"][key]["stepwise_q99"] = stepwise_q99
            stats["action"][key]["global_q01"] = global_q01
            stats["action"][key]["global_q99"] = global_q99
            stats["action"][key]["stepwise_mean"] = stepwise_mean
            stats["action"][key]["stepwise_std"] = stepwise_std
            stats["action"][key]["global_mean"] = global_mean
            stats["action"][key]["global_std"] = global_std

        return stats


def sliding_window_with_replication(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    Construct a sliding-window tensor from the input tensor x (shape: [N, D]).
    The output shape is [N, window_size, D].

    For each starting index i:
        out[i, j, :] =
            x[i + j, :]      if i + j < N
            x[-1, :]         otherwise (replicate the last row when out of bounds)

    Args:
        x (torch.Tensor): Input tensor of shape [N, D]
        window_size (int): Size of the sliding window

    Returns:
        torch.Tensor: Tensor of shape [N, window_size, D]
    """
    assert x.dim() == 2
    assert window_size > 0

    N, D = x.shape

    # shape [N, window_size]
    # indices[i, j] = i + j
    i_indices = torch.arange(N).unsqueeze(1)            # [N, 1]
    j_indices = torch.arange(window_size).unsqueeze(0)  # [1, window_size]
    indices = i_indices + j_indices                     # [N, window_size]

    # N-1
    # torch.clamp  [0, N-1]
    clamped_indices = torch.clamp(indices, min=0, max=N - 1)

    # clamped_indices [N, window_size], x [N, D]
    # out[i, j, :] = x[clamped_indices[i, j], :]
    out = x[clamped_indices]  # [N, window_size, D]

    return out
