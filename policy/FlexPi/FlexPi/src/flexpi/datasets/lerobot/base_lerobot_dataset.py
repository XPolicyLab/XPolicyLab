import json
import random
import torch
import numpy as np
from pathlib import Path
from typing import List, Literal, Dict, Optional, Any, DefaultDict
from tqdm import tqdm
from .lerobot.lerobot_dataset import LeRobotDatasetMetadata, MultiLeRobotDataset

from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback
from flexpi.utils.logging_config import get_logger
from .processors.base_processor import BaseProcessor

logger = get_logger(__name__)

MAX_GETITEM_ATTEMPT = 5

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

        # episode limit (per dataset dir)
        num_episodes: Optional[int] = None,

        # per-frame weighted multi-dataset sampling (training set only). When set,
        # `dataset_weights` is a list parallel to `dataset_dirs` giving the
        # probability of drawing a frame from each dataset, independent of how
        # many frames each holds (e.g. [0.2, 0.8] = 2:8). Consumed by the
        # trainer's WeightedResumableEpochSampler, not here — this class just
        # validates/normalizes them and records per-dataset frame counts.
        # Requires `samples_per_epoch` (number of weighted draws per epoch).
        dataset_weights: Optional[List[float]] = None,
        samples_per_epoch: Optional[int] = None,

        # explicit episode indices to use (overrides train/val split when set)
        episode_ids: Optional[List[int]] = None,

        # task-aware episode selection (robotwin2.0 only)
        # task_names: list of task names resolved against datasets/task_episode_map.json
        # num_episodes_per_task: random-sample N episodes per task with a fixed seed
        #   (defaults to `seed`). Requires task_names. None = all episodes of each task.
        task_names: Optional[List[str]] = None,
        num_episodes_per_task: Optional[int] = None,

        # RGB decode-time subsample. When > 1, non-depth image delta_timestamps
        # are subsampled with `obs_times[::image_sample_stride]` so the decoder
        # (decord/torchcodec/pyav) only decodes the frames the model actually
        # uses. Default 1 = pre-existing behavior (decode all obs_size frames,
        # caller subsamples downstream). Depth path unchanged — depth always
        # decodes only the indices passed by `_decode_per_cam_depth`. The
        # decoded pixels for surviving timestamps are byte-identical to the
        # legacy path; only `image_is_pad` ends up shorter (already-subsampled
        # length instead of obs_size). Downstream consumers detect length to
        # remain idempotent.
        image_sample_stride: int = 1,

        # When True, the inner LeRobot loader resolves each sample's prompt by
        # uniformly sampling one of the available language paraphrases (the
        # 3 annotation columns) instead of always using task_index (paraphrase
        # #1). No effect on datasets lacking those columns. Default off.
        sample_language_annotations: bool = False,
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

        # Depth features use dtype="depth_video" in info.json → lerobot's
        # video_keys filter skips them. We track their timestamps here and
        # decode the .mkv files manually in _get_image.
        self._fps = fps
        self._global_sample_stride = global_sample_stride
        self._past_obs_size = past_obs_size
        self._obs_size = obs_size
        self._depth_delta_timestamps: Dict[str, List[float]] = {}

        self._image_sample_stride = int(image_sample_stride)
        if self._image_sample_stride < 1:
            raise ValueError(f"image_sample_stride must be >= 1; got {image_sample_stride}")
        delta_timestamps = {}
        for meta in self.image_meta:
            key = meta["key"]
            meta["lerobot_key"] = f"observation.images.{key}" if key != "default" else "observation.images"
            obs_times = [
                (t * global_sample_stride) / fps for t in range(-past_obs_size, -past_obs_size + obs_size)
            ]
            if meta.get("is_depth"):
                # depth is decoded manually in _decode_per_cam_depth at the
                # video_sample_indices; full obs_times kept here only because
                # _get_image legacy depth path uses it.
                self._depth_delta_timestamps[meta["lerobot_key"]] = obs_times
            else:
                # RGB: hand lerobot only the timestamps the model will actually
                # consume. Saves ~(stride-1)/stride of decode work when stride > 1.
                if self._image_sample_stride > 1:
                    obs_times = obs_times[::self._image_sample_stride]
                delta_timestamps[meta["lerobot_key"]] = obs_times

        for meta in self.state_meta:
            key = meta["key"]
            meta["lerobot_key"] = f"observation.state.{key}" if key != "default" else "observation.state"
            delta_timestamps[meta["lerobot_key"]] = [
                (t * global_sample_stride) / fps for t in range(-past_obs_size, -past_obs_size + obs_size)
            ]

        for meta in self.action_meta:
            key = meta["key"]
            meta["lerobot_key"] = f"action.{key}" if key != "default" else "action"
            delta_timestamps[meta["lerobot_key"]] = [(t * global_sample_stride) / fps for t in range(-past_action_size, -past_action_size + action_size)]

        # Resolve task_names → episode_ids using the bundled robotwin task map.
        # Leaves episode_ids untouched (and this whole block inactive) when
        # task_names is None, preserving the legacy "all episodes" default.
        # When num_episodes_per_task is also set, the train/val split runs
        # PER TASK here (so every task contributes ≥1 val ep when
        # val_set_proportion >= 1e-6), and the downstream global split is
        # skipped via the `apply_per_task_split` flag.
        apply_per_task_split = (
            task_names is not None
            and num_episodes_per_task is not None
            and val_set_proportion >= 1e-6
        )
        if task_names is not None:
            if episode_ids is not None:
                raise ValueError("Cannot specify both `task_names` and `episode_ids`.")
            task_map_path = Path(__file__).resolve().parent.parent / "task_episode_map.json"
            if not task_map_path.exists():
                raise FileNotFoundError(
                    f"task_episode_map.json not found at {task_map_path}. "
                    "This file is required to resolve `task_names`. If you moved the "
                    "package layout, regenerate it or pass `episode_ids` directly."
                )
            logger.info(f"Loading task_episode_map from {task_map_path}")
            with open(task_map_path) as f:
                task_map = json.load(f)["tasks"]
            resolved_ids: List[int] = []
            pool_size = 0
            for t in task_names:
                if t not in task_map:
                    raise ValueError(
                        f"Unknown task {t!r}. Available: {sorted(task_map.keys())}"
                    )
                task_range = list(range(task_map[t]["episode_start"], task_map[t]["episode_end"] + 1))
                if num_episodes_per_task is not None and num_episodes_per_task < len(task_range):
                    rng = random.Random(seed)
                    rng.shuffle(task_range)
                    task_range = task_range[:num_episodes_per_task]
                pool_size += len(task_range)

                if apply_per_task_split and len(task_range) >= 2:
                    rng_split = np.random.default_rng(seed)
                    shuffled = list(task_range)
                    rng_split.shuffle(shuffled)
                    # Guarantee ≥1 val per task (and ≥1 train) when the task has ≥2 episodes.
                    val_count = max(1, int(round(len(shuffled) * val_set_proportion)))
                    val_count = min(val_count, len(shuffled) - 1)
                    task_range = shuffled[val_count:] if is_training_set else shuffled[:val_count]
                elif apply_per_task_split and len(task_range) == 1:
                    # Can't split a single-episode task — route to train, empty val.
                    task_range = task_range if is_training_set else []

                resolved_ids.extend(task_range)
            episode_ids = resolved_ids
            preview_limit = 5
            if len(task_names) <= preview_limit:
                preview = ", ".join(task_names)
            else:
                preview = ", ".join(task_names[:preview_limit]) + f", ...(+{len(task_names) - preview_limit} more)"
            logger.info(
                f"Resolved {len(task_names)} task_names ({preview}) → pool of {pool_size} episode_ids"
                f"{f' ({num_episodes_per_task}/task, seed={seed})' if num_episodes_per_task is not None else ''}"
                f"{f', per-task train/val split (val_proportion={val_set_proportion})' if apply_per_task_split else ''}"
            )
        elif num_episodes_per_task is not None:
            raise ValueError("`num_episodes_per_task` requires `task_names` to be set.")

        episodes = {}
        if episode_ids is not None:
            # Explicit episode indices provided. If `apply_per_task_split` is set,
            # the split already ran per task above and episode_ids holds just the
            # train OR val slice — use it as-is. Otherwise do a global shuffle+split.
            for meta in metas:
                valid = sorted([i for i in episode_ids if i < meta.total_episodes])
                if apply_per_task_split:
                    episodes.update({meta.repo_id: valid})
                else:
                    rng = np.random.default_rng(seed)
                    rng.shuffle(valid)
                    if val_set_proportion < 1e-6:
                        episodes.update({meta.repo_id: valid})
                    else:
                        split_idx = int(len(valid) * (1 - val_set_proportion))
                        if self.is_training_set:
                            episodes.update({meta.repo_id: valid[:split_idx]})
                        else:
                            episodes.update({meta.repo_id: valid[split_idx:]})
            logger.info(f"Using {sum(len(v) for v in episodes.values())} episode_ids ({'train' if self.is_training_set else 'val'})")
        elif val_set_proportion < 1e-6:
            for meta in metas:
                episodes.update({meta.repo_id: sorted(meta.episodes.keys())})
        else:
            for meta in metas:
                # Use actual episode index keys (not range) to handle datasets with gaps
                episode_indices = sorted(meta.episodes.keys())
                split_idx = int(len(episode_indices) * (1 - val_set_proportion))
                rng = np.random.default_rng(seed)
                rng.shuffle(episode_indices)
                if self.is_training_set:
                    episodes.update({meta.repo_id: episode_indices[:split_idx]})
                else:
                    episodes.update({meta.repo_id: episode_indices[split_idx:]})

        # ── Episode budgeting ────────────────────────────────────────────────
        # num_episodes — flat per-dataset cap: keep the same N episodes for every
        # dataset dir (quick experiments). Multi-dataset *sampling ratios* are no
        # longer expressed by truncating episode pools; they are applied at the
        # sampler level via `dataset_weights` (validated/stored below) so the
        # mixture is independent of each dataset's size.
        if num_episodes is not None:
            for repo_id in episodes:
                episodes[repo_id] = episodes[repo_id][:num_episodes]
            logger.info(f"Limiting to {num_episodes} episodes per dataset dir")

        self.multi_dataset = MultiLeRobotDataset(
            dataset_dirs=self.dataset_dirs,
            episodes=episodes,
            delta_timestamps=delta_timestamps,
        )
        self.multi_dataset.set_sample_language_annotations(sample_language_annotations)

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

        # ── Weighted multi-dataset frame sampling spec (consumed by trainer) ──
        # Per-dataset frame counts (post train/val split + num_episodes cap) so
        # the sampler can map a (dataset, local-frame) draw to a global index.
        self.per_dataset_num_frames = [d.num_frames for d in self.multi_dataset._datasets]
        self.samples_per_epoch = int(samples_per_epoch) if samples_per_epoch is not None else None
        self.dataset_weights = None
        if dataset_weights is not None:
            if len(dataset_weights) != len(self.dataset_dirs):
                raise ValueError(
                    f"`dataset_weights` ({len(dataset_weights)}) must be parallel "
                    f"to `dataset_dirs` ({len(self.dataset_dirs)})."
                )
            weights = [float(w) for w in dataset_weights]
            if any(w < 0 for w in weights) or sum(weights) <= 0:
                raise ValueError(
                    f"`dataset_weights` must be non-negative and sum to > 0; got {weights}."
                )
            if self.samples_per_epoch is None or self.samples_per_epoch <= 0:
                raise ValueError(
                    "`samples_per_epoch` must be a positive int when `dataset_weights` is set."
                )
            wsum = sum(weights)
            self.dataset_weights = [w / wsum for w in weights]  # normalized probabilities
            if self.is_training_set:
                logger.info(
                    "[dataset_weights] per-frame sampling probs %s over %d datasets "
                    "(frame counts %s); samples_per_epoch=%d",
                    [round(w, 4) for w in self.dataset_weights],
                    len(self.dataset_dirs),
                    self.per_dataset_num_frames,
                    self.samples_per_epoch,
                )

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
        if meta.get("is_depth"):
            # depth_video dtype is invisible to lerobot's video_keys filter, so
            # lerobot never loads these frames. Decode the .mkv (FFV1 gray16le)
            # manually here, using the sample's episode_index + stored depth
            # timestamps to pick the same frames the RGB key would have loaded.
            from .utils.depth_codec import decode_depth_mkv_at_timestamps
            from pathlib import Path as _Path
            ds_idx = int(lerobot_sample["dataset_index"]) if "dataset_index" in lerobot_sample else 0
            ep_idx = int(lerobot_sample["episode_index"])
            t0 = float(lerobot_sample["timestamp"])
            deltas = self._depth_delta_timestamps[lerobot_key]
            abs_ts = [t0 + d for d in deltas]
            chunk = ep_idx // 1000
            ds_root = _Path(self.dataset_dirs[ds_idx])
            mkv_path = ds_root / "videos" / f"chunk-{chunk:03d}" / lerobot_key / f"episode_{ep_idx:06d}.mkv"
            depth_mm = decode_depth_mkv_at_timestamps(mkv_path, abs_ts, self._fps)  # (T, H, W) int32
            return depth_mm.unsqueeze(1).to(torch.int32)  # (T, 1, H, W)
        image: torch.Tensor = lerobot_sample[lerobot_key]
        if image.ndim == 3: # time dim will lost when obs_size is 1
            image = image.unsqueeze(0)
        image = (image * 255).to(torch.uint8) # (1, 3, H, W)
        # assert image.shape[1:] == raw_shape, f"Image '{key}' shape {image.shape[1:]} mismatch with {raw_shape}."
        return image

    def _split_lerobot_sample(self, lerobot_sample) -> Dict[str, Any]:
        return lerobot_sample

    def _get_episode_data(self, episode_idx):
        lerobot_sample = self.multi_dataset.get_episode_data(episode_idx)
        lerobot_sample = self._split_lerobot_sample(lerobot_sample)
        state, action = {}, {}
        for meta in self.state_meta:
            s = self._get_state(meta, lerobot_sample)
            state[meta["key"]] = s.unsqueeze(1).float()
        for meta in self.action_meta:
            a = self._get_action(meta, lerobot_sample)
            a = sliding_window_with_replication(a, self.action_size)
            action[meta["key"]] = a.float()
        return {"action": action, "state": state}

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
        # Pick the first non-depth image as the representative for is_pad;
        # depth_video keys are not loaded by lerobot and have no is_pad column.
        _pad_meta = next((m for m in self.image_meta if not m.get("is_depth")), self.image_meta[0])
        sample["image_is_pad"] = lerobot_sample[f"{_pad_meta['lerobot_key']}_is_pad"]

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
        state_min = DefaultDict(list)
        state_max = DefaultDict(list)
        state_mean = DefaultDict(list)
        state_var = DefaultDict(list)
        state_q01 = DefaultDict(list)
        state_q99 = DefaultDict(list)

        action_min = DefaultDict(list)
        action_max = DefaultDict(list)
        action_mean = DefaultDict(list)
        action_var = DefaultDict(list)
        action_q01 = DefaultDict(list)
        action_q99 = DefaultDict(list)

        episodes_num = self.multi_dataset.num_episodes

        def process_episode(episode_idx):
            batch = self._get_episode_data(episode_idx)
            batch = preprocessor.action_state_transform(batch)
            return batch

        multi_thread = True
        if not multi_thread:
            for episode_idx in tqdm(range(episodes_num), desc="Iterating dataset to get normalization"):
                batch = process_episode(episode_idx)
                for meta in self.state_meta:
                    key = meta["key"]
                    cur_state: torch.Tensor = batch["state"][key] # (B, T, dim)
                    state_min[key].append(cur_state.amin(0))
                    state_max[key].append(cur_state.amax(0))
                    state_mean[key].append(cur_state.mean(0))
                    state_var[key].append(cur_state.var(0))
                    state_q01[key].append(torch.quantile(cur_state, 0.01, dim=0, keepdim=False))
                    state_q99[key].append(torch.quantile(cur_state, 0.99, dim=0, keepdim=False))
                for meta in self.action_meta:
                    key = meta["key"]
                    cur_action: torch.Tensor = batch["action"][key] # (B, T, dim)
                    action_min[key].append(cur_action.amin(0))
                    action_max[key].append(cur_action.amax(0))
                    action_mean[key].append(cur_action.mean(0))
                    action_var[key].append(cur_action.var(0))
                    action_q01[key].append(torch.quantile(cur_action, 0.01, dim=0, keepdim=False))
                    action_q99[key].append(torch.quantile(cur_action, 0.99, dim=0, keepdim=False))

        else:
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(process_episode, num) for num in range(episodes_num)]

                for future in tqdm(as_completed(futures), total=episodes_num, desc="Iterating dataset to get normalization"):
                    try:
                        batch = future.result()
                        for meta in self.state_meta:
                            key = meta["key"]
                            cur_state: torch.Tensor = batch["state"][key] # (B, T, dim)
                            state_min[key].append(cur_state.amin(0))
                            state_max[key].append(cur_state.amax(0))
                            state_mean[key].append(cur_state.mean(0))
                            state_var[key].append(cur_state.var(0))
                            state_q01[key].append(torch.quantile(cur_state, 0.01, dim=0, keepdim=False))
                            state_q99[key].append(torch.quantile(cur_state, 0.99, dim=0, keepdim=False))

                        for meta in self.action_meta:
                            key = meta["key"]
                            cur_action: torch.Tensor = batch["action"][key] # (B, T, dim)
                            action_min[key].append(cur_action.amin(0))
                            action_max[key].append(cur_action.amax(0))
                            action_mean[key].append(cur_action.mean(0))
                            action_var[key].append(cur_action.var(0))
                            action_q01[key].append(torch.quantile(cur_action, 0.01, dim=0, keepdim=False))
                            action_q99[key].append(torch.quantile(cur_action, 0.99, dim=0, keepdim=False))

                    except Exception as e:
                        logger.error(f"Error processing episode: {e}")
                        print(traceback.format_exc())
                        raise e

        # assume that each minibatch has equal number of samples
        def get_mean_std(means, vars):
            means = torch.stack(means)
            vars = torch.stack(vars)
            stepwise_mean = means.mean(0)
            stepwise_std = (vars + (means - stepwise_mean) ** 2).mean(0).sqrt()
            global_mean = means.mean((0, 1))
            global_std = (vars + (means - global_mean) ** 2).mean((0, 1)).sqrt()
            return stepwise_mean, stepwise_std, global_mean, global_std

        stats = {"state": DefaultDict(dict), "action": DefaultDict(dict), "num_episodes": episodes_num, "num_transition": self.multi_dataset.num_frames}
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

    # clamped_indices [N, window_size]，x [N, D]
    # out[i, j, :] = x[clamped_indices[i, j], :]
    out = x[clamped_indices]  # [N, window_size, D]

    return out
