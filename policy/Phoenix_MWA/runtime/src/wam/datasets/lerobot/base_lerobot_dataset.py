from dataclasses import dataclass
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Any, Callable, DefaultDict
from tqdm import tqdm
from .lerobot.lerobot_dataset import LeRobotDatasetMetadata, MultiLeRobotDataset

from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback
from wam.utils.logging_config import get_logger
from .processors.base_processor import BaseProcessor

logger = get_logger(__name__)

MAX_GETITEM_ATTEMPT = 5

@dataclass
class _DeltaQueryPlan:
    """One raw query plus role-specific views into shared feature columns."""

    delta_indices: Dict[str, List[int]]
    take_indices: Dict[str, Dict[str, List[int]]]


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
        image_obs_indices: Optional[List[int]] = None,
        load_episode_stats: bool = True,
        check_local_files: bool = True,
        episode_selection: Optional[Dict[str, Any]] = None,
    ):
        assert len(dataset_dirs) > 0, "At least one dataset directory is required"
        assert past_action_size >= 0
        assert past_obs_size >= 0
        assert action_size > 0
        assert obs_size > 0
        
        self.dataset_dirs = dataset_dirs
        self.shape_meta = shape_meta
        self.action_size = action_size
        self.past_action_size = past_action_size
        self.obs_size = obs_size
        self.processor = None  # Will be set externally
        logger.debug(
            "BaseLerobotDataset: loading metadata for %d dataset_dirs (load_episode_stats=%s)",
            len(dataset_dirs),
            load_episode_stats,
        )
        metas = []
        for ds_dir in dataset_dirs:
            ds_root = Path(ds_dir)
            repo_id = ds_dir
            logger.debug("BaseLerobotDataset: loading metadata %s", ds_root)
            meta = LeRobotDatasetMetadata(repo_id=repo_id, root=ds_root, load_episode_stats=load_episode_stats)
            logger.debug(
                "BaseLerobotDataset: metadata ready %s total_episodes=%s total_frames=%s",
                ds_root,
                meta.total_episodes,
                meta.total_frames,
            )
            metas.append(meta)

        fps_list = [m.fps for m in metas]
        assert len(set(fps_list)) == 1, f"All dataset_dirs must have the same fps, got {fps_list}"
        fps = fps_list[0]
        self.fps = float(fps)
        
        self.global_sample_stride = global_sample_stride

        self.val_set_proportion = val_set_proportion
        self.is_training_set = is_training_set
        self.episode_selection = episode_selection

        self.image_meta = shape_meta["images"]
        self.state_meta = shape_meta["state"]
        self.action_meta = shape_meta["action"]

        if image_obs_indices is None:
            image_obs_indices = list(range(obs_size))
        else:
            image_obs_indices = [int(index) for index in image_obs_indices]
            if not image_obs_indices or min(image_obs_indices) < 0 or max(image_obs_indices) >= obs_size:
                raise ValueError(
                    "image_obs_indices must be non-empty offsets within "
                    f"[0, {obs_size}), got {image_obs_indices}"
                )
        self.image_obs_indices = image_obs_indices
        image_obs_offsets = [-past_obs_size + index for index in image_obs_indices]
        for meta in self.image_meta:
            key = meta["key"]
            meta["lerobot_key"] = meta.get("lerobot_key") or (
                f"observation.images.{key}" if key != "default" else "observation.images"
            )
        
        for meta in self.state_meta:
            key = meta["key"]
            meta["lerobot_key"] = meta.get("lerobot_key") or (
                f"observation.state.{key}" if key != "default" else "observation.state"
            )
        
        for meta in self.action_meta:
            key = meta["key"]
            meta["lerobot_key"] = meta.get("lerobot_key") or (
                f"action.{key}" if key != "default" else "action"
            )

        self._default_query_plan = self._build_delta_query_plan(
            image_offsets=image_obs_offsets,
            state_offsets=list(
                range(-past_obs_size, -past_obs_size + obs_size)
            ),
            action_offsets=list(
                range(-past_action_size, -past_action_size + action_size)
            ),
        )
        delta_timestamps = {
            key: [float(index) / float(fps) for index in indices]
            for key, indices in self._default_query_plan.delta_indices.items()
        }


        logger.debug(
            "BaseLerobotDataset: selecting episodes split=is_training_set:%s val_set_proportion=%s",
            self.is_training_set,
            val_set_proportion,
        )
        episodes = {}
        for meta in metas:
            episode_indices = self._select_episode_indices(
                list(range(meta.total_episodes)),
                repo_id=meta.repo_id,
            )
            if val_set_proportion < 1e-6:
                selected = episode_indices
            else:
                split_idx = int(len(episode_indices) * (1 - val_set_proportion))
                rng = np.random.default_rng(seed)
                rng.shuffle(episode_indices)
                if self.is_training_set:
                    selected = episode_indices[:split_idx]
                else:
                    selected = episode_indices[split_idx:]
            episodes.update({meta.repo_id: selected})
            logger.debug(
                "BaseLerobotDataset: selected %d/%d episodes for %s",
                len(selected),
                meta.total_episodes,
                meta.repo_id,
            )

        logger.debug("BaseLerobotDataset: constructing MultiLeRobotDataset")
        self.multi_dataset = MultiLeRobotDataset(
            dataset_dirs=self.dataset_dirs,
            episodes=episodes,
            delta_timestamps=delta_timestamps,
            load_episode_stats=load_episode_stats,
            check_local_files=check_local_files,
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
        logger.info(
            "BaseLerobotDataset: ready num_episodes=%s num_frames=%s",
            self.multi_dataset.num_episodes,
            self.multi_dataset.num_frames,
        )

    @staticmethod
    def _meta_query_offset(meta: Dict[str, Any]) -> int:
        """Return a role's raw-frame offset, rejecting fractional values."""

        value = meta.get("query_offset", 0)
        try:
            offset = int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "query_offset for "
                f"{meta.get('key', '<unknown>')!r} must be an integer, "
                f"got {value!r}"
            ) from exc
        if isinstance(value, float) and not value.is_integer():
            raise ValueError(
                "query_offset for "
                f"{meta.get('key', '<unknown>')!r} must be an integer, "
                f"got {value!r}"
            )
        return offset

    def _build_delta_query_plan(
        self,
        *,
        image_offsets: List[int],
        state_offsets: List[int],
        action_offsets: List[int],
    ) -> _DeltaQueryPlan:
        """Merge shared raw-feature queries without losing role alignment."""

        stride = int(self.global_sample_stride)
        delta_indices: Dict[str, List[int]] = {}
        raw_positions: Dict[str, Dict[int, int]] = {}
        take_indices: Dict[str, Dict[str, List[int]]] = {
            "images": {},
            "state": {},
            "action": {},
        }

        role_specs = (
            ("images", self.image_meta, image_offsets),
            ("state", self.state_meta, state_offsets),
            ("action", self.action_meta, action_offsets),
        )
        for role, metas, offsets in role_specs:
            for meta in metas:
                lerobot_key = str(meta["lerobot_key"])
                query_offset = self._meta_query_offset(meta)
                role_raw_offsets = [
                    int(offset) * stride + query_offset for offset in offsets
                ]
                union = delta_indices.setdefault(lerobot_key, [])
                positions = raw_positions.setdefault(lerobot_key, {})
                role_take: List[int] = []
                for raw_offset in role_raw_offsets:
                    position = positions.get(raw_offset)
                    if position is None:
                        position = len(union)
                        union.append(raw_offset)
                        positions[raw_offset] = position
                    role_take.append(position)
                take_indices[role][str(meta["key"])] = role_take

        return _DeltaQueryPlan(
            delta_indices=delta_indices,
            take_indices=take_indices,
        )

    @staticmethod
    def _take_role_rows(
        value: torch.Tensor,
        query_plan: Optional[_DeltaQueryPlan],
        role: str,
        meta_key: str,
    ) -> torch.Tensor:
        if query_plan is None:
            return value
        return value[query_plan.take_indices[role][str(meta_key)]]

    def _select_episode_indices(
        self,
        episode_indices: List[int],
        repo_id: str,
    ) -> List[int]:
        cfg = self.episode_selection
        if cfg is None:
            return episode_indices
        mode = str(cfg.get("mode", "")).strip().lower()
        if mode in {"", "none", "all"}:
            return episode_indices
        if mode == "periodic_prefix":
            period = int(cfg["period"])
            keep_first = int(cfg["keep_first"])
            offset = int(cfg.get("offset", 0))
            if period <= 0:
                raise ValueError(
                    f"episode_selection.period must be positive, got {period}"
                )
            if keep_first < 0 or keep_first > period:
                raise ValueError(
                    "episode_selection.keep_first must be in [0, period], "
                    f"got {keep_first} for period={period}"
                )
            return [
                episode_idx
                for episode_idx in episode_indices
                if ((int(episode_idx) - offset) % period) < keep_first
            ]
        raise ValueError(
            f"Unsupported episode_selection.mode={mode!r} for repo_id={repo_id!r}. "
            "Expected periodic_prefix."
        )

    def _get_action(
        self,
        meta,
        lerobot_sample,
        query_plan: Optional[_DeltaQueryPlan] = None,
    ) -> torch.Tensor:
        key, lerobot_key, raw_shape = meta["key"], meta["lerobot_key"], meta["raw_shape"]
        action: torch.Tensor = lerobot_sample[lerobot_key] # [T, action_dim]
        action = self._take_role_rows(action, query_plan, "action", key)
        if action.ndim == 1: # for shape of 1, like gripper
            action = action.unsqueeze(-1)
        assert action.shape[-1] == raw_shape, f"Action '{key}' shape {action.shape[-1]} mismatch with meta {raw_shape}."
        return action

    def _get_state(
        self,
        meta,
        lerobot_sample,
        query_plan: Optional[_DeltaQueryPlan] = None,
    ) -> torch.Tensor:
        key, lerobot_key, raw_shape = meta["key"], meta["lerobot_key"], meta["raw_shape"]
        state: torch.Tensor = lerobot_sample[lerobot_key]
        state = self._take_role_rows(state, query_plan, "state", key)
        if state.ndim == 1: # for shape of 1, like gripper
            state = state.unsqueeze(-1)
        # state = state[..., :-1, :]  # use state_{t} as observation_t
        assert state.shape[-1] == raw_shape, f"State '{key}' shape {state.shape[-1]} mismatch with meta {raw_shape}."
        return state
    
    def _get_image(
        self,
        meta,
        lerobot_sample,
        query_plan: Optional[_DeltaQueryPlan] = None,
    ) -> torch.Tensor:
        lerobot_key = meta["lerobot_key"]
        image: torch.Tensor = lerobot_sample[lerobot_key]
        if image.ndim == 3: # time dim will lost when obs_size is 1
            image = image.unsqueeze(0)
        image = self._take_role_rows(
            image, query_plan, "images", meta["key"]
        )
        image = (image * 255).to(torch.uint8) # (1, 3, H, W)
        # For config simplication
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
            query_offset = self._meta_query_offset(meta)
            if query_offset:
                indices = torch.arange(s.shape[0], device=s.device)
                indices.add_(query_offset)
                indices.clamp_(0, max(0, s.shape[0] - 1))
                s = s[indices]
            state[meta["key"]] = s.unsqueeze(1).float()
        for meta in self.action_meta:
            a = self._get_action(meta, lerobot_sample)
            query_offset = self._meta_query_offset(meta)
            if query_offset:
                indices = torch.arange(a.shape[0], device=a.device)
                indices.add_(query_offset)
                indices.clamp_(0, max(0, a.shape[0] - 1))
                a = a[indices]
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

    def get_item(
        self,
        idx: int,
        *,
        delta_indices: Optional[Dict[str, List[int]]] = None,
        delta_indices_factory: Optional[
            Callable[[int], Dict[str, List[int]]]
        ] = None,
        processor_method: str = "preprocess",
        decode_images: bool = True,
    ):
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds {len(self)}.")
        if delta_indices is not None and delta_indices_factory is not None:
            raise ValueError(
                "Pass either delta_indices or delta_indices_factory, not both."
            )

        # Retry with random indices until we successfully load a frame.
        sample_idx = idx
        attempt = 0
        last_exception: Optional[Exception] = None
        while attempt < MAX_GETITEM_ATTEMPT:
            try:
                active_query = (
                    delta_indices_factory(sample_idx)
                    if delta_indices_factory is not None
                    else (
                        self._default_query_plan
                        if delta_indices is None
                        else delta_indices
                    )
                )
                active_query_plan = (
                    active_query
                    if isinstance(active_query, _DeltaQueryPlan)
                    else None
                )
                active_delta_indices = (
                    active_query.delta_indices
                    if isinstance(active_query, _DeltaQueryPlan)
                    else active_query
                )
                lerobot_sample = self.multi_dataset.get_item(
                    sample_idx,
                    delta_indices=active_delta_indices,
                    **({"decode_images": False} if not decode_images else {}),
                )
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
            sample["state"][meta["key"]] = self._get_state(meta, lerobot_sample, active_query_plan)

        for meta in self.action_meta:
            sample["action"][meta["key"]] = self._get_action(meta, lerobot_sample, active_query_plan)

        if decode_images:
            for meta in self.image_meta:
                sample["images"][meta["key"]] = self._get_image(meta, lerobot_sample, active_query_plan)

        action_pad_key = f"{self.action_meta[0]['lerobot_key']}_is_pad"
        state_pad_key = f"{self.state_meta[0]['lerobot_key']}_is_pad"
        image_pad_key = f"{self.image_meta[0]['lerobot_key']}_is_pad"
        sample["action_is_pad"] = self._take_role_rows(
            lerobot_sample[action_pad_key],
            active_query_plan,
            "action",
            self.action_meta[0]["key"],
        )
        sample["state_is_pad"] = self._take_role_rows(
            lerobot_sample[state_pad_key],
            active_query_plan,
            "state",
            self.state_meta[0]["key"],
        )
        sample["image_is_pad"] = self._take_role_rows(
            lerobot_sample[image_pad_key],
            active_query_plan,
            "images",
            self.image_meta[0]["key"],
        )

        sample = self._get_additional_data(sample, lerobot_sample)

        for key in lerobot_sample:
            if key not in sample and "observation" not in key and "action" not in key:
                sample[key] = lerobot_sample[key]

        # Preprocess the sample using the processor
        # for quick data loading
        if self.processor is not None:
            preprocess = getattr(self.processor, str(processor_method), None)
            if not callable(preprocess):
                raise TypeError(
                    f"{type(self.processor).__name__} does not implement "
                    f"processor method {processor_method!r}."
                )
            sample = preprocess(sample)

        return sample

    def get_item_with_offsets(
        self,
        idx: int,
        *,
        image_offsets: List[int],
        state_offsets: List[int],
        action_offsets: List[int],
    ):
        plan = self._build_delta_query_plan(image_offsets=image_offsets,
            state_offsets=state_offsets, action_offsets=action_offsets)
        return self.get_item(idx, delta_indices_factory=lambda _: plan)

    def get_item_with_offset_factory(
        self,
        idx: int,
        *,
        offsets_factory: Callable[
            [int], tuple[List[int], List[int], List[int]]
        ],
        processor_method: str = "preprocess",
        decode_images: bool = True,
    ):
        def build_query(sample_idx):
            images, states, actions = offsets_factory(int(sample_idx))
            return self._build_delta_query_plan(image_offsets=images,
                state_offsets=states, action_offsets=actions)
        return self.get_item(idx, delta_indices_factory=build_query,
                             processor_method=processor_method,
                             decode_images=decode_images)

    def __getitem__(self, idx: int):
        return self.get_item(idx)

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
        state_global_values = DefaultDict(list)

        action_min = DefaultDict(list)
        action_max = DefaultDict(list)
        action_mean = DefaultDict(list)
        action_var = DefaultDict(list)
        action_q01 = DefaultDict(list)
        action_q99 = DefaultDict(list)
        action_global_values = DefaultDict(list)

        episodes_num = self.multi_dataset.num_episodes
        
        def process_episode(episode_idx):
            batch = self._get_episode_data(episode_idx) 
            batch = preprocessor.action_state_transform(batch)
            return batch

        def append_global_values(batch):
            """Collect each original frame exactly once for global statistics.

            Episode actions have shape [num_frames, action_size, dim] after
            ``sliding_window_with_replication``.  Flattening that tensor would
            count overlapping action windows repeatedly and would also count
            replicated tail actions.  The first position of every window is
            the original action for that frame, so use only ``[:, 0]``.
            """
            for meta in self.state_meta:
                key = meta["key"]
                values = batch["state"][key]
                values = values.reshape(-1, values.shape[-1])
                state_global_values[key].append(
                    values.detach().to(device="cpu", dtype=torch.float32).clone()
                )
            for meta in self.action_meta:
                key = meta["key"]
                values = batch["action"][key]
                if values.ndim < 3:
                    raise ValueError(
                        f"Expected episode action {key!r} with shape "
                        f"[num_frames, action_size, dim], got {tuple(values.shape)}"
                    )
                values = values[:, 0, :].reshape(-1, values.shape[-1])
                action_global_values[key].append(
                    values.detach().to(device="cpu", dtype=torch.float32).clone()
                )
        
        multi_thread = True
        if not multi_thread:
            for episode_idx in tqdm(range(episodes_num), desc="Iterating dataset to get normalization"):
                batch = process_episode(episode_idx)
                append_global_values(batch)
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
                        append_global_values(batch)
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

        def get_global_frame_stats(chunks, *, kind, key):
            if not chunks:
                raise ValueError(f"No {kind} values collected for {key!r}")
            values = torch.cat(chunks, dim=0)
            if not bool(torch.isfinite(values).all().item()):
                invalid = int((~torch.isfinite(values)).sum().item())
                raise ValueError(
                    f"Collected {invalid} non-finite {kind} values for {key!r}"
                )
            quantiles = torch.quantile(values, torch.tensor([0.01, 0.99]), dim=0)
            variance, mean = torch.var_mean(values, dim=0, correction=0)
            return {
                "global_min": values.amin(0),
                "global_max": values.amax(0),
                "global_q01": quantiles[0],
                "global_q99": quantiles[1],
                "global_mean": mean,
                "global_std": variance.clamp_min(0.0).sqrt(),
            }

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
            stats["state"][key].update(
                get_global_frame_stats(
                    state_global_values[key], kind="state", key=key
                )
            )

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
            stats["action"][key].update(
                get_global_frame_stats(
                    action_global_values[key], kind="action", key=key
                )
            )

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
