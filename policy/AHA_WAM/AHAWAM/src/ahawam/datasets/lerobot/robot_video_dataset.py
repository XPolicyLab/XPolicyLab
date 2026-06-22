import hashlib
import os
from collections import OrderedDict
from typing import Optional
import numpy as np
import traceback
import torch
import torchvision.transforms.functional as transforms_F

from omegaconf import DictConfig, OmegaConf

from hydra.utils import instantiate
from .base_lerobot_dataset import BaseLerobotDataset
from .utils.normalizer import save_dataset_stats_to_json, load_dataset_stats_from_json
from ..dataset_utils import ResizeSmallestSideAspectPreserving, CenterCrop, Normalize
from ahawam.utils.logging_config import get_logger
from ahawam.utils import misc

logger = get_logger(__name__)


DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"


def _is_main_process_without_init() -> bool:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return int(os.environ.get("RANK", "0")) == 0


class RobotVideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dirs,
        shape_meta,
        num_frames=33,
        video_size=[384, 640],
        camera_key=None,
        processor=None,
        text_embedding_cache_dir=None,
        video_latent_cache_dir: Optional[str] = None,
        context_len=128,
        pretrained_norm_stats=None,
        val_set_proportion=0.05,
        is_training_set=False,
        global_sample_stride=1,
        action_video_freq_ratio: int = 1,
        skip_padding_as_possible: bool = False,
        max_padding_retry: int = 3,
        concat_multi_camera: str = "horizontal",  # "horizontal", "vertical", "robotwin", or None
        override_instruction: Optional[
            str
        ] = None,  # whether to hardcode a specific instruction for all samples, for debugging
        num_history_frames: int = 0,
        history_frame_cache_size: int = 64,
        prepend_episode_first_frame: bool = False,
        max_action_offset: int = 0,
        action_chunk_size: int = 16,
        action_horizon: int = 0,
    ):
        self.video_sample_indices = list(
            range(0, num_frames, action_video_freq_ratio)
        )
        self.max_action_offset = int(max_action_offset)
        if self.max_action_offset < 0:
            raise ValueError(
                f"`max_action_offset` must be >= 0, got {max_action_offset}"
            )
        self.action_chunk_size = int(action_chunk_size)
        if self.action_chunk_size <= 0:
            raise ValueError(
                f"`action_chunk_size` must be positive, got {action_chunk_size}"
            )
        self.action_horizon = (
            int(action_horizon)
            if int(action_horizon) > 0
            else int(num_frames - 1 - self.max_action_offset)
        )
        self._action_offset_enabled = self.max_action_offset > 0
        if self._action_offset_enabled:
            if self.action_horizon <= 0:
                raise ValueError(
                    "`action_horizon` must be positive when action-offset sampling is enabled."
                )
            if self.action_horizon % self.action_chunk_size != 0:
                raise ValueError(
                    f"`action_horizon` ({self.action_horizon}) must be divisible by "
                    f"`action_chunk_size` ({self.action_chunk_size})."
                )
            if self.action_horizon + self.max_action_offset > num_frames - 1:
                raise ValueError(
                    "Offset action window exceeds sampled action horizon: "
                    f"action_horizon={self.action_horizon}, "
                    f"max_action_offset={self.max_action_offset}, "
                    f"num_frames={num_frames}."
                )
        self._chunk_start_offsets = list(
            range(0, self.action_horizon, self.action_chunk_size)
        )
        if self._action_offset_enabled:
            image_sample_indices = sorted(
                set(self.video_sample_indices)
                | set(self._chunk_start_offsets)
                | {
                    offset + chunk_start
                    for offset in range(self.max_action_offset + 1)
                    for chunk_start in self._chunk_start_offsets
                }
            )
        else:
            image_sample_indices = self.video_sample_indices
        self._image_offset_to_sample_position = {
            int(offset): i for i, offset in enumerate(image_sample_indices)
        }
        self.lerobot_dataset = BaseLerobotDataset(
            dataset_dirs=dataset_dirs,
            shape_meta=OmegaConf.to_container(shape_meta, resolve=True),
            obs_size=num_frames,
            action_size=num_frames - 1,
            val_set_proportion=val_set_proportion,
            is_training_set=is_training_set,
            global_sample_stride=global_sample_stride,
            image_sample_indices=image_sample_indices,
        )

        self.num_frames = num_frames
        self.action_video_freq_ratio = action_video_freq_ratio
        self.global_sample_stride = global_sample_stride

        assert (num_frames - 1) % self.action_video_freq_ratio == 0, (
            f"num_frames-1 must be divisible by action_video_freq_ratio, got {num_frames - 1} and {self.action_video_freq_ratio}"
        )
        assert ((num_frames - 1) // self.action_video_freq_ratio) % 4 == 0, (
            f"video frames must be divisible by 4 for tokenization, got {(num_frames - 1) // self.action_video_freq_ratio}"
        )
        self._dataset_returns_sampled_video = True

        self.camera_key = camera_key
        self.lerobot_dataset._set_return_images(True)

        self.video_size = video_size
        self.text_embedding_cache_dir = text_embedding_cache_dir
        self.video_latent_cache_dir = video_latent_cache_dir
        self.context_len = context_len
        self._text_context_cache: OrderedDict[str, tuple[torch.Tensor, torch.Tensor]] = OrderedDict()
        self._text_context_cache_maxsize = 128
        self.skip_padding_as_possible = skip_padding_as_possible
        self.max_padding_retry = max_padding_retry
        self.concat_multi_camera = concat_multi_camera
        self.override_instruction = override_instruction
        self.num_history_frames = int(num_history_frames)
        self.prepend_episode_first_frame = bool(prepend_episode_first_frame)
        self._video_history_valid_len_cache: torch.Tensor | None = None
        self._video_history_valid_len_cache_key: tuple[int, int, int] | None = None
        self._history_frame_cache: OrderedDict[int, torch.Tensor] = OrderedDict()
        self._history_frame_cache_maxsize = max(int(history_frame_cache_size), 0)
        if self.num_history_frames < 0:
            raise ValueError(
                f"`num_history_frames` must be >= 0, got {num_history_frames}"
            )
        if self.video_latent_cache_dir:
            os.makedirs(self.video_latent_cache_dir, exist_ok=True)

        self.resize_transform = ResizeSmallestSideAspectPreserving(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.crop_transform = CenterCrop(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.normalize_transform = Normalize(
            args={"mean": 0.5, "std": 0.5},
        )
        if processor is not None:
            if isinstance(processor, DictConfig):
                processor = instantiate(processor)
            if not pretrained_norm_stats:
                if not is_training_set:
                    raise ValueError(
                        "pretrained_norm_stats must be provided for validation/test sets since we don't want to calculate stats on them."
                    )
                if _is_main_process_without_init():
                    logger.info("Calculating dataset stats for normalization...")
                    dataset_stats = self.lerobot_dataset.get_dataset_stats(processor)
                    work_dir = misc.get_work_dir()
                    if work_dir is None:
                        raise ValueError(
                            "Failed to resolve work directory for dataset stats."
                        )
                    save_dataset_stats_to_json(
                        dataset_stats, os.path.join(work_dir, "dataset_stats.json")
                    )
                else:
                    dataset_stats = None
                if (
                    torch.distributed.is_available()
                    and torch.distributed.is_initialized()
                ):
                    obj_list = [dataset_stats]
                    torch.distributed.broadcast_object_list(obj_list, src=0)
                    dataset_stats = obj_list[0]
            else:
                dataset_stats = load_dataset_stats_from_json(pretrained_norm_stats)
                logger.info(f"Using dataset stats: {pretrained_norm_stats}")
                if _is_main_process_without_init():
                    work_dir = misc.get_work_dir()
                    if work_dir is None:
                        raise ValueError(
                            "Failed to resolve work directory for dataset stats."
                        )
                    save_dataset_stats_to_json(
                        dataset_stats, os.path.join(work_dir, "dataset_stats.json")
                    )

            processor.set_normalizer_from_stats(dataset_stats)
            self.lerobot_dataset.set_processor(processor)

    def __len__(self):
        return len(self.lerobot_dataset)

    def configure_video_history_memory(
        self, *, num_history_frames: int, prepend_episode_first_frame: bool | None = None
    ) -> None:
        if int(num_history_frames) < 0:
            raise ValueError(
                f"`num_history_frames` must be >= 0, got {num_history_frames}"
            )
        self.num_history_frames = int(num_history_frames)
        if prepend_episode_first_frame is not None:
            self.prepend_episode_first_frame = bool(prepend_episode_first_frame)
        self._video_history_valid_len_cache = None
        self._video_history_valid_len_cache_key = None

    def _find_episode_bounds(self, sample_idx: int) -> tuple[int, int]:
        starts = self.lerobot_dataset.episode_data_index["from"]
        ends = self.lerobot_dataset.episode_data_index["to"]
        idx = int(sample_idx)
        matches = ((starts <= idx) & (idx < ends)).nonzero(as_tuple=False)
        if int(matches.numel()) == 0:
            raise IndexError(f"Sample index {idx} is outside episode bounds.")
        ep_idx = int(matches[0].item())
        return int(starts[ep_idx].item()), int(ends[ep_idx].item())

    def get_video_history_valid_len_for_index(self, sample_idx: int) -> int:
        if self.num_history_frames <= 0:
            return 0
        ep_start, _ = self._find_episode_bounds(int(sample_idx))
        horizon_stride = int(self.num_frames - 1) * int(self.global_sample_stride)
        if horizon_stride <= 0:
            raise ValueError(f"Invalid horizon stride: {horizon_stride}")
        available_horizons = max(0, (int(sample_idx) - ep_start) // horizon_stride)
        valid_len = min(int(self.num_history_frames), int(available_horizons))
        if self.prepend_episode_first_frame and int(sample_idx) > ep_start:
            first_frame_already_included = (
                valid_len > 0
                and (int(sample_idx) - horizon_stride * valid_len) == ep_start
            )
            if not first_frame_already_included and valid_len < self.num_history_frames:
                valid_len += 1
        return valid_len

    def get_video_history_valid_len_for_all_indices(self) -> torch.Tensor:
        dataset_len = len(self.lerobot_dataset)
        horizon_stride = int(self.num_frames - 1) * int(self.global_sample_stride)
        if horizon_stride <= 0:
            raise ValueError(f"Invalid horizon stride: {horizon_stride}")
        cache_key = (int(self.num_history_frames), int(horizon_stride), int(dataset_len))
        if (
            self._video_history_valid_len_cache is not None
            and self._video_history_valid_len_cache_key == cache_key
        ):
            return self._video_history_valid_len_cache

        valid_len = torch.zeros(dataset_len, dtype=torch.int16)
        if self.num_history_frames > 0 and dataset_len > 0:
            starts = self.lerobot_dataset.episode_data_index["from"].to(
                device="cpu", dtype=torch.long
            )
            ends = self.lerobot_dataset.episode_data_index["to"].to(
                device="cpu", dtype=torch.long
            )
            for ep_start, ep_end in zip(starts.tolist(), ends.tolist()):
                start = max(0, int(ep_start))
                end = min(dataset_len, int(ep_end))
                if end <= start:
                    continue
                rel = torch.arange(end - start, dtype=torch.long)
                ep_valid_len = torch.div(
                    rel, horizon_stride, rounding_mode="floor"
                ).clamp(max=int(self.num_history_frames))
                if self.prepend_episode_first_frame:
                    bonus = (
                        (rel > 0)
                        & (ep_valid_len < self.num_history_frames)
                        & (rel % horizon_stride != 0)
                    ).to(dtype=torch.long)
                    ep_valid_len = (ep_valid_len + bonus).clamp(
                        max=int(self.num_history_frames)
                    )
                valid_len[start:end] = ep_valid_len.to(dtype=torch.int16)

        self._video_history_valid_len_cache = valid_len
        self._video_history_valid_len_cache_key = cache_key
        return valid_len

    def _get_episode_relative_frame_index(self, sample_idx: int) -> int:
        ep_start, _ = self._find_episode_bounds(int(sample_idx))
        return int(sample_idx) - ep_start

    def _get_video_rope_frame_index(self, sample_idx: int) -> int:
        rope_stride = int(self.action_video_freq_ratio)
        if rope_stride <= 0:
            raise ValueError(f"Invalid action_video_freq_ratio: {rope_stride}")
        return self._get_episode_relative_frame_index(sample_idx) // rope_stride

    def _get_video_latent_frame_offsets(self) -> torch.Tensor:
        temporal_downsample_factor = 4
        rope_stride = int(self.action_video_freq_ratio)
        if rope_stride <= 0:
            raise ValueError(f"Invalid action_video_freq_ratio: {rope_stride}")
        return torch.tensor(
            [
                int(offset) // rope_stride
                for offset in self.video_sample_indices[::temporal_downsample_factor]
            ],
            dtype=torch.long,
        )

    def _process_sampled_video_tensor(self, video: torch.Tensor) -> torch.Tensor:
        num_cameras = 1
        if video.ndim == 5:
            num_cameras, T_video, C, H, W = video.shape
        else:
            assert video.ndim == 4, (
                f"Expected video to have shape [T, C, H, W], but got {video.shape}"
            )
            T_video, C, H, W = video.shape

        video = video.view(num_cameras, T_video, C, H, W)
        if self.concat_multi_camera == "robotwin":
            if num_cameras != 3:
                raise ValueError(
                    f"`concat_multi_camera='robotwin'` requires exactly 3 cameras, got {num_cameras}"
                )
            cam_top = transforms_F.resize(
                video[0],
                size=[256, 320],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )
            cam_left = transforms_F.resize(
                video[1],
                size=[128, 160],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )
            cam_right = transforms_F.resize(
                video[2],
                size=[128, 160],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
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
        return video.permute(1, 0, 2, 3)

    def _process_video_tensor(self, video: torch.Tensor) -> torch.Tensor:
        if bool(getattr(self, "_dataset_returns_sampled_video", False)):
            return self._process_sampled_video_tensor(video)
        if video.ndim == 5:
            video = video[:, self.video_sample_indices, :, :, :]
        else:
            assert video.ndim == 4, (
                f"Expected video to have shape [T, C, H, W], but got {video.shape}"
            )
            video = video[self.video_sample_indices, :, :, :]
        return self._process_sampled_video_tensor(video)

    def _select_sampled_video_offsets(
        self, video: torch.Tensor, offsets: list[int]
    ) -> torch.Tensor:
        positions = [self._image_offset_to_sample_position[int(offset)] for offset in offsets]
        index = torch.as_tensor(positions, dtype=torch.long, device=video.device)
        if video.ndim == 5:
            return video.index_select(1, index)
        if video.ndim != 4:
            raise ValueError(
                f"Expected sampled video to be [T,C,H,W] or [N,T,C,H,W], got {tuple(video.shape)}"
            )
        return video.index_select(0, index)

    def _sample_action_offset(self) -> int:
        if self.max_action_offset <= 0:
            return 0
        return int(np.random.randint(self.max_action_offset + 1))

    def _build_chunk_obs_images(
        self,
        video: torch.Tensor,
        *,
        action_offset: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        offset_indices = [int(action_offset) + idx for idx in self._chunk_start_offsets]
        no_offset_indices = [int(idx) for idx in self._chunk_start_offsets]
        offset_video = self._select_sampled_video_offsets(video, offset_indices)
        no_offset_video = self._select_sampled_video_offsets(video, no_offset_indices)
        return (
            self._process_sampled_video_tensor(offset_video).permute(1, 0, 2, 3).contiguous(),
            self._process_sampled_video_tensor(no_offset_video).permute(1, 0, 2, 3).contiguous(),
        )

    def _resolve_lerobot_dataset_for_global_index(self, sample_idx: int):
        start_idx = 0
        for dataset in self.lerobot_dataset.multi_dataset._datasets:
            dataset_len = int(dataset.num_frames)
            if sample_idx < start_idx + dataset_len:
                return dataset, sample_idx - start_idx
            start_idx += dataset_len
        raise IndexError(f"Sample index {sample_idx} out of bounds.")

    def _get_single_frame_pixel_values(self, sample_idx: int) -> torch.Tensor:
        processor = self.lerobot_dataset.processor
        if processor is None:
            raise ValueError("Processor must be set before loading history frames.")
        dataset, local_idx = self._resolve_lerobot_dataset_for_global_index(
            int(sample_idx)
        )
        item = dataset.hf_dataset[int(local_idx)]
        ep_idx = int(item["episode_index"].item())
        current_ts = float(item["timestamp"].item())
        video_frames = dataset._query_videos(
            {vid_key: [current_ts] for vid_key in dataset.meta.video_keys},
            ep_idx,
        )

        transforms = (
            processor.train_transforms if processor.is_train else processor.val_transforms
        )
        processed_images = []
        for meta in self.lerobot_dataset.image_meta:
            key = meta["key"]
            lerobot_key = meta["lerobot_key"]
            if lerobot_key not in video_frames:
                raise KeyError(
                    f"Missing single-frame history image key `{lerobot_key}`."
                )
            image = video_frames[lerobot_key]
            if image.ndim == 3:
                image = image.unsqueeze(0)
            if image.ndim != 4:
                raise ValueError(
                    f"Expected single-frame image `{lerobot_key}` to be [T,C,H,W], "
                    f"got shape {tuple(image.shape)}."
                )
            image = (image * 255).to(torch.uint8)
            current_transforms = (
                transforms[key] if isinstance(transforms, dict) else transforms
            )
            for trans in current_transforms:
                image = trans(image)
            processed_images.append(image)

        pixel_values = torch.stack(processed_images, dim=0)
        num_output_cameras = int(processor.num_output_cameras)
        if num_output_cameras > int(pixel_values.shape[0]):
            out = torch.zeros(
                (num_output_cameras,) + tuple(pixel_values.shape[1:]),
                device=pixel_values.device,
                dtype=pixel_values.dtype,
            )
            out[: pixel_values.shape[0]] = pixel_values
            pixel_values = out
        elif num_output_cameras < int(pixel_values.shape[0]):
            pixel_values = pixel_values[:num_output_cameras]
        return pixel_values

    def _get_processed_single_frame(self, sample_idx: int) -> torch.Tensor:
        cache_key = int(sample_idx)
        if cache_key in self._history_frame_cache:
            self._history_frame_cache.move_to_end(cache_key)
            return self._history_frame_cache[cache_key].clone()
        pixel_values = self._get_single_frame_pixel_values(cache_key)
        video = self._process_sampled_video_tensor(pixel_values)
        frame = video[:, 0].contiguous()
        if self._history_frame_cache_maxsize > 0:
            self._history_frame_cache[cache_key] = frame
            if len(self._history_frame_cache) > self._history_frame_cache_maxsize:
                self._history_frame_cache.popitem(last=False)
        return frame.clone()

    def _build_video_history_frames(
        self, sample_idx: int, current_frame: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.num_history_frames <= 0:
            return (
                torch.empty(0),
                torch.tensor(0, dtype=torch.long),
                torch.empty(0, dtype=torch.long),
            )
        ep_start, _ = self._find_episode_bounds(sample_idx)
        horizon_stride = int(self.num_frames - 1) * int(self.global_sample_stride)
        history_indices = []
        for offset in range(self.num_history_frames, 0, -1):
            history_idx = int(sample_idx) - horizon_stride * offset
            if history_idx >= ep_start:
                history_indices.append(history_idx)
        valid_len = len(history_indices)

        if self.prepend_episode_first_frame and sample_idx > ep_start:
            if ep_start not in history_indices:
                if valid_len < self.num_history_frames:
                    history_indices.insert(0, ep_start)
                    valid_len += 1
                else:
                    history_indices[0] = ep_start

        frames = [self._get_processed_single_frame(idx) for idx in history_indices]
        if frames:
            pad_frame = frames[0].new_zeros(frames[0].shape)
        elif current_frame is not None:
            pad_frame = current_frame.new_zeros(current_frame.shape)
        else:
            current_frame = self._get_processed_single_frame(sample_idx)
            pad_frame = current_frame.new_zeros(current_frame.shape)
        rope_stride = int(self.action_video_freq_ratio)
        history_frame_indices = [
            (int(history_idx) - ep_start) // rope_stride
            for history_idx in history_indices
        ]
        for _ in range(self.num_history_frames - valid_len):
            frames.insert(0, pad_frame)
            history_frame_indices.insert(0, 0)
        return (
            torch.stack(frames, dim=1),
            torch.tensor(valid_len, dtype=torch.long),
            torch.tensor(history_frame_indices, dtype=torch.long),
        )

    def _get(self, idx):
        sample_idx = idx
        sample = None
        for attempt in range(self.max_padding_retry + 1):
            sample = self.lerobot_dataset[sample_idx]

            if not self.skip_padding_as_possible:
                break

            action_is_pad = sample["action_is_pad"]
            image_is_pad = sample["image_is_pad"]
            proprio_is_pad = sample["proprio_is_pad"]
            has_pad = False
            if bool(action_is_pad.any().item()):
                has_pad = True
            if bool(image_is_pad.any().item()):
                has_pad = True
            if bool(proprio_is_pad.any().item()):
                has_pad = True

            if not has_pad or attempt >= self.max_padding_retry:
                break

            sample_idx = np.random.randint(len(self.lerobot_dataset))

        if sample is None:
            raise RuntimeError(f"Failed to load sample at idx {idx}.")
        image_is_pad = sample["image_is_pad"]

        video = sample["pixel_values"]  # [T, C, H, W] or [num_cameras, T, C, H, W]
        action_offset = self._sample_action_offset()
        chunk_obs_images = None
        chunk_obs_images_no_offset = None
        if self._action_offset_enabled:
            chunk_obs_images, chunk_obs_images_no_offset = self._build_chunk_obs_images(
                video, action_offset=action_offset
            )
            main_positions = [
                self._image_offset_to_sample_position[int(offset)]
                for offset in self.video_sample_indices
            ]
            image_is_pad = image_is_pad[main_positions]
            video = self._select_sampled_video_offsets(video, self.video_sample_indices)
        elif not bool(getattr(self, "_dataset_returns_sampled_video", False)):
            image_is_pad = image_is_pad[self.video_sample_indices]
        video = self._process_video_tensor(video)

        # Proxy (from lerobot):
        #   action: [num_frames-1, action_dim] # start from t0, except the last frame
        #   proprio: [num_frames, proprio_dim] # start from t0 to the last frame, aligned with video frames
        action = sample["action"]  # [T-1, action_dim]
        proprio = sample["proprio"][:-1, :]  # [T-1, state_dim]， to align with action
        if video.shape[1] <= 1:
            raise ValueError(
                f"`video` must have at least 2 frames, got shape {tuple(video.shape)}"
            )
        if action.shape[0] % (video.shape[1] - 1) != 0:
            raise ValueError(
                f"`action` horizon must be divisible by `video` transitions, got {action.shape[0]} and {video.shape[1] - 1}"
            )

        task = sample["instruction"]

        # FIXME
        if self.override_instruction is not None:
            task = self.override_instruction
        instruction = DEFAULT_PROMPT.format(task=task)

        context, context_mask = self._get_cached_text_context(instruction)
        # NOTE: to keep consistent with wan2.2's behavior
        context[~context_mask] = 0.0
        context_mask = torch.ones_like(context_mask)

        data = {
            "video": video,
            "action": action,
            "proprio": proprio,
            "prompt": instruction,
            "context": context,
            "context_mask": context_mask,
            "image_is_pad": image_is_pad,
            "action_is_pad": sample["action_is_pad"],
            "proprio_is_pad": sample["proprio_is_pad"],
            "_sample_idx": sample_idx,
        }
        if self._action_offset_enabled:
            data["action_offset"] = torch.tensor(action_offset, dtype=torch.long)
            data["chunk_obs_images"] = chunk_obs_images
            data["chunk_obs_images_no_offset"] = chunk_obs_images_no_offset
        current_frame_index = self._get_video_rope_frame_index(sample_idx)
        data["video_current_frame_index"] = torch.tensor(
            current_frame_index, dtype=torch.long
        )
        data["video_temporal_position_ids"] = (
            current_frame_index + self._get_video_latent_frame_offsets()
        )
        if self.num_history_frames > 0:
            (
                video_history,
                video_history_valid_len,
                video_history_frame_indices,
            ) = self._build_video_history_frames(
                sample_idx, current_frame=video[:, 0]
            )
            data["video_history"] = video_history
            data["video_history_valid_len"] = video_history_valid_len
            data["video_history_frame_indices"] = video_history_frame_indices
        return data

    def _get_cached_text_context(self, prompt: str):
        if self.text_embedding_cache_dir is None:
            raise ValueError("text_embedding_cache_dir is not set.")
        cache_dir = self.text_embedding_cache_dir
        assert cache_dir is not None
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if hashed in self._text_context_cache:
            self._text_context_cache.move_to_end(hashed)
            ctx, mask = self._text_context_cache[hashed]
            return ctx.clone(), mask.clone()
        cache_path = os.path.join(
            cache_dir, f"{hashed}.t5_len{self.context_len}.wan22ti2v5b.pt"
        )
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Missing text embedding cache: {cache_path}. "
                "Run scripts/precompute_text_embeds.py first."
            )
        payload = torch.load(cache_path, map_location="cpu")
        context = payload["context"]
        context_mask = payload["mask"].bool()
        if context.ndim != 2:
            raise ValueError(
                f"Cached `context` must be 2D [L, D], got shape {tuple(context.shape)} in {cache_path}"
            )
        if context_mask.ndim != 1:
            raise ValueError(
                f"Cached `mask` must be 1D [L], got shape {tuple(context_mask.shape)} in {cache_path}"
            )
        if context.shape[0] != self.context_len:
            raise ValueError(
                f"Cached context_len mismatch: expected {self.context_len}, got {context.shape[0]} in {cache_path}"
            )
        if context_mask.shape[0] != self.context_len:
            raise ValueError(
                f"Cached mask_len mismatch: expected {self.context_len}, got {context_mask.shape[0]} in {cache_path}"
            )

        self._text_context_cache[hashed] = (context, context_mask)
        if len(self._text_context_cache) > self._text_context_cache_maxsize:
            self._text_context_cache.popitem(last=False)
        return context.clone(), context_mask.clone()

    def __getitem__(self, idx):
        try:
            data = self._get(idx)
        except Exception as e:
            print(
                f"Error processing sample idx {idx}: {e}. Returning a random sample instead."
            )
            # trace back
            print(traceback.format_exc())
            random_idx = np.random.randint(len(self))
            data = self._get(random_idx)
        sample_idx = int(data.pop("_sample_idx"))
        if self.video_latent_cache_dir is not None:
            data["video_latent_cache_path"] = os.path.join(
                self.video_latent_cache_dir,
                f"{sample_idx:09d}.pt",
            )
        return data
